import pickle
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np
import time
import copy
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from data_id_age_cox import get_id
from dataset_age_cox import Dataset
from loss_surv import CoxPHLoss
import pandas as pd
from vmamba import vmamba_tiny_s1l8,vmamba_base_s2l15,vmamba_small_s2l15,vmamba_small_s1l20
import torch
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import argparse




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NewNetwork(nn.Module):
    def __init__(self, num_classes):
        super(NewNetwork, self).__init__()
        
        # Load the pre-trained DenseNet model
        model_ft = models.densenet201(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
                    nn.Linear(num_ftrs, N_CLASSES),
                    nn.Sigmoid()
                )
        #model_ft.load_state_dict(torch.load(os.path.join(path,'weights','age_pretrain.pth')))

        model_ft.classifier = nn.Identity()

        '''
        for param in model_ft.parameters():
            param.requires_grad = False
        '''
        self.mamba = model_ft

        # 添加非线性激活和全局平均池化
        self.features_mamba = nn.Sequential(
            self.mamba,  # 使用完整的 Mamba 模型
            nn.LeakyReLU(inplace=True),
            # nn.AdaptiveAvgPool2d((1, 1))
        )


        # 初始化最终的分类 MLP
        self.combine_mlp = nn.Sequential(
            nn.Linear(1920, 128),  # Mamba + Demographic + Gen
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),  # Mamba + Demographic + Gen
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x_img):
        # 使用 Mamba 提取特征
        features_mamba = self.features_mamba(x_img)  # 提取特征直到 AdaptiveAvgPool2d
        features_mamba = features_mamba.view(features_mamba.size(0), -1)  # Flatten
        # print(features_mamba.shape)
        # features_mamba = self.mlp(features_mamba)  # 映射特征到目标维度


        # 拼接特征 (Mamba + Demographic + Gen)
        #combined_features = torch.cat((features_mamba, x_demographic, x_gen), dim=1)

        # 最终分类
        output = self.combine_mlp(features_mamba)

        return output

def weighted_binary_cross_entropy(outputs,labels):

    pos_weight = 1 - labels.sum() / (labels.sum() + (1 - labels).sum())
    y_true = torch.clamp(labels, 1e-7, 1 - 1e-7)
    y_pred = torch.clamp(outputs, 1e-7, 1 - 1e-7)
    
    # Compute the binary cross-entropy loss
    log_loss = -(y_true * torch.log(y_pred) * pos_weight + (1 - y_true) * torch.log(1 - y_pred) * (1 - pos_weight))
    
    return torch.mean(log_loss)
def get_train_val_p_id(files, fold, total_num_fold):
    num = len(files)
    test_num = num // total_num_fold

    if fold == total_num_fold:
        val_file = files[((fold-1) * test_num):]
    else:
        val_file = files[((fold-1) * test_num):fold * test_num]
    
    train_file = np.concatenate((files[0:((fold-1) * test_num)], files[(fold * test_num):]), axis=0)  
    return train_file, val_file
def get_train_test_p_id(files, fold, train_fold, total_num_fold):
    
    num = len(files)
    test_num = num // total_num_fold

    if fold == total_num_fold:
        test_file = files[((fold-1) * test_num):]
    else:
        test_file = files[((fold-1) * test_num):fold * test_num]
    
    train_file = np.concatenate((files[0:((fold-1) * test_num)], files[(fold * test_num):]), axis=0)  
    
    train_file, validation_file = get_train_val_p_id(train_file,train_fold,total_num_fold)
    #validation_file = train_file[int(0.8*len(train_file)):]
    #train_file = train_file[0:int(0.8*len(train_file))]


    return train_file, validation_file, test_file


class Counter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_AUCs(gt, pred):
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    return roc_auc_score(gt_np[:,0], pred_np[:,0])
def compute_L1(gt, pred):
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    return np.sum(np.abs(gt_np-pred_np))
def cross_auc(R_a_0, R_b_1): 
    scores = np.array(list(R_a_0.cpu().numpy()) + list(R_b_1.cpu().numpy()))
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    return roc_auc_score(y_true, scores)


def group_auc(labels, outputs, groups):
    
    group0p = []
    group0n = []
    group1p = []
    group1n = []

    for i in range(len(labels)):
        if groups[i] == 0:
            if labels[i][0] == 1:
                group0p.append(i)
            if labels[i][0] == 0:
                group0n.append(i)
        if groups[i] == 1:
            if labels[i][0] == 1:
                group1p.append(i)
            if labels[i][0] == 0:
                group1n.append(i)

                
    groupp = group0p+group1p 
    groupn = group0n+group1n
    outputs_ = outputs.clone().detach().cpu()
    
    try:
        AUC = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        AUC = 1
    try:
        A00 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        A00 = 1
    try:
        A11 = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        A11 = 1
    try:
        A0a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group0p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A0a = 1
    try:
        A1a = cross_auc(torch.index_select(outputs_,0,torch.tensor(group1p)), torch.index_select(outputs_,0,torch.tensor(groupn)))
    except:
        A1a = 1
    try:
        Aa0 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group0n)))
    except:
        Aa0 = 1
    try:
        Aa1 = cross_auc(torch.index_select(outputs_,0,torch.tensor(groupp)), torch.index_select(outputs_,0,torch.tensor(group1n)))
    except:
        Aa1 = 1
                

    group_num = [len(group0p),len(group0n),len(group1p),len(group1n)]
    
    return AUC, A00, A11, A0a, A1a, Aa0, Aa1, group_num




# def train_model(dataloaders,model, criterion, optimizer, scheduler, num_epochs=25):
def train_model(dataloaders, model, criterion, optimizer, name='age_train', train_fold=1, num_epochs=25):
    since = time.time()
    fopen = open(os.path.join(path,'d_cox_train.txt'), "w")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_AUROC_avg = 1000000000.0
    cox_criterion = CoxPHLoss()
    losses = Counter()
    best_c_index = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        
        
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            gt = torch.FloatTensor().to(device)
            pred = torch.FloatTensor().to(device)
            times = torch.FloatTensor().to(device)
            events = torch.IntTensor().to(device)
            losses.reset()
            groups = []
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # Iterate over data.
            t = tqdm(enumerate(dataloaders[phase]),  desc='Loss: **** ', total=len(dataloaders[phase]), bar_format='{desc}{bar}{r_bar}')
            for batch_idx, (inputs, labels, event, group, ages, IDs) in t:
                # if batch_idx == 0:
                #     continue
                # print(torch.isnan(inputs).sum())
                inputs = inputs.to(device)
                labels = labels.to(device)
                event = event.to(device)
                ages = ages.to(device)
                mask = (labels == 12.0)

                #print(inputs.shape, labels.shape)
                # print('the lables is',torch.unique(labels))
               
                # print(len(torch.unique(labels)))
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if (phase =='train'):
                        outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    #print(outputs.shape)
                    cox_age = outputs - ages
                    L1_loss = criterion(outputs, ages)
                    masked_loss = L1_loss * mask.float()
                    L1_loss = masked_loss.sum() / ( mask.float().sum() + 1e-8)

                    cox_loss = cox_criterion(cox_age, labels[:,0], event)

                    loss = 2 * cox_loss + L1_loss
                    #loss = cox_loss
                    #print(cox_loss,L1_loss)
                    
                    gt = torch.cat((gt, ages), 0)
                    pred = torch.cat((pred, outputs.data), 0)
                    groups += group

                    # print('outputs shape',outputs.shape)
                    # print('labels shape', labels.shape)
                    # print('groups shape', group.shape)
                    #loss = criterion(outputs, ages)
                    #print('barrier 1')
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    #print('barrier 2')
                # statistics
                losses.update(loss.data.item(), inputs.size(0))
                t.set_description('Loss: %.3f ' % (losses.avg))
            
            AUCs = losses.avg
            AUROC_avg = AUCs
            #AUC, A00, A11, A0a, A1a, Aa0, Aa1, group_num = group_auc(gt, pred, groups)
            
            if phase == "val":
                
                # scheduler.step(losses.avg)
                c_index = get_val_CIndex(train_loader,val_loader,model)
                if best_c_index < c_index:
                    best_c_index = c_index
                    torch.save(model.state_dict(), os.path.join('/prj0129/hed4006/eye/VMamba/result_p' , name + '_fold_' + str(train_fold) + '.pth'))
                fopen.write('\nEpoch {} \t [{}] : \t {C_index:.3f}\n'.format(epoch, phase, C_index=c_index))
                fopen.write('{} \t {}\n'.format(CLASS_NAMES, AUCs))
                fopen.write('-' * 100)
                    
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, batch_idx + 1, len(dataloaders[phase]), loss=losses))
            print('{} : \t {AUROC_avg:.3f}'.format(phase, AUROC_avg=AUROC_avg))
            if (phase =="val"):
                 print('{} : \t {C_index:.3f}'.format(phase, C_index=c_index))
            '''
            print('AUC',AUC)
            print('A00',A00)
            print('A11',A11)
            print('A0a',A0a)
            print('A1a',A1a)
            print('Aa0',Aa0)
            print('Aa1',Aa1)
            print('Group Num',group_num)
            '''
            fopen.flush()
    fopen.close()
    return model


def test_model(test_loader, model, name='empty',train_fold = 1):
    model.eval()
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    groups = []
    ID = []
    with torch.no_grad():
        for batch_idx, (inputs, labels, event, group, ages, IDs) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            ages = ages.to(device)
            outputs = model(inputs)
            gt = torch.cat((gt, ages), 0)
            pred = torch.cat((pred, outputs.data), 0)
            ID.append(IDs)
            groups += group
    AUCs = compute_L1(gt, pred)
    '''
    AUC, A00, A11, A0a, A1a, Aa0, Aa1, group_num = group_auc(gt, pred, groups)
    print('AUCs',AUCs)
    print('AUC',AUC)
    print('A00',A00)
    print('A11',A11)
    print('A0a',A0a)
    print('A1a',A1a)
    print('Aa0',Aa0)
    print('Aa1',Aa1)
    print('Group Num',group_num)
    '''
    pred1 = pred.cpu()
    pred2 = pred1.numpy()
    gt1 = gt.cpu()
    gt2 = gt1.numpy()
    #np.savez('/prj0129/hed4006/eye/code/AMD/d_age_trained.npz', prediction=pred2, label=gt2, group=groups) 
#     np.savetxt('/prj0129/mil4012/AREDS/result_p/densenet201_sex_amd_balance_cross5.txt', pred2)
    ID = np.concatenate(ID)
    pred = pred.cpu().numpy().squeeze()
    print(pred.shape)
    print(ID.shape)
    df = pd.DataFrame({'ID':ID,'pred':pred})
    if (name == 'empty'): return df
    df.to_csv('fold_'+str(train_fold)+'_'+name+'.csv')
    return df
def find_id(df2):
    id_dict = {}
    id_list = df2['ID'].tolist()
    risk_list = df2['pred'].tolist()
    for x in range(len(id_list)):
        now_id = id_list[x]
        now_risk = risk_list[x]
        id = now_id.split('/')[0]
        if (' RE ' in now_id): lr ='re'
        else: lr = 'le'
        id = id+'_'+lr
        #if (now_id not in id_dict.keys()): continue
        if (id in id_dict.keys()): continue
        else:   id_dict[id] = now_risk

    new_df = pd.DataFrame.from_dict(id_dict, orient='index', columns=['bio_age'])
    new_df.index.name = 'id'
    new_df = new_df.reset_index()
    print(new_df)
    df1 = pd.read_csv('cox_gen_all.csv')
    a_indexed = df1.set_index('id')
    #new = pd.read_csv('fold_3_trans_hashed.csv')
    b_indexed = new_df.set_index('id')


    merged = b_indexed.join(a_indexed, how='inner')

    merged.reset_index(inplace=True)
    #print(merged)
    #merged.to_csv('fold_3_trans_output.csv')
    return merged
def cox(train_df, test_df):
    cols = ['DRSZWI','INCPWI','RPEDWI','GEOAWI','DRSOFT','DRARWI']
    mask = (train_df[cols] < 8).all(axis=1)
    train_df = train_df[mask]

    mask = (test_df[cols] < 8).all(axis=1)
    test_df = test_df[mask]


    all_cols = [c for c in train_df.columns if c not in ("times", "status")]
    exclude_cols = ["Unnamed: 0.1","Unnamed: 0", "g_id",'id', "label", "age", "biomarker","GEOACT",'GEOACS','SUBFF2','NDRUF2','SSRF2','SUBHF2','school','race',]
    use_cols = [c for c in all_cols if c not in exclude_cols and '_' not in c]
    use_cols.append('bio_age')
    print(use_cols)

    significant_vars = [] 
    duration_col = "times"  
    event_col = "status"     

    for var in use_cols:
        
        df_uni = train_df[[duration_col, event_col, var]].copy()


        cph_uni = CoxPHFitter(penalizer=0.001)
        try:
            cph_uni.fit(
            df_uni, 
            duration_col=duration_col, 
            event_col=event_col,
            fit_options={'step_size':0.5}
            )
        except ConvergenceError as e:
            print("Caught ConvergenceError:", str(e))
            return 0.0
        
        
        
        p_value = cph_uni.summary.loc[var, "p"]   
        
        if p_value < 0.05:  
            significant_vars.append(var)
        #cph_uni.print_summary()

    print("Single-variate significant variables:", significant_vars)

    formula_str = " + ".join(significant_vars)
    print("Using formula:", formula_str)


    print(np.sum(test_df['status']==1))

    cph = CoxPHFitter()
    try:
        cph.fit(train_df, duration_col=duration_col, event_col=event_col, 
            formula=formula_str,show_progress=True)
    except ConvergenceError as e:
        print("Caught ConvergenceError:", str(e))
        return 0.0

    cph.print_summary()

    test_risk_scores = cph.predict_partial_hazard(test_df)
    ci_test2 = cph.score(test_df, scoring_method="concordance_index")

    print(f"测试集 C-index (score方法): {ci_test2:.3f}")
    return ci_test2

def get_val_CIndex(train_loader, val_loader, model):
    val_df = test_model(val_loader, model)
    train_df = test_model(train_loader, model)
    print('inference finished')
    hash_val_df = find_id(val_df)
    hash_train_df = find_id(train_df)
    val_cindex = cox(hash_train_df, hash_val_df)
    return val_cindex
      
def load_pretrained_weights(model, pretrained_path, model_ema=None):
    # 加载权重文件
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # 加载模型权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # 加载 EMA 模型权重（如果需要）
    if model_ema is not None:
        key = "model_ema" if "model_ema" in checkpoint else "model"
        if key in checkpoint:
            model_ema.ema.load_state_dict(checkpoint[key], strict=False)

    # 清理显存
    del checkpoint
    torch.cuda.empty_cache()   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='/prj0129/hed4006/eye/VMamba')
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--sweep_count", type=int, default=1)
    parser.add_argument("--train_fold", type=int, default=1)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--pretrain_path", type=str, default='./pretrain')
    parser.add_argument("--arch", type=str, default='vmamba_small_s2l15')



    args = parser.parse_args()

    fold = args.fold
    train_fold = args.train_fold
    total_num_fold = 5
    
    train_sampler = None
    batch_size =96
    workers = 0
    N_CLASSES = 1
    CLASS_NAMES = 'AMD'

    random_seed = 42 
    random.seed(random_seed)
    # Set a fixed seed for NumPy 
    np.random.seed(random_seed) 
    # Set a fixed seed for PyTorch 
    torch.manual_seed(random_seed)
    path = args.path
    path_r = os.path.join(path,'result_p')
    data_path=os.path.join(path,'AMD_224/AMD_224/')
    label_path=os.path.join(path,'AMD_gen_52new.csv')
    
    label_path1 = os.path.join(path,'patient_gen_52.csv')
    tmp = np.loadtxt(label_path1, dtype=str, delimiter=",")
    tmp = tmp[1:] 
    #because the patient (id: 54262 does not has age information, we should not use this patient
#     tmp = np.delete(tmp,2985, axis = 0)

    train_name, validation_name, test_name = get_train_test_p_id(tmp, fold, train_fold, total_num_fold)
    
    #train_path, train_labels, train_groups, train_ages = get_id(data_path,label_path,data_id = train_name, train = True)
    #val_path, val_labels, val_groups, val_ages = get_id(data_path,label_path,data_id = validation_name, train = True)
    test_path, test_labels, test_groups, test_ages = get_id(data_path,label_path,data_id = test_name, train = False)
    
    #train_labels = np.asarray(train_labels, dtype=int)
    #train_groups = np.asarray(train_groups, dtype=int)
    
#     train_labels = train_labels.astype(int)
#     val_labels = val_labels.astype(int)
#     test_labels = test_labels.astype(int)
    
#     train_groups = train_groups.astype(int)
#     val_groups = val_groups.astype(int)
#     test_groups = test_groups.astype(int)

    #print('the len of training group 0 is',len(np.argwhere(train_groups==0)))
    #print('the len of training group 1 is',len(np.argwhere(train_groups==1)))

    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize(299),
            #transforms.CenterCrop(299),
            transforms.RandomRotation(10),
            # transforms.ToPILImage(),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(299),
            #transforms.CenterCrop(299),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    #train_dataset = Dataset(train_path,train_labels,groups=train_groups,ages= train_ages, transform = data_transforms["train"])
    #val_dataset = Dataset(val_path,val_labels,groups=val_groups, ages=val_ages, transform = data_transforms["val"])
    test_dataset = Dataset(test_path,test_labels,groups=test_groups, ages=test_ages, transform = data_transforms["val"])
    
    '''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
        
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
    #                                        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)
    '''
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                           num_workers=workers, pin_memory=True, sampler=train_sampler)

    #dataloaders = {"train": train_loader, "val": val_loader}

    #model_ft = vmamba_tiny_s1l8()
    name = args.arch
    if (name =='vmamba_small_s2l15'):
        model_ft = vmamba_small_s2l15(num_classes=1000)
        num_ftrs = model_ft.num_features
    if (name =='vmamba_tiny_s1l8'):
        model_ft = vmamba_tiny_s1l8(num_classes=1000)
        num_ftrs = model_ft.num_features
    
    #print(model_ft)
    pretrain_path = os.path.join(args.pretrain_path,name +'.pth')
    #pretrain_path = "/prj0129/puw4002/AMD/pretrain_weight/pre.pth"
    #model_ft.classifier = nn.Identity()
    load_pretrained_weights(model_ft, pretrain_path)
    
    print(num_ftrs)
    model_ft.classifier.head = nn.Linear(num_ftrs, 1)
    #print(model_ft)


    #model_ft = NewNetwork(num_classes=2)
    model_ft = model_ft.to(device)  # Move model to GPU
    #print(model_ft)
    #model_ft = torch.nn.DataParallel(model_ft) 


    # model_ft = model_ft.to(device)        # 先 to(device)
    # model_ft = torch.nn.DataParallel(model_ft)  # 然后再 DataParallel

    '''
    model_ft = models.inception_v3(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
                nn.Linear(num_ftrs, N_CLASSES),
                nn.Sigmoid()
            )
    '''
    # model_ft = models.densenet201(pretrained=True)
    # num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Sequential(
    #             nn.Linear(num_ftrs, N_CLASSES),
    #             nn.Sigmoid()
    #         )
    # model_ft.classifier = nn.Linear(num_ftrs, N_CLASSES)
    #model_ft = model_ft.to(device)
    
    # model_ft = torch.nn.DataParallel(model_ft)  # 然后再 DataParallel

    
    criterion = nn.L1Loss(reduction='none')
    
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
    
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', patience=2, eps=1e-08, verbose=True)

    # model_ft = train_model(dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=20)
    
    #model_ft = train_model(dataloaders, model_ft, criterion, optimizer_ft, name, train_fold, num_epochs=100)
    
    # model_ft.load_state_dict(torch.load(os.path.join(path_r, name+'_fold_'+str(train_fold)+'.pth')))
    model_ft.load_state_dict(torch.load(os.path.join(path_r, name+'_fold_'+str(train_fold)+'.pth')))
    #model_ft.load_state_dict(torch.load(os.path.join(path_r, name+'.pth')))


    #model_ft.load_state_dict(torch.load("/prj0129/hed4006/eye/VMamba/result_p/age_cox_train_2.pth"))
    test_model(test_loader,model_ft,'test',train_fold)
    #test_model(val_loader,model_ft,'val',train_fold)
    #test_model(train_loader,model_ft,'train',train_fold)
    #c_index = get_val_CIndex(train_loader,test_loader,model_ft)
    #print('{} : \t {C_index:.3f}'.format('test', C_index=c_index))
