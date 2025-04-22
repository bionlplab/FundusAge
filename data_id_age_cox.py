import os
import numpy as np



def get_id(data_path, label_path, data_id, train=True):
    tmp = np.loadtxt(label_path, dtype=np.str_, delimiter=",")
    tmp = np.delete(tmp, 0, axis=0)
    ID = tmp[:, 0]
    Path = tmp[:, 2]
    lr = tmp[:, 3]
    tmp1 = tmp[:, 4]
    #tmp2 = tmp[:, 5]
    age = tmp[:,6]

    gender = tmp[:, 7]

    
    images_path = []
    labels = []
    groups = []
    years = []
    le = 0
    
    #for gender
    for i in range(len(data_id)):        
        ind = np.argwhere(ID==data_id[i])
        for j in range(len(ind)):
            label = tmp1[int(ind[j])].astype(np.float64)
#             label = float(tmp1[int(ind[j])])
            year = age[int(ind[j])].astype(np.float64)
            label = 13.0 - label
            #if (train==True) and (label>1.0) : continue
            data_paths = (ID[int(ind[j])] + '/' + Path[int(ind[j])])   
            images_path = np.append(images_path,data_paths)
            le += 1
            #if (label>1.0): continue
            '''
            if label >= 10:
                label = 1
            else:
                label = 0
            '''
            labels.append([label])
            years.append([year])
            #year = 
#             if float(gender[int(ind[j])]) == 0:
            if gender[int(ind[j])].astype(np.float64) == 0:
                # groups = np.append(groups,0)
                groups.append(0)
            else:
                # groups = np.append(groups,1)
                groups.append(1)
    return images_path, labels, groups, years    
    
    
#      #for age
#     for i in range(len(data_id)):        
#         ind = np.argwhere(ID==data_id[i])
#         for j in range(len(ind)):
#             data_paths = (ID[int(ind[j])] + '/' + Path[int(ind[j])])   
#             images_path = np.append(images_path,data_paths)
#             le += 1
#             label = tmp1[int(ind[j])].astype(np.float)
#             if label >= 10:
#                 label = 1
#             else:
#                 label = 0
#             labels.append([label])
#             if gender[int(ind[j])].astype(float) < 65:
#                 # groups = np.append(groups,0)
#                 groups.append(0)
#             elif gender[int(ind[j])].astype(float) >= 65 and gender[int(ind[j])].astype(float) < 75:
#                 # groups = np.append(groups,0)
#                 groups.append(1)
#             else:
#                 # groups = np.append(groups,1)
#                 groups.append(2)
    
#     return images_path, labels, groups
    
    
    
    # ## for education
    # for i in range(len(data_id)):        
    #     ind = np.argwhere(ID==data_id[i])
    #     for j in range(len(ind)):
    #         data_paths = (ID[int(ind[j])] + '/' + Path[int(ind[j])])   
    #         images_path = np.append(images_path,data_paths)
    #         le += 1
    #         label = tmp1[int(ind[j])].astype(np.float)
    #         if label >= 10:
    #             label = 1
    #         else:
    #             label = 0
    #         labels.append([label])
    #         if gender[int(ind[j])].astype(float) <= 2:
    #             # groups = np.append(groups,0)
    #             groups.append(0)
    #         else:
    #             # groups = np.append(groups,1)
    #             groups.append(1) 
    
    
                
#     ## for cfh or arms2
#     for i in range(len(data_id)):        
#         ind = np.argwhere(ID==data_id[i])
#         for j in range(len(ind)):
#             data_paths = (ID[int(ind[j])] + '/' + Path[int(ind[j])])   
#             images_path = np.append(images_path,data_paths)
#             le += 1
#             label = tmp1[int(ind[j])].astype(np.float)
#             if label >= 10:
#                 label = 1
#             else:
#                 label = 0
#             labels.append([label])
#             if gender[int(ind[j])].astype(float) == 0:
#                 # groups = np.append(groups,0)
#                 groups.append(0)
#             elif gender[int(ind[j])].astype(float) == 1:
#                 # groups = np.append(groups,0)
#                 groups.append(1)
#             else:
#                 # groups = np.append(groups,1)
#                 groups.append(2)   
#     return images_path, labels, groups
    
    #     # for smk
    # for i in range(len(data_id)):        
    #     ind = np.argwhere(ID==data_id[i])
    #     for j in range(len(ind)):
    #         data_paths = (ID[int(ind[j])] + '/' + Path[int(ind[j])])   
    #         images_path = np.append(images_path,data_paths)
    #         le += 1
    #         label = tmp1[int(ind[j])].astype(np.float)
    #         if label >= 10:
    #             label = 1
    #         else:
    #             label = 0
    #         labels.append([label])
    #         if gender[int(ind[j])].astype(float) == 1:
    #             # groups = np.append(groups,0)
    #             groups.append(0)
    #         elif gender[int(ind[j])].astype(float) == 2:
    #             # groups = np.append(groups,0)
    #             groups.append(1)
    #         else:
    #             # groups = np.append(groups,1)
    #             groups.append(2)                  
    # return images_path, labels, groups