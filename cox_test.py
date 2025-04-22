import pandas as pd
import numpy as np
import os
from d_age_cox import find_id,cox
def merge():
    df1 =  pd.read_csv('fold_1'+'_test.csv')
    df2 =  pd.read_csv('fold_2'+'_test.csv')
    df3 =  pd.read_csv('fold_3'+'_test.csv')
    df4 =  pd.read_csv('fold_4'+'_test.csv')
    df5 =  pd.read_csv('fold_5'+'_test.csv')
    df_all = pd.concat([df1, df2, df3, df4, df5], axis=0)

    df_result = df_all.groupby('ID', as_index=False)['pred'].mean()
    return df_result
if __name__ == '__main__':

    test_fold = 0
    
    if (test_fold==0):
        test_df = merge()
        train_df = pd.read_csv(os.path.join('./csv_result','emsemble_train.csv'))
    else:
        test_df = pd.read_csv(os.path.join('./','fold_'+str(test_fold)+'_test.csv'))
        train_df = pd.read_csv(os.path.join('./csv_result','fold_'+str(test_fold)+'_all.csv'))
    


    hash_test_df = find_id(test_df)
    hash_train_df = find_id(train_df)
    val_cindex = cox(hash_train_df, hash_test_df)
    print('{} : \t {C_index:.3f}'.format('test', C_index=val_cindex))