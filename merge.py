import pandas as pd
import os
import numpy as np

for fold in range(1,6):
    #if (fold==3) or (fold==4): continue
    df1 = pd.read_csv('fold_'+str(fold)+'_train.csv')
    df2 = pd.read_csv('fold_'+str(fold)+'_val.csv')
    df_concat = pd.concat([df1, df2], axis=0, ignore_index=True)
    save_name = 'fold_'+str(fold)+'_all.csv'
    df_concat.to_csv(os.path.join('./csv_result',save_name))
    print(df_concat)
'''
df1 =  pd.read_csv('fold_1'+'_test.csv')
df2 =  pd.read_csv('fold_2'+'_test.csv')
df3 =  pd.read_csv('fold_3'+'_test.csv')
df4 =  pd.read_csv('fold_4'+'_test.csv')
df5 =  pd.read_csv('fold_5'+'_test.csv')
df_all = pd.concat([df1, df2, df3, df4, df5], axis=0)

# 2. 按照 id 分组，对 pred 列求平均
df_result = df_all.groupby('ID', as_index=False)['pred'].mean()

# 3. 查看合并后的结果
print(df_result)
df_result.to_csv('emsemble_test.csv')
'''