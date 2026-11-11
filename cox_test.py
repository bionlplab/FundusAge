import pandas as pd
import numpy as np
import os
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sksurv.metrics import (
    cumulative_dynamic_auc,
    brier_score, integrated_brier_score
)
from lifelines.utils import concordance_index
from sksurv.util import Surv   

def find_id(df2,base= False):
    id_dict = {}
    df2 = df2.sort_values("ID", ascending=True)
    #print(df2)
    id_list = df2['ID'].tolist()
    risk_list = df2['pred'].tolist()
    for x in range(len(id_list)):
        
        now_id = id_list[x]
        if (base == True and 'QUA' not in now_id): 
            continue
        now_risk = risk_list[x]
        id = now_id.split('/')[0]
        if (' RE ' in now_id): lr ='re'
        else: lr = 'le'
        id = id+'_'+lr
       
        if (id in id_dict.keys() and 'QUA' not in now_id): continue
        else:   id_dict[id] = now_risk
    new_df = pd.DataFrame.from_dict(id_dict, orient='index', columns=['bio_age'])
    new_df.index.name = 'id'
    new_df = new_df.reset_index()
    print(new_df)

    df1 = pd.read_csv('cox_gen_all.csv')
    a_indexed = df1.set_index('id')
    b_indexed = new_df.set_index('id')
    merged = b_indexed.join(a_indexed, how='inner')
    merged.reset_index(inplace=True)

    return merged
def merge(path,tag):
    df1 =  pd.read_csv(os.path.join(path,'fold_1'+'_'+tag+'.csv'))
    df2 =  pd.read_csv(os.path.join(path,'fold_2'+'_'+tag+'.csv'))
    df3 =  pd.read_csv(os.path.join(path,'fold_3'+'_'+tag+'.csv'))
    df4 =  pd.read_csv(os.path.join(path,'fold_4'+'_'+tag+'.csv'))
    df5 =  pd.read_csv(os.path.join(path,'fold_5'+'_'+tag+'.csv'))
    df_all = pd.concat([df1, df2, df3, df4, df5], axis=0)

    df_result = df_all.groupby('ID', as_index=False)['pred'].mean()
    return df_result
def count_id(now_df):
    id_list = now_df['id'].tolist()
    id_set = set()
    for now_id in id_list:
        id = now_id.split('_')[0]
        id_set.add(id)
    return len(id_set),len(id_list)
def cox(train_df, test_df, bio = True):

    # delete images with late amd
    cols = ['times']
    mask = (train_df[cols]>=1).all(axis=1)
    train_df = train_df[mask]
    mask = (test_df[cols] >=1 ).all(axis=1)
    test_df = test_df[mask]

    cols = ['INCPWI','RPEDWI','GEOAWI','DRARWI']
    
    mask = (train_df[cols] < 8).all(axis=1)
    train_df = train_df[mask]
    mask = (test_df[cols] < 8).all(axis=1)
    test_df = test_df[mask]

    train_df['age_gap'] = train_df['bio_age'] - train_df['age']
    test_df['age_gap'] = test_df['bio_age'] - test_df['age']


    all_cols = [c for c in train_df.columns if c not in ("times", "status")]
    exclude_cols = ["Unnamed: 0.1","Unnamed: 0", "g_id",'id', "label", "age", "biomarker","GEOACT",'GEOACS','SUBFF2','NDRUF2','SSRF2','SUBHF2','DRSZWI','DRSOFT','school','race','gender']
   #exclude_cols = ["Unnamed: 0.1","Unnamed: 0", "g_id",'id', "label", "age", "biomarker","GEOACT",'GEOACS','SUBFF2','NDRUF2','SSRF2','SUBHF2','school','race','gender']

    use_cols = [c for c in all_cols if c not in exclude_cols and '_' not in c]
    if (bio==True):
        use_cols.append('bio_age')
    else:
        use_cols.append('age')


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

    cph = CoxPHFitter(penalizer=0.01) #gap+age 0.1 other 0.01
    try:
        cph.fit(train_df, duration_col=duration_col, event_col=event_col, 
            formula=formula_str,show_progress=True)
    except ConvergenceError as e:
        print("Caught ConvergenceError:", str(e))
        return 0.0

    cph.print_summary()
    summary = cph.summary.loc[significant_vars, ["coef", "coef lower 95%", "coef upper 95%"]]
    coef      = summary["coef"]
    ci_lower  = summary["coef lower 95%"]
    ci_upper  = summary["coef upper 95%"]
    print(coef)
    print(ci_lower)
    print(ci_upper)
    test_risk_scores = cph.predict_partial_hazard(test_df)
    train_risk_scores = cph.predict_partial_hazard(train_df)
    ci_test2 = cph.score(test_df, scoring_method="concordance_index")

    print(f"C-index (score method): {ci_test2:.3f}")
    risk_scores = cph.predict_partial_hazard(test_df).values.ravel()
    times  = test_df["times"].values
    events = test_df["status"].values          # 1 = event, 0 = censor
    ci_test = concordance_index(times, -risk_scores, event_observed=events)

    # -------- Bootstrap --------
    B = 1000
    rng = np.random.default_rng(seed=42)
    ci_boot = np.empty(B)

    n = len(test_df)
    for b in range(B):
        idx = rng.integers(0, n, size=n)       
        ci_boot[b] = concordance_index(
            times[idx], -risk_scores[idx], event_observed=events[idx]
        )

    # 95% CI
    ci_lower, ci_upper = np.percentile(ci_boot, [2.5, 97.5])
    
    print(f"Testing C-index = {ci_test:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f})")
    
    train_df['risk'] = train_risk_scores
    test_df['risk'] = test_risk_scores

    y_train = Surv.from_dataframe("status", "times", train_df)
    y_test  = Surv.from_dataframe("status", "times", test_df)

    risk_series = cph.predict_partial_hazard(test_df)   
    test_df = test_df.assign(risk=risk_series)

    
    risk_scores = cph.predict_partial_hazard(test_df).values.ravel()

    t_eval = 5            # in year

    t_grid = np.arange(2 , 8, dtype=float)
    
    aucs, mean_aucs = cumulative_dynamic_auc(
        y_train, y_test, risk_scores, t_grid
    )
    print(aucs)


    n_boot = 1000             # 
    rng    = np.random.default_rng(42)

    auc_boot = np.zeros((n_boot, len(t_grid)))

    for i in range(n_boot):
        idx      = rng.integers(0, len(y_test), len(y_test))  
        y_test_b = y_test[idx]
        scores_b = risk_scores[idx]
        try:
            auc_boot[i], _ = cumulative_dynamic_auc(
                y_train, y_test_b, scores_b, t_grid
            )
        except ValueError:
            auc_boot[i] = np.nan         

    ci_low  = np.nanpercentile(auc_boot,  2.5, axis=0)
    ci_high = np.nanpercentile(auc_boot, 97.5, axis=0)

    # ---------------------------------------------
    # AUC 95% CI
    # ---------------------------------------------
    for t, auc, lo, hi in zip(t_grid, aucs, ci_low, ci_high):
        print(f"AUC @ {t:.0f} yr = {auc:.3f}  (95% CI {lo:.3f} – {hi:.3f})")
    

    surv_funcs = cph.predict_survival_function(test_df, times=[t_eval])

    pred_surv_at_t = surv_funcs.T.values   # shape = (n_test, 1)

    # Brier score
    bs_times, bs_vals = brier_score(
        y_train, y_test, pred_surv_at_t, np.array([t_eval])
    )
    print(bs_vals)
    bs_5yr = bs_vals[0]
    print(f"Brier@5yr = {bs_5yr:.3f}")

    rng      = np.random.default_rng(42)
    n_boot   = 1000

    bs_base_A = np.empty(n_boot)   # Baseline Brier


    for i in range(n_boot):
        idx = rng.integers(0, len(y_test), len(y_test))   
        y_b    = y_test[idx]

        bs_base_A[i] = brier_score(
            y_train, y_b, pred_surv_at_t[idx], np.array([5])
        )[1][0]

    bri_lower, bri_upper = np.percentile(bs_base_A, [2.5, 97.5])

    print(f"Testing Brier Score = {bs_5yr:.3f} (95% CI: {bri_lower:.3f}–{bri_upper:.3f})")
   
    return ci_test2
def merge_cox(now_df):

    df1 = pd.read_csv('cox_gen_all.csv')
    a_indexed = df1.set_index('id')
    b_indexed = now_df.set_index('id')


    merged = b_indexed.join(a_indexed, how='inner')

    merged.reset_index(inplace=True)
    return merged
if __name__ == '__main__':

    test_fold = 0

    
    if (test_fold==0):
        test_df = merge('./result_v3','test')
        train_df =  merge('./result_v3','all')
 
    else:
        test_df = pd.read_csv(os.path.join('./result_v3','fold_'+str(test_fold)+'_test.csv'))
        train_df = pd.read_csv(os.path.join('./result_v3','fold_'+str(test_fold)+'_all.csv'))
    
    hash_test_df = find_id(test_df,base=True)
    hash_train_df = find_id(train_df)
    
    val_cindex = cox(hash_train_df, hash_test_df, bio=True)
    