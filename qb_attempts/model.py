from __future__ import annotations
from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURES

MODEL_VERSION='attempts-v1.0'
CANDIDATES=('recent_mean','ridge','boosting','opportunity')

def estimator(name):
    if name=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=150.))
    return HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=7,learning_rate=.035,l2_regularization=20.,min_samples_leaf=60,random_state=730,early_stopping=False)

def fit(name,df):
    if name=='recent_mean':return None
    if name=='opportunity':
        a=make_pipeline(StandardScaler(),Ridge(alpha=150.));b=make_pipeline(StandardScaler(),Ridge(alpha=150.))
        a.fit(df[FEATURES],df.plays);b.fit(df[FEATURES],df.attempts/df.plays.clip(lower=1))
        return (a,b)
    obj=estimator(name);obj.fit(df[FEATURES],df.attempts);return obj

def predict(name,obj,df):
    if name=='recent_mean':return df.qb_mean5.to_numpy().clip(12,55)
    if name=='opportunity':return (obj[0].predict(df[FEATURES]).clip(40,85)*obj[1].predict(df[FEATURES]).clip(.2,.8)).clip(12,55)
    return obj.predict(df[FEATURES]).clip(12,55)

def probabilities(mu,residuals,line):
    """Discrete, nonnegative smoothed empirical residual mixture; tails include exits.

    Kernel bandwidth fixed at one attempt; no normal/Poisson variance assumption.
    Integer lines retain explicit push mass.
    """
    r=np.asarray(residuals,dtype=float)
    def cdf(k):
        if k<0:return 0.
        return float(np.mean(norm.cdf((k+.5-mu-r)/1.0)))
    below=int(np.ceil(line))-1;at=int(np.floor(line))
    under=cdf(below);over=1-cdf(at);push=max(0.,1-under-over)
    return over,under,push

def crps(mu,residuals,y):
    # Discrete ranked probability score across all count thresholds, in attempts.
    ks=np.arange(0,90)
    c=norm.cdf((ks[:,None]+.5-mu-np.asarray(residuals)[None,:])).mean(axis=1)
    return float(np.sum((c-(y<=ks))**2))

def point_metrics(y,p):return {'n':len(y),'mae':float(np.abs(y-p).mean()),'rmse':float(np.sqrt(np.mean((y-p)**2))),'bias':float(np.mean(p-y))}

def train_validate(df,output:Path):
    output.mkdir(parents=True,exist_ok=True)
    all_oof=[];summary={}
    # Expanding-season fits; 2019-2021 create distribution training residuals.
    for name in CANDIDATES:
        parts=[]
        for season in range(2019,2026):
            tr=df[df.season<season];te=df[df.season==season].copy()
            if te.empty:continue
            m=fit(name,tr);te['prediction']=predict(name,m,te);te['candidate']=name
            te['residual']=te.attempts-te.prediction;parts.append(te)
        oof=pd.concat(parts,ignore_index=True)
        val=oof[oof.season.between(2022,2024)]
        summary[name]={'development_2022_2024':point_metrics(val.attempts,val.prediction)}
        scores=[]
        for row in val.itertuples():
            residual=oof[oof.season<row.season].residual.to_numpy()
            scores.append(crps(row.prediction,residual,row.attempts))
        summary[name]['development_2022_2024']['crps']=float(np.mean(scores))
        all_oof.append(oof)
    # Predeclared selection criterion: lowest development CRPS (entire distribution).
    chosen=min(CANDIDATES,key=lambda n:summary[n]['development_2022_2024']['crps'])
    oof=pd.concat(all_oof,ignore_index=True)
    selected=oof[oof.candidate==chosen]
    hold=selected[selected.season==2025]
    residuals=selected[selected.season<2025].residual.to_numpy()
    hm=point_metrics(hold.attempts,hold.prediction)
    hm['crps']=float(np.mean([crps(r.prediction,residuals,r.attempts) for r in hold.itertuples()]))
    # Fixed threshold grid is distribution QA, explicitly not historical book lines.
    probs=[];truth=[]
    for r in hold.itertuples():
        for line in (24.5,29.5,34.5,39.5,44.5):
            probs.append(probabilities(r.prediction,residuals,line)[0]);truth.append(int(r.attempts>line))
    p=np.clip(probs,1e-6,1-1e-6);y=np.array(truth)
    hm['fixed_threshold_brier']=float(np.mean((p-y)**2));hm['fixed_threshold_log_loss']=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    hm['coverage_80']=float(np.mean((hold.attempts>=hold.prediction+np.quantile(residuals,.1))&(hold.attempts<=hold.prediction+np.quantile(residuals,.9))))
    hm['week1']=point_metrics(hold[hold.week==1].attempts,hold[hold.week==1].prediction)
    baseline=oof[(oof.candidate=='recent_mean')&(oof.season==2025)]
    bm=point_metrics(baseline.attempts,baseline.prediction)
    br=oof[(oof.candidate=='recent_mean')&(oof.season<2025)].residual.to_numpy()
    bm['crps']=float(np.mean([crps(r.prediction,br,r.attempts) for r in baseline.itertuples()]))
    # Game-block bootstrap reflects QB outcomes correlated within the same matchup.
    errors=hold[['game_id','attempts','prediction']].copy();errors['base_prediction']=baseline.prediction.to_numpy()
    diff=errors.assign(diff=(errors.attempts-errors.prediction).abs()-(errors.attempts-errors.base_prediction).abs()).groupby('game_id')['diff'].mean().to_numpy()
    rng=np.random.default_rng(730);boot=rng.choice(diff,(2000,len(diff)),replace=True).mean(axis=1)
    improvement_ci=np.quantile(boot,[.025,.975]).tolist()
    result={'version':MODEL_VERSION,'chosen':chosen,'selection_rule':'Lowest development 2022–2024 CRPS; 2025 reserved untouched for final evaluation. No historical prop prices available.','candidates':summary,'holdout_2025':hm,'baseline_2025':bm,'mae_difference_95ci':improvement_ci,'historical_market_roi':None,'market_edge_validated':False,'training_rows':len(df),'feature_names':FEATURES,'excluded_information':['same-game results','final game weather','historical closing spread/total','current-game pass volume','realized starting-QB attempt threshold']}
    # Production retains OOF residuals only; never in-sample residuals.
    artifact={'version':MODEL_VERSION,'name':chosen,'estimator':fit(chosen,df),'residuals':selected.residual.to_numpy(),'metrics':result}
    with (output/'model.pkl').open('wb') as f:pickle.dump(artifact,f)
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    oof[['game_id','season','week','player_id','player','attempts','candidate','prediction','residual']].to_csv(output/'oof_predictions.csv',index=False)
    return artifact
