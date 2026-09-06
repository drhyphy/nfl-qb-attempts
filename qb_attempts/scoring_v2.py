"""Prospective V2 policy: independent forecast, transparent market blend and stress."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import team_key
from .game_features import game_features
from .game_model import scenario_analysis
from .model import probabilities
from .scoring import name_key, payout, implied, no_vig, ev, fair_odds, bet_to, resolve_quotes, NY_BOOKS

POLICY={'version':'independent-blend-v2','min_ev':.03,'min_robust_ev':0.,'min_other_books':1,'max_offer_outlier':.10,'min_qb_starts':4,'model_weight':.50,'model_weight_limited_history_or_change':.35,'probability_stress':.02,'max_observation_age_minutes':30,'weights_prospectively_tested':True,'weights_historically_validated':False}

def score(quotes,artifact,records,context,now,game_markets):
    if not quotes:return [],[],[]
    enriched=[]
    for q in quotes:
        m=game_markets.get(q['game_id'],{})
        try:
            observed=pd.Timestamp(m['observed_at'])
            if observed.tzinfo is None or observed < now-pd.Timedelta(minutes=POLICY['max_observation_age_minutes']) or observed > now+pd.Timedelta(minutes=2):continue
            margin=float(m['home_expected_margin']);total=float(m['total_line'])
            if not np.isfinite(margin) or not np.isfinite(total) or not 10<total<100 or abs(margin)>40:continue
            # Home field indicator can be zero at neutral venues; use schedule identity.
            margin=margin if team_key(q['team'])==team_key(q['home_team']) else -margin
            enriched.append({**q,'expected_margin':margin,'game_total':total,'game_market':m})
        except (KeyError,ValueError,TypeError):continue
    quotes=enriched
    if not quotes:return [],[],[]
    requests={}
    for q in quotes:requests[(q['game_id'],q['player_id'])]=q
    features=game_features(records,list(requests.values()));features['mean']=artifact['estimator'].predict(features)
    shares=artifact['exposure'].predict(features)
    scenarios=np.array(scenario_analysis(artifact,features)).T
    features['script_shares']=list(shares);features['scenario_means']=list(scenarios)
    info={(r['game_id'],r['player_id']):r for r in features.to_dict('records')}
    residuals=artifact['residuals'];candidates=[]
    for q in quotes:
        if q['book'] not in NY_BOOKS:continue
        peers=[x for x in quotes if x['game_id']==q['game_id'] and x['player_id']==q['player_id'] and x['line']==q['line'] and x['book']!=q['book']]
        # Real books outside the user's state can inform consensus, never recommendations.
        ps=[no_vig(x['over_odds'],x['under_odds']) for x in peers]
        own=no_vig(q['over_odds'],q['under_odds'])
        market=float(np.median(ps)) if ps else own
        f=info[(q['game_id'],q['player_id'])]
        po,pu,push=probabilities(f['mean'],residuals,q['line']);conditional_model=po/(1-push)
        weight=POLICY['model_weight_limited_history_or_change'] if f['qb_games']<12 or f['coach_change'] or f['qb_team_change'] else POLICY['model_weight']
        blend=market+weight*(conditional_model-market)
        ctx=context.get(q['team'],{})
        common=[]
        if len(peers)<POLICY['min_other_books']:common.append('No independent other book at this line')
        if abs(own-market)>POLICY['max_offer_outlier']:common.append('Unusual price: verify source')
        if f['qb_games']<POLICY['min_qb_starts']:common.append('Insufficient NFL starts')
        if q['roster_status']!='ACT':common.append('Roster does not list active QB')
        if not ctx.get('available') or name_key(ctx.get('starter',''))!=name_key(q['player']):common.append('Starting role not independently verified')
        if ctx.get('injuries'):common.append('QB injury designation needs review')
        try:
            observed_context=pd.Timestamp(ctx.get('observed_at'))
            updated_context=pd.Timestamp(ctx.get('source_updated_at'))
            if pd.isna(observed_context) or pd.isna(updated_context) or observed_context.tzinfo is None or updated_context.tzinfo is None:raise ValueError()
            if observed_context<now-pd.Timedelta(hours=2) or observed_context>now+pd.Timedelta(minutes=2) or updated_context<now-pd.Timedelta(hours=48) or updated_context>now+pd.Timedelta(minutes=2):raise ValueError()
            if int(ctx.get('season',0))!=int(q['season']):raise ValueError()
        except (ValueError,TypeError):common.append('Starting context stale or unverified')
        for side,p_m,p_market,p_final,price in [('Over',po,market,blend*(1-push),q['over_odds']),('Under',pu,1-market,(1-blend)*(1-push),q['under_odds'])]:
            conditional=p_final/(1-push)
            # A fixed 2pp miss in the blended forecast; this is not a confidence interval.
            worst=conditional-POLICY['probability_stress']
            robust=ev(max(0,worst)*(1-push),price,push)
            estimate=ev(p_final,price,push)
            reasons=common.copy()
            if estimate<POLICY['min_ev']:reasons.append('Estimated EV below 3%')
            if robust<=POLICY['min_robust_ev']:reasons.append('Edge fails probability stress test')
            if ev(p_m,price,push)<=0:reasons.append('Independent model does not support positive EV')
            quantiles=np.quantile(residuals,[.1,.9])+f['mean']
            candidates.append({**{k:q[k] for k in ['game_id','player_id','player','team','opponent','line','book','source_url','observed_at','source_updated_at']},'kickoff':q['kickoff'].isoformat(),'side':side,'odds':int(price),'mean':float(f['mean']),'interval80':np.clip(quantiles,0,90).tolist(),'p_model':float(p_m),'p_market':float(p_market*(1-push)),'p_final':float(p_final),'p_push':float(push),'ev':float(estimate),'robust_ev':float(robust),'market_only_ev':float(ev(p_market*(1-push),price,push)),'fair_odds':fair_odds(conditional),'bet_to':bet_to(p_final,push,target=POLICY['min_ev']),'independent_model_ev':float(ev(p_m,price,push)),
                'expected_margin':float(f['expected_margin']),'game_total':float(f['game_total']),
                'implied_team_points':float((f['game_total']+f['expected_margin'])/2),
                'game_market':q['game_market'],
                'script_shares':dict(zip(('leading','close','trailing'),map(float,f['script_shares']))),
                'scenario_means':dict(zip(('underdog_7','pickem','favorite_7'),map(float,f['scenario_means']))),
                'state_pass_rates':{k:float(f[k]) for k in ('lead_pass12','neutral_pass12','trail_pass12')},
                'warnings':(['Large model–market disagreement'] if abs(conditional_model-market)>.15 else [])+(['Only one reference book'] if len(peers)==1 else [])+(['Wide peer price range'] if ps and max(ps)-min(ps)>.10 else []),
                'policy_sensitivity':{str(w):{'p_win':float((w*(p_m/(1-push))+(1-w)*p_market)*(1-push)),'ev':float(ev((w*(p_m/(1-push))+(1-w)*p_market)*(1-push),price,push))} for w in (0.,.35,.50,.65,1.)},
                'model_weight':float(weight),'other_books':len(peers),'reference_books':sorted({x['book'] for x in peers}),'status':'qualified' if not reasons else 'held','reasons':reasons,'starting_status':ctx,'feature_flags':{k:f[k] for k in ['coach_change','qb_team_change','qb_games','season_games']}})
    # One side/line/book per quarterback; choose qualified robust EV before raw EV.
    best={}
    for r in sorted(candidates,key=lambda x:(x['status']=='qualified',x['robust_ev'],x['ev']),reverse=True):best.setdefault((r['game_id'],r['player_id']),r)
    watch=sorted(best.values(),key=lambda x:x['ev'],reverse=True)
    selected=[r for r in watch if r['status']=='qualified']
    # At most one QB per game, at most five per slate to limit correlated exposure.
    games=set();recommend=[]
    for r in sorted(selected,key=lambda x:x['robust_ev'],reverse=True):
        if r['game_id'] in games or len(recommend)>=5:
            r['status']='held';r['reasons'].append('Slate exposure limit');continue
        games.add(r['game_id']);recommend.append(r)
    return recommend,watch,candidates
