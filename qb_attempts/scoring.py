from __future__ import annotations
import math,re,unicodedata
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .features import History,team_key
from .model import predict,probabilities

# Frozen launch policy, selected from prior-model failure modes, not today's bets.
POLICY={'version':'market-anchor-v1','min_ev':.025,'min_robust_ev':0.,'min_other_books':2,'max_disagreement':.15,'max_peer_range':.10,'max_offer_outlier':.10,'min_qb_starts':4,'model_weight_week1':.15,'model_weight_inseason':.25,'probability_stress':.015,'max_ev':.15,'max_observation_age_minutes':30}
NY_BOOKS={'draftkings','fanduel','caesars','fanatics','betmgm','betrivers','ballybet','thescorebet'}
def name_key(x):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',str(x)).encode('ascii','ignore').decode().lower())
def payout(odds):
    if not math.isfinite(float(odds)) or abs(odds)<100:raise ValueError('Invalid American odds')
    return odds/100 if odds>0 else 100/(-odds)
def implied(odds):return 1/(1+payout(odds))
def no_vig(over,under):
    po,pu=implied(over),implied(under);return po/(po+pu)
def ev(p,odds,push=0.):return p*payout(odds)-(1-p-push)
def fair_odds(p):
    p=float(np.clip(p,.001,.999));return int(round(100*(1-p)/p if p<=.5 else -100*p/(1-p)))
def bet_to(p,push=0.,target=.02):
    b=(1-p-push+target)/max(p,.001)
    # Round toward the more favorable offer so the threshold never overstates EV.
    return math.ceil(100*b) if b>=1 else math.ceil(-100/b)

def resolve_quotes(quotes,roster,games,now):
    errors=[];resolved=[]
    roster=roster.copy();roster['key']=roster.full_name.map(name_key);roster['team']=roster.team.map(team_key)
    roster=roster[(roster.position=='QB')&(roster.week<=int(games[games.kickoff>now].week.min()) if len(games[games.kickoff>now]) else True)]
    roster=roster.sort_values('week').drop_duplicates('gsis_id',keep='last')
    for q in quotes:
        try:
            kick=pd.Timestamp(q['kickoff']);observed=pd.Timestamp(q['observed_at'])
            if kick.tzinfo is None or observed.tzinfo is None:raise ValueError('missing timezone')
            if kick<=now or kick>now+pd.Timedelta(days=9):continue
            if observed>now+pd.Timedelta(minutes=2) or observed<now-pd.Timedelta(minutes=POLICY['max_observation_age_minutes']):raise ValueError('stale or future observation')
            if q.get('source_updated_at'):
                updated=pd.Timestamp(q['source_updated_at'])
                if updated.tzinfo is None or updated>now+pd.Timedelta(minutes=2) or updated<now-pd.Timedelta(hours=24):raise ValueError('source update outside 24-hour age policy or in future')
            matches=games[(games.home_team==team_key(q['home_team']))&(games.away_team==team_key(q['away_team']))&((games.kickoff-kick).abs()<pd.Timedelta(minutes=10))]
            if len(matches)!=1:raise ValueError('schedule identity mismatch')
            g=matches.iloc[0]
            if team_key(q['team']) not in (g.home_team,g.away_team):raise ValueError('player team not a participant in scheduled game')
            r=roster[(roster.key==name_key(q['player']))&(roster.team==team_key(q['team']))]
            if len(r)!=1:raise ValueError('roster identity mismatch')
            r=r.iloc[0];side='home' if r.team==g.home_team else 'away';other='away' if side=='home' else 'home'
            if q['opponent']!=g[other+'_team']:raise ValueError('opponent identity mismatch')
            resolved.append({**q,'provider_event_id':q['event_id'],'game_id':g.game_id,'player_id':r.gsis_id,'season':int(g.season),'week':int(g.week),'kickoff':g.kickoff,'as_of':now,'home':int(side=='home' and g.location!='Neutral'),'rest':float(g[side+'_rest']),'coach':g[side+'_coach'],'dome':int(g.roof in ('dome','closed')),'roster_status':r.status})
        except (ValueError,KeyError,TypeError) as e:errors.append(f"{q.get('player','Unknown')}: {e}")
    # Same book never gets two votes; latest capture wins.
    unique={}
    for q in sorted(resolved,key=lambda x:pd.Timestamp(x['observed_at'])):unique[(q['game_id'],q['player_id'],q['line'],q['book'])]=q
    return list(unique.values()),sorted(set(errors))

def score(quotes,artifact,records,context,now):
    if not quotes:return [],[],[]
    requests={}
    for q in quotes:requests[(q['game_id'],q['player_id'])]=q
    features=History(records).features(list(requests.values()));features['mean']=predict(artifact['name'],artifact['estimator'],features)
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
        base_weight=POLICY['model_weight_week1'] if f['season_games']<3 else POLICY['model_weight_inseason']
        weight=base_weight*min(1.,f['qb_games']/12)*(.5 if f['coach_change'] or f['qb_team_change'] else 1.)
        weight*=max(.25,1-abs(conditional_model-market)/.25)
        blend=market+weight*(conditional_model-market)
        blend=float(np.clip(blend,market-.04,market+.04))
        ctx=context.get(q['team'],{})
        common=[]
        if len(peers)<POLICY['min_other_books']:common.append('Fewer than 2 other books at this line')
        if ps and max(ps)-min(ps)>POLICY['max_peer_range']:common.append('Sportsbooks disagree materially')
        if abs(conditional_model-market)>POLICY['max_disagreement']:common.append('Large model–market disagreement')
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
            # Stress both model weighting and a 1.5pp probability miss; never label a CI.
            worst=min(p_market,conditional)-POLICY['probability_stress']
            robust=ev(max(0,worst)*(1-push),price,push)
            estimate=ev(p_final,price,push)
            reasons=common.copy()
            if estimate<POLICY['min_ev']:reasons.append('Estimated EV below 2.5%')
            if robust<=POLICY['min_robust_ev']:reasons.append('Edge fails probability stress test')
            if estimate>POLICY['max_ev']:reasons.append('Unusually large EV needs review')
            quantiles=np.quantile(residuals,[.1,.9])+f['mean']
            candidates.append({**{k:q[k] for k in ['game_id','player_id','player','team','opponent','line','book','source_url','observed_at','source_updated_at']},'kickoff':q['kickoff'].isoformat(),'side':side,'odds':int(price),'mean':float(f['mean']),'interval80':np.clip(quantiles,0,90).tolist(),'p_model':float(p_m),'p_market':float(p_market*(1-push)),'p_final':float(p_final),'p_push':float(push),'ev':float(estimate),'robust_ev':float(robust),'market_only_ev':float(ev(p_market*(1-push),price,push)),'fair_odds':fair_odds(conditional),'bet_to':bet_to(p_final,push),'model_weight':float(weight),'other_books':len(peers),'reference_books':sorted({x['book'] for x in peers}),'status':'qualified' if not reasons else 'held','reasons':reasons,'starting_status':ctx,'feature_flags':{k:f[k] for k in ['coach_change','qb_team_change','qb_games','season_games']}})
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
