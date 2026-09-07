"""Source fidelity and challenger integration contract tests."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from challenger.engine.features import FEATURES
from challenger.engine.model import predict_rows, probabilities
from challenger.engine.market import devig, implied_center
from qb_attempts.challenger import (ROOT, final_center, load_challenger,
                                   native_policy_reasons, score_challenger)


@pytest.fixture(scope='module')
def bundle():
    return load_challenger(prefer_runtime=False)


@pytest.fixture
def inputs(bundle):
    now=pd.Timestamp('2026-09-06T14:00:00Z')
    game=dict(game_id='2026_01_CHI_GB',season=2026,week=1,kickoff=now+pd.Timedelta(days=2),
              home_team='GB',away_team='CHI',home_score=np.nan,away_score=np.nan,home_coach='Matt LaFleur',
              away_coach='Ben Johnson',home_rest=7.,away_rest=7.,location='Home',roof='outdoors',div_game=1)
    player=bundle['records'][bundle['records'].team=='GB'].iloc[-1]
    quotes=[]
    for book in ('draftkings','fanduel','caesars'):
        quotes.append(dict(game_id=game['game_id'],player_id=player.player_id,player=player.player,team='GB',opponent='CHI',
                           home_team='GB',away_team='CHI',season=2026,week=1,kickoff=game['kickoff'],book=book,
                           observed_at=now.isoformat(),source_updated_at=None,source_url='https://example.com/public',
                           roster_status='ACT',line=30.5,over_odds=-110,under_odds=-110))
    market={game['game_id']:dict(home_expected_margin=3.,total_line=45.,observed_at=now.isoformat())}
    context={'GB':dict(available=True,starter=player.player,injuries=[],observed_at=now.isoformat(),source_updated_at=now.isoformat(),season=2026)}
    return quotes,pd.DataFrame([game]),market,context,now


def test_source_snapshot_sha256_matches_provenance():
    root=ROOT/'challenger';manifest=json.loads((root/'provenance.json').read_text())
    for name in ('features','model','market','edges'):
        assert hashlib.sha256((root/f'source/{name}.py').read_bytes()).hexdigest()==manifest['files'][f'src/{name}.py']['sha256']
    assert hashlib.sha256((root/'frozen/records.parquet').read_bytes()).hexdigest()==manifest['files']['data/records.parquet']['sha256']


def test_runtime_feature_and_math_function_bodies_are_original():
    root=ROOT/'challenger'
    assert (root/'source/features.py').read_bytes()==(root/'engine/features.py').read_bytes()
    original={n.name:ast.dump(n) for n in ast.parse((root/'source/model.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    runtime={n.name:ast.dump(n) for n in ast.parse((root/'engine/model.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    assert len(runtime)>=8
    assert all(original[name]==body for name,body in runtime.items())


def test_numeric_predictions_probabilities_and_market_policy_match_saved_source(bundle):
    # Expected values were computed with the original saved sklearn estimator,
    # original model.py function AST and original feature rows, before adapter.
    fixture=json.loads((ROOT/'challenger/frozen/fidelity.json').read_text())
    rows=pd.DataFrame([{**r['features'],'player_id':r['player_id']} for r in fixture])
    means,scales=predict_rows(bundle['artifact'],rows)
    for i,expected in enumerate(fixture):
        assert means[i]==pytest.approx(expected['mean'],abs=1e-12)
        assert scales[i]==pytest.approx(expected['scale'],abs=1e-12)
        resid=bundle['artifact']['residuals']
        centers=[]
        for j,(line,prices) in enumerate(zip(expected['lines'],expected['prices'])):
            assert probabilities(means[i],resid,line,scales[i])==pytest.approx(expected['probabilities'][j],abs=1e-12)
            centers.append(implied_center(line,devig(*prices),resid,scales[i]))
        center=float(np.median(centers));final=final_center(means[i],center,bundle['policy'])
        assert center==pytest.approx(expected['market_center'],abs=1e-12)
        assert final==pytest.approx(expected['final_mean'],abs=1e-12)
        for j,line in enumerate(expected['lines']):
            assert probabilities(final,resid,line,scales[i])==pytest.approx(expected['final_probabilities'][j],abs=1e-12)


@pytest.mark.parametrize('gap,expected',[(1.49,30.),(-1.49,30.),(1.5,30.525),(-1.5,29.475),(5.,31.75)])
def test_native_deadzone_and_mean_blend(bundle,gap,expected):
    assert final_center(30+gap,30,bundle['policy'])==pytest.approx(expected)


@pytest.mark.parametrize('field,value,reason',[
    ('estimate',.029,'ev<floor'),('estimate',.151,'ev>cap:verify'),
    ('price',-251,'price too short'),('gap',6.01,'model/market gap>6'),
    ('n_books',2,'thin market'),('quote_age',36.1,'stale quote'),('book','pinnacle','not NY-legal')])
def test_native_gates(bundle,field,value,reason):
    args=dict(estimate=.05,price=-110,gap=2.,n_books=3,quote_age=None,book='draftkings',policy=bundle['policy']);args[field]=value
    assert reason in native_policy_reasons(**args)


def test_same_quotes_target_included_crossline_center(inputs,bundle):
    quotes,games,markets,context,now=inputs
    quotes[0]['line']=29.5;quotes[1]['line']=31.;quotes[2]['line']=32.5
    result=score_challenger(quotes,games,markets,context,now,bundle)
    assert len(result['evaluated_quotes'])==6
    row=result['evaluated_quotes'][0]
    centers=[implied_center(q['line'],devig(q['over_odds'],q['under_odds']),bundle['artifact']['residuals'],row['scale']) for q in quotes]
    assert row['mu_market']==pytest.approx(np.median(centers))
    assert row['n_books']==3 and row['other_books']==2
    assert row['book'] in row['reference_books']
    assert result['metadata']['market_center_includes_target_book'] is True
    assert row['bet_to_target']==.02


def test_repeated_same_book_does_not_inflate_book_count(inputs,bundle):
    quotes,games,markets,context,now=inputs
    quotes[2]['book']='draftkings';quotes[2]['line']=31.5
    result=score_challenger(quotes,games,markets,context,now,bundle)
    assert all(r['n_books']==2 and 'thin market' in r['native_reasons'] for r in result['evaluated_quotes'])
    assert result['recommendations']==[]


def test_injury_and_roster_gates_are_explicit_adapter_rules(inputs,bundle):
    quotes,games,markets,context,now=inputs
    clean=score_challenger(quotes,games,markets,context,now,bundle)
    context['GB']['injuries']=['Questionable'];quotes[0]['roster_status']='INA'
    held=score_challenger(quotes,games,markets,context,now,bundle)
    clean_by={(r['book'],r['side']):r for r in clean['evaluated_quotes']}
    for r in held['evaluated_quotes']:
        assert r['native_reasons']==clean_by[r['book'],r['side']]['native_reasons']
        assert 'Adapter: QB injury designation needs review' in r['shared_safety_reasons']
        assert r['status']=='held'
    assert not held['recommendations']


def test_stale_quotes_do_not_influence_center(inputs,bundle):
    quotes,games,markets,context,now=inputs
    quotes[0]['observed_at']=(now-pd.Timedelta(minutes=31)).isoformat()
    result=score_challenger(quotes,games,markets,context,now,bundle)
    assert result['metadata']['eligible_quotes']==2
    assert len(result['metadata']['excluded_quotes'])==1
    assert all(r['n_books']==2 for r in result['evaluated_quotes'])


def test_missing_game_market_fails_closed(inputs,bundle):
    quotes,games,markets,context,now=inputs
    result=score_challenger(quotes,games,{},context,now,bundle)
    assert result['status']=='source_failure' and not result['recommendations']


def test_new_completed_games_cannot_use_frozen_stale_history(inputs,bundle):
    quotes,games,markets,context,now=inputs
    past=games.iloc[0].to_dict();past.update(game_id='2026_01_BUF_NE',kickoff=now-pd.Timedelta(days=1),home_score=20.,away_score=10.)
    games=pd.concat([games,pd.DataFrame([past])],ignore_index=True)
    result=score_challenger(quotes,games,markets,context,now,bundle)
    assert result['status']=='stale_history'
    assert result['metadata']['missing_history_games']==['2026_01_BUF_NE']
    assert not result['recommendations']


def test_selected_source_distribution_preserves_integer_push_mass(bundle):
    over,under,push=probabilities(31.,bundle['artifact']['residuals'],31.)
    assert push>0
    assert over+under+push==pytest.approx(1.)
    assert probabilities(31.,bundle['artifact']['residuals'],31.5)[2]==pytest.approx(0.)


def test_native_policy_retains_no_five_bet_card_cap(inputs,bundle,monkeypatch):
    import qb_attempts.challenger as module
    quotes,games,markets,context,now=inputs
    new_quotes=[];new_games=[];new_markets={};new_context={}
    for i,team in enumerate(('GB','CHI','MIN','DET','WAS','PHI')):
        game=games.iloc[0].to_dict();game.update(game_id=f'2026_01_TB_{team}',home_team=team,away_team='TB')
        new_games.append(game);new_markets[game['game_id']]=copy.deepcopy(next(iter(markets.values())))
        for quote in quotes:
            quote={**quote,'game_id':game['game_id'],'player_id':f'test_qb_{i}','player':f'Test QB {i}',
                   'team':team,'opponent':'TB','home_team':team,'away_team':'TB','over_odds':100,'under_odds':-120}
            new_quotes.append(quote)
        new_context[team]={**context['GB'],'starter':f'Test QB {i}'}
    monkeypatch.setattr(module,'predict_rows',lambda artifact,frame:(np.repeat(30.,len(frame)),np.ones(len(frame))))
    monkeypatch.setattr(module,'implied_center',lambda *args:30.)
    monkeypatch.setattr(module,'probabilities',lambda *args:(.55,.45,0.))
    result=module.score_challenger(new_quotes,pd.DataFrame(new_games),new_markets,new_context,now,bundle)
    assert len(result['recommendations'])==6
    assert len({r['game_id'] for r in result['recommendations']})==6
    assert all(r['side']=='Over' for r in result['recommendations'])


def test_refresh_adds_current_season_and_refits_only_base(inputs,tmp_path):
    import os
    import shutil
    from challenger.engine.features import PBP_COLS
    from scripts.refresh_challenger import refresh_challenger
    quotes,games,markets,context,now=inputs
    root=tmp_path/'challenger';(root/'frozen').mkdir(parents=True)
    for name in ('records.parquet','model.json'):
        shutil.copy2(ROOT/'challenger/frozen'/name,root/'frozen'/name)
    original=json.loads((root/'frozen/model.json').read_text())
    game=games.iloc[0].to_dict();game.update(kickoff=now-pd.Timedelta(days=1),home_score=20.,away_score=10.,
         home_qb_id='test_home',away_qb_id='test_away',home_qb_name='Home QB',away_qb_name='Away QB',result=10.,total=30.,spread_line=3.,total_line=45.)
    stats=pd.DataFrame([dict(game_id=game['game_id'],player_id='test_'+side,attempts=1.,carries=1.,sacks_suffered=0.) for side in ('home','away')])
    rows=[]
    for team,opponent in [('GB','CHI'),('CHI','GB')]:
        row={key:0 for key in PBP_COLS}
        row.update(game_id=game['game_id'],season=2026,week=1,season_type='REG',posteam=team,defteam=opponent,
                   home_team='GB',away_team='CHI',pass_attempt=1,play_type='pass',qb_dropback=1,
                   game_seconds_remaining=3500,qtr=1,xpass=.5,fixed_drive=1)
        rows.append(row)
    cache=tmp_path/'pbp';cache.mkdir();path=cache/'play_by_play_2026.parquet'
    pd.DataFrame(rows).to_parquet(path,index=False);os.utime(path,(now.timestamp(),now.timestamp()))
    result=refresh_challenger(stats,pd.DataFrame([game]),root=root,now=now,cache=cache)
    refreshed=json.loads((root/'runtime/model.json').read_text())
    assert result['base_refitted'] is True
    assert result['processed_game_ids']==[game['game_id']]
    assert refreshed['trained_through']==2026
    assert refreshed['residuals']==original['residuals']
    assert refreshed['eb']==original['eb']
    assert refreshed['scale']==original['scale']
    assert refreshed['estimator']!=original['estimator']
    assert result['records']==4724
