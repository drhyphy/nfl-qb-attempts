from copy import deepcopy
from datetime import datetime, timezone

import pytest

from qb_attempts.quote_verification import parse_fanduel, verify_against_offers, verify_quotes

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
Q = {'game_id':'2026_01_DEN_KC','player_id':'00-0039913','player':'Bo Nix',
     'home_team':'KC','away_team':'DEN','kickoff':'2026-09-15T00:15:00+00:00',
     'book':'fanduel','line':32.5,'over_odds':-110,'under_odds':-110,
     'source':'scoresandodds','observed_at':'2026-09-14T11:58:00+00:00'}


def payload():
    return {'attachments': {'events': {'123': {'name':'Denver Broncos @ Kansas City Chiefs','openDate':Q['kickoff']}},
        'markets': {'456': {'eventId':123,'marketId':'456','marketName':'Bo Nix - Pass Attempts',
            'marketStatus':'OPEN','inPlay':False,
            'runners':[{'runnerName':s.title(),'handicap':32.5,'selectionId':i,'runnerStatus':'ACTIVE',
                        'winRunnerOdds':{'americanDisplayOdds':{'americanOdds':-110}}} for i,s in enumerate(('over','under'),1)]}}}}


def offers():
    return parse_fanduel(payload(), NOW.isoformat(), 'proof.json')


def check(q=None, rows=None):
    return verify_against_offers([q or deepcopy(Q)], offers() if rows is None else rows, NOW)[0]['quote_verification']


def test_exact_pair_preserves_original_price_and_provenance_without_role_confirmation():
    q = dict(Q, roster_status='unknown', confirmed_starter=False)
    original = deepcopy(q)
    result = verify_against_offers([q], offers(), NOW)[0]
    assert q == original
    assert all(result[k] == v for k,v in original.items())
    evidence = result['quote_verification']
    assert evidence['status']=='verified'
    assert evidence['source_updated_at'] is None  # retrieval is not sportsbook update time
    assert evidence['market_id']=='456'
    assert evidence['evidence_level']=='sportsbook_direct'
    assert evidence['original_provenance']['source']=='scoresandodds'
    assert evidence['sides']['under']['selection_id']=='2'


@pytest.mark.parametrize('field,value', [('line',33.5), ('over_odds',-115), ('under_odds',100)])
def test_price_or_line_mismatch_does_not_substitute(field,value):
    q = dict(Q, **{field:value})
    assert check(q)['status']=='unverified'
    assert 'changed' in check(q)['reason']


@pytest.mark.parametrize('field,value', [('book','draftkings'),('player','Patrick Mahomes'),
    ('home_team','DEN'),('away_team','KC'),('kickoff','2026-09-16T00:15:00+00:00')])
def test_exact_identity_required(field,value):
    assert check(dict(Q, **{field:value}))['status']=='unverified'


@pytest.mark.parametrize('field,value', [('observed_at','2026-09-14T11:29:00+00:00'),
    ('observed_at','2026-09-14T12:01:00+00:00'),('observed_at','2026-09-14T12:00:00'),
    ('available',False),('available',None),('selection_id',''),('market_id',''),('jurisdiction','US-NJ')])
def test_incomplete_stale_future_or_unavailable_evidence(field,value):
    rows=offers();rows[0][field]=value
    assert check(rows=rows)['status']=='unverified'


def test_pair_requires_same_market_and_source():
    rows=offers();rows[0]['market_id']='other'
    assert check(rows=rows)['status']=='unverified'
    rows=offers();rows[0]['source']='some_comparison'
    assert check(rows=rows)['status']=='unverified'


def test_conflicting_selection_never_uses_first_match():
    rows=offers();rows.append(dict(rows[0],odds=-120))
    assert 'conflicting_source_quotes' in check(rows=rows)['reason']


def test_timestamped_comparison_needs_fresh_source_time():
    rows=[dict(o,evidence_level='timestamped_comparison',source='comparison') for o in offers()]
    assert check(rows=rows)['status']=='unverified'
    rows=[dict(o,source_updated_at='2026-09-14T11:55:00+00:00') for o in rows]
    assert check(rows=rows)['status']=='verified'
    rows[1]['source_updated_at']='2026-09-14T11:20:00+00:00'
    assert check(rows=rows)['status']=='unverified'


@pytest.mark.parametrize('title',['Bo Nix - Alt Pass Attempts','Bo Nix - Passing Yds',
    'Bo Nix - 1st Half Pass Attempts','Bo Nix - Pass Attempts Boost'])
def test_parser_rejects_non_full_game_attempts(title):
    p=payload();p['attachments']['markets']['456']['marketName']=title
    assert parse_fanduel(p,NOW.isoformat())==[]


@pytest.mark.parametrize('field,value',[('marketStatus','SUSPENDED'),('inPlay',True),('inPlay',None)])
def test_parser_preserves_unavailable_as_unverified(field,value):
    p=payload();p['attachments']['markets']['456'][field]=value
    rows=parse_fanduel(p,NOW.isoformat())
    assert len(rows)==2
    assert check(rows=rows)['status']=='unverified'


def test_missing_runner_status_is_not_assumed_active():
    p=payload();del p['attachments']['markets']['456']['runners'][0]['runnerStatus']
    assert check(rows=parse_fanduel(p,NOW.isoformat()))['status']=='unverified'


def test_empty_feed_and_network_failure_cannot_verify(monkeypatch,tmp_path):
    def fail(*args,**kwargs):
        raise ConnectionError('public endpoint unavailable')
    monkeypatch.setattr('qb_attempts.quote_verification.requests.get',fail)
    result,errors=verify_quotes([Q],NOW,tmp_path)
    assert result[0]['quote_verification']['status']=='unverified'
    assert errors and 'discovery failed' in errors[0]
    assert list(tmp_path.glob('*/fanduel-nfl.error.json'))
    assert check(rows=[])['status']=='unverified'


def test_unsupported_book_skips_network(monkeypatch,tmp_path):
    def fail(*args,**kwargs):
        pytest.fail('unsupported book should not request FanDuel')
    monkeypatch.setattr('qb_attempts.quote_verification.requests.get',fail)
    result,errors=verify_quotes([dict(Q,book='draftkings')],NOW,tmp_path)
    assert not errors
    assert result[0]['quote_verification']['status']=='unverified'


def test_event_requests_are_deduplicated(monkeypatch,tmp_path):
    calls=[]
    def fake_fetch(url,params,directory,label):
        calls.append((url,params))
        return payload(),datetime.now(timezone.utc).isoformat(),directory/(label+'.json')
    monkeypatch.setattr('qb_attempts.quote_verification._fetch',fake_fetch)
    verify_quotes([Q,dict(Q,line=34.5)],NOW,tmp_path)
    assert len(calls)==2  # one NFL discovery and one event, not one call per quote


def test_supplementary_comparison_verifies_other_book_without_fanduel_request(monkeypatch,tmp_path):
    now=datetime.now(timezone.utc)
    from datetime import timedelta
    import json
    q=dict(Q,book='draftkings',kickoff=(now+timedelta(days=1)).isoformat())
    rows=[dict(o,book='draftkings',kickoff=q['kickoff'],observed_at=now.isoformat(),
               source_updated_at=now.isoformat(),source='bettingpros',evidence_level='timestamped_comparison') for o in offers()]
    def fail(*args,**kwargs):
        pytest.fail('Other-book supplementary evidence needs no FanDuel request')
    monkeypatch.setattr('qb_attempts.quote_verification.requests.get',fail)
    enriched,errors=verify_quotes([q],now,tmp_path,evidence_offers=rows)
    assert not errors
    assert enriched[0]['quote_verification']['status']=='verified'
    assert enriched[0]['quote_verification']['source']=='bettingpros'
    summary=json.loads(next(tmp_path.glob('*/verification-summary.json')).read_text())
    assert summary['unsupported_books']==[]
    assert summary['evidence_sources']==['bettingpros']


def test_supplementary_offer_must_still_be_timestamped_available_and_exact(monkeypatch,tmp_path):
    now=datetime.now(timezone.utc)
    from datetime import timedelta
    q=dict(Q,book='draftkings',kickoff=(now+timedelta(days=1)).isoformat())
    rows=[dict(o,book='draftkings',kickoff=q['kickoff'],observed_at=now.isoformat(),
               source='bettingpros',evidence_level='timestamped_comparison') for o in offers()]
    enriched,_=verify_quotes([q],now,tmp_path,evidence_offers=rows)
    assert enriched[0]['quote_verification']['status']=='unverified'
    for row in rows:row['source_updated_at']=now.isoformat()
    rows[0]['available']=False
    enriched,_=verify_quotes([q],now,tmp_path,evidence_offers=rows)
    assert enriched[0]['quote_verification']['status']=='unverified'
    rows[0]['available']=True;rows[0]['odds']=-120
    enriched,_=verify_quotes([q],now,tmp_path,evidence_offers=rows)
    assert enriched[0]['quote_verification']['status']=='unverified'
    assert enriched[0]['over_odds']==q['over_odds']
