from copy import deepcopy
import pandas as pd
from qb_attempts.early_entry import select_early_entries

NOW = pd.Timestamp('2026-09-14T12:00Z')


def fixtures():
    q = dict(game_id='g', player_id='p', book='fanduel', line=30.5, player='QB',
             team='DEN', season=2026, roster_status='ACT',
             quote_verification={'status':'verified', 'observed_at':NOW.isoformat()})
    row = {**q, 'kickoff':'2026-09-17T20:00Z', 'side':'Over', 'odds':-110,
           'ev':.1, 'robust_ev':.06, 'status':'held', 'reasons':['Starting role not independently verified',
           'Starting context stale or unverified'], 'p_final':.58}
    return q, row


def test_days_early_unknown_role_qualifies_without_mutating_research():
    q, row = fixtures()
    frozen = deepcopy(row)
    selected, _ = select_early_entries([row], [q], {}, NOW)
    assert len(selected) == 1
    assert selected[0]['warnings']
    assert row == frozen
    assert selected[0]['p_final'] == row['p_final']


def test_questionable_warns_out_withdraws_and_unverified_price_holds():
    q, row = fixtures()
    context = {'DEN':dict(available=True, starter='QB', season=2026, observed_at=NOW.isoformat(),
                          source_updated_at=NOW.isoformat(), injuries=['Questionable'])}
    assert len(select_early_entries([row], [q], context, NOW)[0]) == 1
    context['DEN']['injuries'] = ['Out']
    assert not select_early_entries([row], [q], context, NOW)[0]
    q['quote_verification']['status'] = 'unverified'
    assert not select_early_entries([row], [q], {}, NOW)[0]


def test_numeric_holds_remain_and_exposure_survives_refresh():
    q, row = fixtures()
    row['reasons'].append('Estimated EV below 3%')
    assert not select_early_entries([row], [q], {}, NOW)[0]
    row['reasons'].pop()
    ledger = {'entries':[dict(model='champion', game_id='g', player_id='other', side='under', line=30.5)]}
    assert not select_early_entries([row], [q], {}, NOW, ledger=ledger)[0]


def test_stale_context_out_does_not_become_new_unavailability():
    q, row = fixtures()
    context = {'DEN':dict(available=True, starter='QB', season=2026, observed_at=NOW.isoformat(),
                          source_updated_at='2026-09-01T12:00Z', injuries=['Out'])}
    assert len(select_early_entries([row], [q], context, NOW)[0]) == 1
