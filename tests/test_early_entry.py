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


def test_source_gap_is_not_a_model_rejection_and_reason_is_preserved():
    q, row = fixtures()
    q['quote_verification'] = {'status':'unverified','reason':'matching_paired_market_not_found'}
    selected, watch = select_early_entries([row], [q], {}, NOW)
    assert not selected
    assert watch[0]['model_status'] == 'qualifies'
    assert watch[0]['model_reasons'] == []
    assert watch[0]['price_status'] == 'unverified'
    assert 'matching_paired_market_not_found' in watch[0]['reasons'][0]


def test_verified_alternative_is_shown_ahead_of_an_unverified_high_ev_quote():
    q, row = fixtures()
    row['reasons'] = ['Estimated EV below 3%']
    row['ev'] = .01
    other_q = {**q, 'book':'draftkings', 'quote_verification':{'status':'unverified','reason':'no feed'}}
    other_row = {**row, 'book':'draftkings', 'ev':.2, 'robust_ev':.18, 'reasons':[]}
    _, watch = select_early_entries([row, other_row], [q, other_q], {}, NOW)
    assert watch[0]['book'] == 'fanduel'
    assert watch[0]['model_status'] == 'held'
    assert watch[0]['price_status'] == 'verified'


def test_comparison_update_must_still_be_fresh_at_final_decision_time():
    q, row = fixtures()
    q['quote_verification'].update(evidence_level='timestamped_comparison', source_updated_at='2026-09-14T11:29Z')
    assert not select_early_entries([row], [q], {}, NOW)[0]
