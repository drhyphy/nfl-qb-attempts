from copy import deepcopy
from qb_attempts.quote_selection import select_current_quotes


def quote(**changes):
    return dict(game_id='g', player_id='p', book='draftkings', kickoff='2026-09-20T17:00Z',
                line=30.5, over_odds=-110, under_odds=-110, source='scoresandodds',
                observed_at='2026-09-16T20:00Z', **changes)


def test_current_verified_pair_replaces_stale_source_line_without_relabelling():
    old = quote()
    fresh = {**old, 'source':'bettingpros', 'line':32.5, 'over_odds':-120, 'under_odds':100,
             'quote_verification':{'status':'verified','evidence_level':'timestamped_comparison'}}
    frozen = deepcopy([old, fresh])
    assert select_current_quotes([old, fresh]) == [fresh]
    assert [old, fresh] == frozen


def test_one_book_one_main_line_but_other_books_remain_independent():
    a = quote()
    b = {**a, 'source':'bettingpros', 'line':31.5}
    c = {**a, 'book':'fanduel'}
    selected = select_current_quotes([a, b, c])
    assert len(selected) == 2
    assert next(q for q in selected if q['book']=='draftkings')['line'] == 31.5


def test_verified_direct_source_takes_precedence_over_comparison():
    a = quote(quote_verification={'status':'verified','evidence_level':'sportsbook_direct'})
    b = {**a, 'source':'bettingpros', 'quote_verification':{'status':'verified','evidence_level':'timestamped_comparison'}}
    assert select_current_quotes([b, a]) == [a]
