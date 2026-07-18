"""Cost ledger aggregation: totals derived from rows, ordering, and folding."""

from __future__ import annotations

from pathlib import Path

from cc_session_core.cost.ledger import ProjectLedger, build_ledger

FIXTURES = Path(__file__).parent / "fixtures"


def _ledger() -> ProjectLedger:
    return build_ledger(FIXTURES)


def test_a_row_per_session_file() -> None:
    ledger = _ledger()
    assert ledger.sessions == len(list(FIXTURES.rglob("*.jsonl")))
    assert len(ledger.rows) == ledger.sessions


def test_totals_are_derived_from_rows() -> None:
    ledger = _ledger()
    assert ledger.total_cost_usd == sum(r.cost_usd for r in ledger.rows)
    assert ledger.total_tokens == sum(r.tokens for r in ledger.rows)
    assert ledger.total_requests == sum(r.requests for r in ledger.rows)
    # by_model is the same money, folded a different way
    assert abs(ledger.total_cost_usd - sum(m.cost_usd for m in ledger.by_model)) < 1e-6


def test_rows_and_models_sorted_by_cost_descending() -> None:
    ledger = _ledger()
    assert [r.cost_usd for r in ledger.rows] == sorted(
        (r.cost_usd for r in ledger.rows), reverse=True
    )
    assert [m.cost_usd for m in ledger.by_model] == sorted(
        (m.cost_usd for m in ledger.by_model), reverse=True
    )


def test_days_sorted_ascending_and_deduped() -> None:
    ledger = _ledger()
    dates = [d.date for d in ledger.by_day]
    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))


def test_computed_totals_serialize() -> None:
    dumped = _ledger().model_dump()
    for key in ("sessions", "total_cost_usd", "total_tokens", "start_date", "end_date"):
        assert key in dumped
