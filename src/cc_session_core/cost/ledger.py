"""Cost ledger: aggregate every session under a directory into one rollup.

Pure data — per-session rows plus by-day / by-model rollups, computed once from
the de-duplicated :class:`~cc_session_core.session.Session` cost views so token and cost
figures are counted once per API request. Presentation layers (cc-session-explorer's
Sankey HTML reports and API) render this; they do not recompute it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field

from .. import types as t
from ..models import SnakeModel
from ..session import Session, load_project


class LedgerRow(SnakeModel):
    """One session's line in the ledger."""

    id: t.SessionId = ""
    title: t.SessionTitle = ""
    project: t.DisplayName = ""
    source: t.FilePath | None = None
    date: t.DateString = ""
    model: t.ModelId = ""
    requests: t.Count = 0
    tool_calls: t.Count = 0
    tokens: t.TokenCount = 0
    cost_usd: t.CostUsd = 0.0


class DayCost(SnakeModel):
    date: t.DateString
    cost_usd: t.CostUsd = 0.0
    tokens: t.TokenCount = 0
    sessions: t.Count = 0


class ModelCost(SnakeModel):
    model: t.ModelId
    cost_usd: t.CostUsd = 0.0
    requests: t.Count = 0
    tokens: t.TokenCount = 0


class ProjectLedger(SnakeModel):
    """Every session under a directory, with by-day / by-model rollups. The ``total_*``
    fields are derived from ``rows`` so they cannot drift from the lines that back them.
    ``by_day`` and ``start_date``/``end_date`` cover only dated sessions — a session
    with no timestamp still counts in the totals but has no day to fall under, so
    ``sum(by_day cost) <= total_cost_usd``."""

    root: t.DisplayPath = ""
    rows: list[LedgerRow] = []
    by_day: list[DayCost] = []
    by_model: list[ModelCost] = []

    @computed_field
    @property
    def sessions(self) -> t.Count:
        return len(self.rows)

    @computed_field
    @property
    def total_cost_usd(self) -> t.CostUsd:
        return sum(r.cost_usd for r in self.rows)

    @computed_field
    @property
    def total_tokens(self) -> t.TokenCount:
        return sum(r.tokens for r in self.rows)

    @computed_field
    @property
    def total_requests(self) -> t.Count:
        return sum(r.requests for r in self.rows)

    @computed_field
    @property
    def total_tool_calls(self) -> t.Count:
        return sum(r.tool_calls for r in self.rows)

    @computed_field
    @property
    def start_date(self) -> t.DateString:
        return self.by_day[0].date if self.by_day else ""

    @computed_field
    @property
    def end_date(self) -> t.DateString:
        return self.by_day[-1].date if self.by_day else ""


def session_row(session: Session, root: Path) -> tuple[LedgerRow, list[ModelCost]]:
    """A ledger row for one session, plus its per-model cost contributions."""
    info = session.info()
    cost = session.cost_summary()
    tokens = cost.total_tokens
    # A row carries session-wide totals, so label it by the model that cost the
    # most (not the one with the most requests) — the dominant contributor.
    top_model = max(cost.by_model, key=lambda m: m.cost_usd or 0.0).model if cost.by_model else ""
    source = Path(info.source) if info.source else None
    project = source.parent.name if source and source.parent != root else ""
    per_model = [
        ModelCost(
            model=mu.model,
            cost_usd=mu.cost_usd or 0.0,
            requests=mu.requests,
            tokens=mu.total_tokens,
        )
        for mu in cost.by_model
    ]
    row = LedgerRow(
        id=info.id,
        title=info.title,
        project=project,
        source=str(source) if source else None,
        date=info.started.date().isoformat() if info.started else "",
        model=top_model,
        requests=cost.requests,
        tool_calls=info.tool_calls,
        tokens=tokens,
        cost_usd=cost.total_cost_usd or 0.0,
    )
    return row, per_model


def _fold_days(rows: list[LedgerRow]) -> list[DayCost]:
    by_day: dict[str, DayCost] = {}
    for row in rows:
        if not row.date:
            continue
        day = by_day.setdefault(row.date, DayCost(date=row.date))
        day.cost_usd += row.cost_usd
        day.tokens += row.tokens
        day.sessions += 1
    return [by_day[d] for d in sorted(by_day)]


def _fold_models(all_models: list[ModelCost]) -> list[ModelCost]:
    acc: dict[str, ModelCost] = {}
    for mc in all_models:
        into = acc.setdefault(mc.model, ModelCost(model=mc.model))
        into.cost_usd += mc.cost_usd
        into.requests += mc.requests
        into.tokens += mc.tokens
    return sorted(acc.values(), key=lambda m: m.cost_usd, reverse=True)


def build_ledger(directory: str | Path) -> ProjectLedger:
    """Aggregate every ``.jsonl`` session under ``directory`` into one ledger."""
    root = Path(directory)
    rows: list[LedgerRow] = []
    all_models: list[ModelCost] = []
    for session in load_project(root):
        row, per_model = session_row(session, root)
        rows.append(row)
        all_models.extend(per_model)

    rows.sort(key=lambda r: r.cost_usd, reverse=True)
    return ProjectLedger(
        root=root.name or str(root),
        rows=rows,
        by_day=_fold_days(rows),
        by_model=_fold_models(all_models),
    )
