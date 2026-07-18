"""Typed cost model for Claude Code transcript usage.

Generalizes the ad-hoc ``RATES``/``cost_of`` in ``contextmap.py`` into a
reusable ``PriceTable`` and a pure ``cost_for`` that prices a ``Usage`` record.

The rates in ``EXAMPLE_PRICING`` are published list rates for usage valuation,
not billed amounts — verify before trusting a dollar figure derived from them,
and callers processing real invoices should pass their own table.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, computed_field

from .. import types as t
from ..models import Usage

TOKENS_PER_MILLION = 1_000_000


class ModelPrice(BaseModel):
    model_config = t.SNAKE_CONFIG

    input: t.UsdPerMillion
    output: t.UsdPerMillion
    cache_write_5m: t.UsdPerMillion
    cache_write_1h: t.UsdPerMillion
    cache_read: t.UsdPerMillion


PriceTable = dict[t.ModelId, ModelPrice]

# Exact-id price *overrides* only — every current opus/sonnet/haiku/fable id already
# resolves to the same rate through FAMILY_PRICING (see :func:`price_for_usage`), so the
# only entry that must live here is the one that differs from its family: Sonnet 5's
# introductory rate ($2/$10), in effect through 2026-08-31. price_for_model date-gates
# its reversion to the sonnet family rate on 2026-09-01.
# USD per 1M tokens; cache_write_5m = 1.25x input, cache_write_1h = 2x input, cache_read = 0.1x.
EXAMPLE_PRICING: PriceTable = {
    "claude-sonnet-5": ModelPrice(
        input=2.0, output=10.0, cache_write_5m=2.50, cache_write_1h=4.0, cache_read=0.20
    ),
}


# Sonnet 5's introductory rate is in effect through 2026-08-31; on 2026-09-01 it
# reverts to the standard sonnet family rate. Resolution is date-gated so the
# reversion happens automatically rather than needing a manual table edit.
_SONNET_5 = "claude-sonnet-5"
_SONNET_5_INTRO_END = date(2026, 8, 31)
_SONNET_5_STANDARD = ModelPrice(
    input=3.0, output=15.0, cache_write_5m=3.75, cache_write_1h=6.0, cache_read=0.30
)


def price_for_model(model: t.ModelId | None, table: PriceTable) -> ModelPrice | None:
    """Resolve a model id in ``table``. The Sonnet 5 intro-rate reversion is gated on
    today's wall-clock date and applies only to :data:`EXAMPLE_PRICING` itself (an
    identity check, not equality) — a caller's own price table, even one keyed by the
    same model id, is never silently rewritten out from under it. Wall-clock gating
    also means this prices by *today's* rate, not the rate in effect when the usage
    was actually billed; a timestamp-aware variant would need one on the call, which
    :func:`request_cost` doesn't currently take."""
    if model is None:
        return None
    price = table.get(model)
    if (
        price is not None
        and model == _SONNET_5
        and table is EXAMPLE_PRICING
        and datetime.now(UTC).date() > _SONNET_5_INTRO_END
    ):
        return _SONNET_5_STANDARD
    return price


class CostBreakdown(BaseModel):
    """One request's cost, split by token kind — the same categories a Sankey or
    per-kind report groups by, so callers never re-derive the pricing formula."""

    model_config = t.SNAKE_CONFIG

    input: t.CostUsd
    output: t.CostUsd
    cache_read: t.CostUsd
    cache_write_5m: t.CostUsd
    cache_write_1h: t.CostUsd

    @computed_field
    @property
    def total(self) -> t.CostUsd:
        return (
            self.input + self.output + self.cache_read + self.cache_write_5m + self.cache_write_1h
        )


def cost_breakdown_for(usage: Usage | None, price: ModelPrice | None) -> CostBreakdown | None:
    if usage is None or price is None:
        return None
    # Usage.cache_creation is always present (non-optional), so its 5m/1h split is
    # authoritative for cache-write cost.
    return CostBreakdown(
        input=usage.input_tokens * price.input / TOKENS_PER_MILLION,
        output=usage.output_tokens * price.output / TOKENS_PER_MILLION,
        cache_read=usage.cache_read_input_tokens * price.cache_read / TOKENS_PER_MILLION,
        cache_write_5m=usage.cache_creation.ephemeral_5m_input_tokens
        * price.cache_write_5m
        / TOKENS_PER_MILLION,
        cache_write_1h=usage.cache_creation.ephemeral_1h_input_tokens
        * price.cache_write_1h
        / TOKENS_PER_MILLION,
    )


def cost_for(usage: Usage | None, price: ModelPrice | None) -> t.CostUsd | None:
    breakdown = cost_breakdown_for(usage, price)
    return breakdown.total if breakdown is not None else None


# --------------------------------------------------------------------------- #
# family-based pricing — resolve any model id to a price by family, so a dated
# snapshot or future point release prices off its family even when it is absent
# from the exact-id table.
# --------------------------------------------------------------------------- #
Family = Literal["opus", "opus_legacy", "sonnet", "haiku", "fable"]

# USD per 1M tokens; cache_write_5m = 1.25x input, cache_write_1h = 2x input,
# cache_read = 0.1x input — Anthropic's published list rates.
FAMILY_PRICING: dict[Family, ModelPrice] = {
    "opus": ModelPrice(
        input=5.0, output=25.0, cache_write_5m=6.25, cache_write_1h=10.0, cache_read=0.50
    ),
    "opus_legacy": ModelPrice(
        input=15.0, output=75.0, cache_write_5m=18.75, cache_write_1h=30.0, cache_read=1.50
    ),
    "sonnet": ModelPrice(
        input=3.0, output=15.0, cache_write_5m=3.75, cache_write_1h=6.0, cache_read=0.30
    ),
    "haiku": ModelPrice(
        input=1.0, output=5.0, cache_write_5m=1.25, cache_write_1h=2.0, cache_read=0.10
    ),
    "fable": ModelPrice(
        input=10.0, output=50.0, cache_write_5m=12.5, cache_write_1h=20.0, cache_read=1.0
    ),
}


def family_of(model: str | None) -> Family | None:
    """The pricing family for a Claude model id, by substring — so any opus/sonnet/haiku/
    fable variant (dated snapshots, future point releases) prices, not just exact ids."""
    if model is None:
        return None
    name = model.lower()
    if "haiku" in name:
        return "haiku"
    if "sonnet" in name:
        return "sonnet"
    if "opus" in name:
        legacy = ("opus-4-1", "opus-4-0", "opus-4.1", "opus-4.0")
        return "opus_legacy" if any(tag in name for tag in legacy) else "opus"
    if "fable" in name or "mythos" in name:
        return "fable"
    return None


def price_for_family(
    model: str | None, table: dict[Family, ModelPrice] = FAMILY_PRICING
) -> ModelPrice | None:
    """Resolve a model id to its family price (see :func:`family_of`)."""
    family = family_of(model)
    return table.get(family) if family is not None else None


# --------------------------------------------------------------------------- #
# fast mode + server-tool pricing
# --------------------------------------------------------------------------- #
# Fast mode (``Usage.speed == "fast"``, research preview) charges a premium across
# the full context window; only Opus 4.8 / 4.7 support it. Cache tiers follow the
# standard 1.25x / 2x / 0.1x multipliers on the fast input rate.
_FAST_MODE_PRICING: dict[str, ModelPrice] = {
    "opus-4-8": ModelPrice(
        input=10.0, output=50.0, cache_write_5m=12.5, cache_write_1h=20.0, cache_read=1.0
    ),
    "opus-4-7": ModelPrice(
        input=30.0, output=150.0, cache_write_5m=37.5, cache_write_1h=60.0, cache_read=3.0
    ),
}

# Web search is billed per search on top of token cost: $10 per 1,000 searches.
WEB_SEARCH_USD_PER_REQUEST = 10.0 / 1000


def fast_price(model: str | None) -> ModelPrice | None:
    """Fast-mode (``speed="fast"``) pricing for the models that support it, else None."""
    if model is None:
        return None
    name = model.lower()
    for tag, price in _FAST_MODE_PRICING.items():
        if tag in name:
            return price
    return None


def price_for_usage(
    model: t.ModelId | None, usage: Usage, table: PriceTable = EXAMPLE_PRICING
) -> ModelPrice | None:
    """Resolve the price for a request, honoring fast mode: when ``usage.speed`` is
    ``"fast"`` and the model has a fast rate that wins; otherwise the exact-model
    price, then the family price."""
    if usage.speed == "fast":
        fast = fast_price(model)
        if fast is not None:
            return fast
    return price_for_model(model, table) or price_for_family(model)


_US_INFERENCE_SURCHARGE = 1.1  # us-region inference is billed at 1.1x the list rate
_US_GEO = "us"


def web_search_cost(usage: Usage) -> t.CostUsd:
    """USD for the server-side web searches in a request ($10 per 1,000)."""
    server = usage.server_tool_use
    requests = server.web_search_requests if server is not None else 0
    return requests * WEB_SEARCH_USD_PER_REQUEST


def request_cost(
    model: t.ModelId | None, usage: Usage, table: PriceTable = EXAMPLE_PRICING
) -> t.CostUsd | None:
    """Total cost of one request: token cost (fast-mode and US-inference-surcharge aware)
    plus web-search fees. The surcharge applies to inference only, not the flat
    per-request web-search fee."""
    token_cost = cost_for(usage, price_for_usage(model, usage, table))
    if token_cost is None:
        return None
    if usage.inference_geo == _US_GEO:
        token_cost *= _US_INFERENCE_SURCHARGE
    return token_cost + web_search_cost(usage)
