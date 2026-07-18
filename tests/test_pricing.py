"""Cost model: usage + price table -> USD, or None where either side is missing."""

from __future__ import annotations

from datetime import date

import pytest

import cc_session_core.cost.pricing as pricing_module
from cc_session_core.cost.pricing import (
    EXAMPLE_PRICING,
    ModelPrice,
    cost_for,
    fast_price,
    price_for_model,
    price_for_usage,
    request_cost,
    web_search_cost,
)
from cc_session_core.models import CacheCreation, ServerToolUse, Usage


def _usage() -> Usage:
    return Usage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=300,
        cache_creation=CacheCreation(ephemeral_1h_input_tokens=100, ephemeral_5m_input_tokens=200),
    )


def test_cost_for_known_model_computes_expected_total() -> None:
    # sonnet-4-6 resolves to the stable $3/$15 sonnet family rate (it has no exact-id
    # override; sonnet-5 is the only intro exception, priced until 2026-09-01).
    price = price_for_usage("claude-sonnet-4-6", _usage())
    cost = cost_for(_usage(), price)
    # (1000*3.0 + 500*15.0 + 200*0.30 + 200*3.75 + 100*6.0) / 1e6
    assert cost is not None
    assert abs(cost - 0.01191) < 1e-9


def test_sonnet_5_uses_intro_pricing() -> None:
    assert EXAMPLE_PRICING["claude-sonnet-5"].input == 2.0
    assert EXAMPLE_PRICING["claude-sonnet-5"].output == 10.0


def test_sonnet_5_reverts_to_the_standard_rate_after_the_intro_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the "after 2026-08-31" branch without waiting for the calendar.
    monkeypatch.setattr(pricing_module, "_SONNET_5_INTRO_END", date(2020, 1, 1))
    price = price_for_model("claude-sonnet-5", EXAMPLE_PRICING)
    assert price is not None
    assert price.input == 3.0
    assert price.output == 15.0


def test_sonnet_5_reversion_never_overrides_a_callers_own_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A caller processing real invoices passes their own table (the module's own
    # docstring says to); even with the same key, the date gate must never rewrite it.
    monkeypatch.setattr(pricing_module, "_SONNET_5_INTRO_END", date(2020, 1, 1))
    custom_table = {
        "claude-sonnet-5": ModelPrice(
            input=1.23, output=4.56, cache_write_5m=0.0, cache_write_1h=0.0, cache_read=0.0
        )
    }
    price = price_for_model("claude-sonnet-5", custom_table)
    assert price is not None
    assert price.input == 1.23
    assert price.output == 4.56


def test_fast_mode_pricing() -> None:
    opus8, opus7 = fast_price("claude-opus-4-8"), fast_price("claude-opus-4-7")
    assert opus8 is not None and opus8.input == 10.0
    assert opus7 is not None and opus7.output == 150.0
    assert fast_price("claude-sonnet-5") is None  # no fast rate for sonnet
    fast_usage = Usage(
        input_tokens=1000,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0),
        speed="fast",
    )
    fast, std = (
        price_for_usage("claude-opus-4-8", fast_usage),
        price_for_usage("claude-opus-4-8", _usage()),
    )
    assert fast is not None and fast.input == 10.0  # fast wins
    assert std is not None and std.input == 5.0  # standard otherwise


def test_web_search_cost_and_request_cost() -> None:
    usage = Usage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0),
        server_tool_use=ServerToolUse(web_search_requests=3, web_fetch_requests=0),
    )
    assert web_search_cost(usage) == 0.03
    # request_cost adds the search fee on top of (here zero) token cost
    assert request_cost("claude-opus-4-8", usage) == 0.03


def test_request_cost_applies_us_inference_surcharge_to_tokens_not_search() -> None:
    # The surcharge is inference-only: it multiplies token cost but not the flat
    # per-request web-search fee.
    base = request_cost(
        "claude-opus-4-8",
        Usage(
            input_tokens=1000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0),
            server_tool_use=ServerToolUse(web_search_requests=3, web_fetch_requests=0),
        ),
    )
    us = request_cost(
        "claude-opus-4-8",
        Usage(
            input_tokens=1000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0),
            server_tool_use=ServerToolUse(web_search_requests=3, web_fetch_requests=0),
            inference_geo="us",
        ),
    )
    assert base is not None and us is not None
    token_cost = base - 0.03  # strip the flat search fee both share
    assert us == round(token_cost * 1.1 + 0.03, 10)
    assert us != base


def test_price_for_unknown_model_is_none() -> None:
    assert price_for_model("not-a-real-model", EXAMPLE_PRICING) is None


def test_cost_for_missing_price_is_none() -> None:
    assert cost_for(_usage(), None) is None


def test_cost_for_missing_usage_is_none() -> None:
    price = price_for_model("claude-sonnet-5", EXAMPLE_PRICING)
    assert cost_for(None, price) is None


def test_price_for_none_model_is_none() -> None:
    assert price_for_model(None, EXAMPLE_PRICING) is None
