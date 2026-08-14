"""
tests/test_strategy_factory.py

Tests for src/summarization/strategy_factory.py.

No mocking needed -- this module has no external dependencies (no
network, no LLM calls), just Python object construction. Each test
proves one specific claim about the registry's behavior.
"""

import pytest

from src.summarization.map_reduce_strategy import MapReduceStrategy
from src.summarization.refine_strategy import RefineStrategy
from src.summarization.strategy_factory import (
    UnknownStrategyError,
    available_strategies,
    get_strategy,
)
from src.summarization.stuff_strategy import StuffStrategy

# ---------------------------------------------------------------------------
# available_strategies
# ---------------------------------------------------------------------------


def test_available_strategies_returns_all_three_registered_names():
    assert available_strategies() == ["stuff", "map_reduce", "refine"]


def test_available_strategies_returns_a_new_list_each_time():
    # Guards against a subtle bug: if available_strategies() ever
    # returned a reference to the registry's internal list/keys view
    # instead of a fresh copy, a caller mutating the returned list
    # (e.g. app.py doing .sort() or .append()) could corrupt the
    # factory's internal state for every subsequent call.
    first_call = available_strategies()
    first_call.append("not_a_real_strategy")

    assert available_strategies() == ["stuff", "map_reduce", "refine"]


# ---------------------------------------------------------------------------
# get_strategy -- correct type returned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("stuff", StuffStrategy),
        ("map_reduce", MapReduceStrategy),
        ("refine", RefineStrategy),
    ],
)
def test_get_strategy_returns_correct_type(name, expected_class):
    strategy = get_strategy(name)
    assert isinstance(strategy, expected_class)
    assert strategy.name == name


def test_get_strategy_returns_a_new_instance_each_call():
    # Deliberately NOT cached (unlike get_llm()) -- proves that design
    # decision from the file walkthrough actually holds.
    first = get_strategy("stuff")
    second = get_strategy("stuff")

    assert first is not second
    assert isinstance(first, StuffStrategy)
    assert isinstance(second, StuffStrategy)


# ---------------------------------------------------------------------------
# get_strategy -- unknown name handling
# ---------------------------------------------------------------------------


def test_get_strategy_raises_on_unknown_name():
    with pytest.raises(UnknownStrategyError, match="Unknown summarization strategy"):
        get_strategy("not_a_real_strategy")


def test_get_strategy_error_lists_valid_options():
    with pytest.raises(UnknownStrategyError) as exc_info:
        get_strategy("bogus")

    message = str(exc_info.value)
    assert "stuff" in message
    assert "map_reduce" in message
    assert "refine" in message


def test_get_strategy_never_silently_falls_back_to_a_default(mocker):
    # Explicit regression guard for the design decision NOT to
    # silently default to Stuff (or any strategy) on an unrecognized
    # name -- spies on StuffStrategy's constructor to prove it's never
    # invoked when an unknown name is requested.
    spy = mocker.spy(StuffStrategy, "__init__")

    with pytest.raises(UnknownStrategyError):
        get_strategy("totally_made_up")

    spy.assert_not_called()


@pytest.mark.parametrize("bad_name", ["", "STUFF", "Stuff", " stuff", "stuff "])
def test_get_strategy_is_case_and_whitespace_sensitive(bad_name):
    # Documents actual behavior: names must match exactly. This is a
    # deliberate choice, not an oversight -- app.py's dropdown always
    # passes one of the exact registered names, so silently normalizing
    # here would hide a real bug elsewhere if it ever mismatched.
    with pytest.raises(UnknownStrategyError):
        get_strategy(bad_name)
