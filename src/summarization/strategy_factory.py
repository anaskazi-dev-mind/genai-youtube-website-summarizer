"""
src/summarization/strategy_factory.py

Single switch point mapping a strategy name (as chosen in the
Streamlit UI dropdown) to the correct SummarizationStrategy
implementation. This is the ONLY file that needs to change when a
new strategy is added later -- app.py and everything else depends
only on the SummarizationStrategy interface (base_strategy.py), never
on Stuff/Map-Reduce/Refine's internals directly.
"""

from typing import Dict, List, Type

from src.summarization.base_strategy import SummarizationStrategy
from src.summarization.map_reduce_strategy import MapReduceStrategy
from src.summarization.refine_strategy import RefineStrategy
from src.summarization.stuff_strategy import StuffStrategy

# Registry order also defines display order in the UI dropdown (see
# available_strategies()) -- Stuff first (simplest, fastest), then
# Map-Reduce, then Refine, matching the order they're introduced in
# this project's own documentation.
_STRATEGY_REGISTRY: Dict[str, Type[SummarizationStrategy]] = {
    StuffStrategy.name: StuffStrategy,
    MapReduceStrategy.name: MapReduceStrategy,
    RefineStrategy.name: RefineStrategy,
}


class UnknownStrategyError(ValueError):
    """Raised when an unrecognized strategy name is requested."""


def available_strategies() -> List[str]:
    """
    Returns the list of valid strategy names. Used by app.py to
    populate the Streamlit dropdown, so the UI's list of options is
    always derived from what's actually registered here -- never a
    separately hardcoded list that could drift out of sync.
    """
    return list(_STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> SummarizationStrategy:
    """
    Returns a NEW instance of the strategy registered under `name`.

    A fresh instance per call, not a cached singleton (unlike get_llm()
    or the embedding model) -- strategy objects hold no expensive
    resources themselves; the actual expensive resources (the Groq
    client, the HuggingFace model) are already singletons one layer
    down, in llm.py and deduplicator.py. Caching strategy instances
    here would add complexity with no real performance benefit.

    Raises UnknownStrategyError for any unrecognized name -- fails
    loudly rather than silently defaulting to some strategy, since a
    silent default could summarize with very different cost/latency
    characteristics than what the user actually selected in the UI.
    """
    strategy_class = _STRATEGY_REGISTRY.get(name)
    if strategy_class is None:
        raise UnknownStrategyError(
            f"Unknown summarization strategy: '{name}'. "
            f"Valid options are: {', '.join(available_strategies())}."
        )
    return strategy_class()
