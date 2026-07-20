"""evals package — imports all evals to trigger registration."""

from __future__ import annotations

import importlib

from ..registry import EVAL_REGISTRY

# trigger registration
for mod in [
    "jspace_tests",
    "frontier_rubric",
    "openwiki_knowledge",
    "perplexity",
    "probes",
    "needle",
]:
    try:
        importlib.import_module(f"harness.evals.{mod}")
    except Exception as e:
        print(f"[evals] failed to load {mod}: {e}")

__all__ = ["EVAL_REGISTRY"]
