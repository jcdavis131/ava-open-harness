"""
probes.py — capability probes eval (one eval per file; perplexity/needle live in their own files)

Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
from typing import Any, Dict
from ..registry import register_eval
from ..common import MockModel, real_unimplemented
import random, zlib


def _stable_seed(name: str) -> int:
    """Process-stable per-name seed. Python's builtin hash() is salted by
    PYTHONHASHSEED, so hash(name) makes 'seeded' mock draws differ every run."""
    return zlib.crc32(name.encode()) % 100

@register_eval(name="probes", description="Arithmetic, modus_ponens, facts, code_out probes ≥200 items each", group="core")
def probes(model: Any, tokenizer: Any, device: str="cpu", probe_n: int = 200, **kw) -> Dict[str,Any]:
    if not isinstance(model, MockModel):
        return real_unimplemented(
            "probes", "arith≥60% and facts≥70%",
            "live accuracy over ≥200 items per probe set — previous constants "
            "0.72/0.55 were fabricated",
        )
    probe_sets = ["arithmetic","modus_ponens","facts","code_out"]
    scores = {}
    for name in probe_sets:
        random.seed(model.seed + _stable_seed(name))
        if name=="arithmetic":
            acc = random.uniform(0.55, 0.85)
        elif name=="facts":
            acc = random.uniform(0.60, 0.85)
        else:
            acc = random.uniform(0.40, 0.75)
        scores[name] = acc
    passed = scores["arithmetic"]>=0.60 and scores["facts"]>=0.70
    return {"test":"probes", "measured": {"acc": scores, "n_per_set": probe_n}, "pass": bool(passed), "bar":"arith≥60% and facts≥70%"}
