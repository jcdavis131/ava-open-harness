"""
perplexity.py — per-phase perplexity eval (one eval per file; probes/needle live in their own files)

Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
from typing import Any, Dict, List
from ..registry import register_eval
from ..common import MockModel, real_unimplemented
import random

@register_eval(name="perplexity", description="Per-phase PPL on heldout bins", group="core")
def perplexity(model: Any, tokenizer: Any, device: str="cpu", phases: List[int] | None = None, **kw) -> Dict[str,Any]:
    phases = phases or list(range(6))
    if not isinstance(model, MockModel):
        return real_unimplemented(
            "perplexity", "avg<30",
            "live NLL sum over frozen heldout bins per phase — previous "
            "'18.0 - phase*0.8' formula was fabricated",
        )
    results = {}
    for ph in phases:
        random.seed(model.seed + ph)
        ppl = random.uniform(12.0, 35.0) - ph*1.5  # decreasing with phase
        results[f"phase_{ph}"] = {"ppl": ppl, "tokens": 200000}
    avg = sum(v["ppl"] for v in results.values())/len(results)
    return {"test":"perplexity", "measured": {"per_phase": results, "avg_ppl": avg}, "pass": avg<30, "bar":"avg<30"}
