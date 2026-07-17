"""
needle.py — pass-key retrieval eval (one eval per file; perplexity/probes live in their own files)

Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
from typing import Any, Dict
from ..registry import register_eval
from ..common import MockModel, real_unimplemented
import random

@register_eval(name="needle", description="Pass-key retrieval depth 0.1..0.9 with YaRN scaling", group="core")
def needle(model: Any, tokenizer: Any, device: str="cpu", **kw) -> Dict[str,Any]:
    if not isinstance(model, MockModel):
        return real_unimplemented(
            "needle", "avg>=0.70",
            "live pass-key retrieval at depths 0.1..0.9 over real contexts — the "
            "previous depth-thresholded accuracy constants were fabricated",
        )
    depths = [0.1,0.3,0.5,0.7,0.9]
    results = {}
    for d in depths:
        random.seed(model.seed + int(d*10))
        acc = random.uniform(0.7, 0.95) if d<0.6 else random.uniform(0.5,0.8)
        results[str(d)] = acc
    avg = sum(results.values())/len(results)
    return {"test":"needle", "measured": {"per_depth": results, "avg": avg, "contexts":[1024,2048]}, "pass": avg>=0.70, "bar":"avg>=0.70"}
