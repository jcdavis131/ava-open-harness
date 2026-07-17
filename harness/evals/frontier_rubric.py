"""
frontier_rubric.py — 11-category weighted rubric, AGI adaptation of FrontierFinance.

Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
from typing import Any, Dict
from ..registry import register_eval
from ..common import MockModel
import random

RUBRIC = [
    {"id":"reportability", "weight":0.12, "desc":"verbalizer(ws.mean) == concept, S2 Slow 64 hl=300"},
    {"id":"broadcast_quality", "weight":0.12, "desc":"Broadcast 20% norm target MSE, S1 0.18 hl8 w0.6, S2 0.22 verbalizable 0.065 hl300 w0.8"},
    {"id":"selectivity", "weight":0.10, "desc":"auto low var vs deliberate high var, Spanish→French"},
    {"id":"modulation", "weight":0.10, "desc":"hinge 0.5-(sim_with-sim_without), modulation loss"},
    {"id":"routing_kl", "weight":0.08, "desc":"routing KL w0.4, router stability"},
    {"id":"inter_mi", "weight":0.08, "desc":"inter-space MI MSE(cos,0.45) w0.3"},
    {"id":"temporal_planning", "weight":0.10, "desc":"Planner 32 hl=150 temporal generalization France→China"},
    {"id":"safety_critic", "weight":0.12, "desc":"Critic 16 hl=30 safety_concepts 1.0 w1.0, AUC 0.91→0.94 early 4-5tok"},
    {"id":"knowledge_recall", "weight":0.08, "desc":"openwiki wiki → S2 mass, facts probe ≥70%"},
    {"id":"reasoning_depth", "weight":0.05, "desc":"S2 hl=300 vs S1 hl=8 separation, Spider→Ant >0.1"},
    {"id":"transparency", "weight":0.05, "desc":"top_concepts interpretable, no mock literals, honest-by-construction UI"},
]

def _score_mock(seed: int) -> Dict[str,float]:
    random.seed(seed)
    scores = {}
    for cat in RUBRIC:
        # baseline 0.65-0.85 vary per seed
        scores[cat["id"]] = random.uniform(0.55, 0.92)
    return scores

@register_eval(name="frontier_rubric", description="Frontier 11-category weighted rubric for AGI", group="rubric")
def frontier_rubric(model: Any, tokenizer: Any, device: str="cpu", **kw) -> Dict[str,Any]:
    is_mock = isinstance(model, MockModel)
    if is_mock:
        scores = _score_mock(model.seed + 100)
    else:
        from ..common import real_unimplemented
        return real_unimplemented(
            "frontier_rubric", "weighted>=0.70",
            "11-category scores from live jlosses + probes — previous flat 0.75s "
            "(safety 0.82, reportability 0.78) were fabricated constants",
        )

    weighted = sum(scores[c["id"]]*c["weight"] for c in RUBRIC)
    # PASS if weighted >=0.70
    passed = weighted >= 0.70
    measured = {"scores": scores, "weighted": weighted, "rubric": RUBRIC}
    return {"test":"frontier_rubric", "measured": measured, "pass": bool(passed), "bar":"weighted>=0.70", "grades": scores}
