"""
openwiki_knowledge.py — test if model recalls facts from wiki (openwiki personal → S2)

Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
from typing import Any, Dict, List
from ..registry import register_eval
from ..common import MockModel
import os, random, pathlib

def scan_wiki(wiki_path: str | None = None) -> List[pathlib.Path]:
    candidates = []
    if wiki_path and os.path.isdir(wiki_path):
        candidates.append(pathlib.Path(wiki_path))
    # default locations
    home = pathlib.Path.home()
    candidates.extend([
        home / ".openwiki" / "wiki",
        pathlib.Path.cwd() / "openwiki",
        pathlib.Path.cwd().parent / "family-brain-os" / "src" / "wiki",  # family brain inspired version
    ])
    found = []
    for p in candidates:
        if p.exists():
            found.extend(list(p.rglob("*.md"))[:50])
    return found

@register_eval(name="openwiki_knowledge", description="Wiki recall: S2 reportability mass for openwiki pages", group="knowledge")
def openwiki_knowledge(model: Any, tokenizer: Any, device: str="cpu", wiki_path: str | None = None, **kw) -> Dict[str,Any]:
    # Real-mode short-circuit BEFORE the filesystem walk (no wasted scan).
    if not isinstance(model, MockModel):
        from ..common import real_unimplemented
        return real_unimplemented(
            "openwiki_knowledge", "mass>=0.06",
            "live S2 recall mass over wiki concept probes — the previous recall constant / "
            "per-file random.uniform were fabricated",
        )
    wiki_files = scan_wiki(wiki_path)
    if not wiki_files:
        random.seed(model.seed)  # reproducible across runs (was unseeded)
        measured = {"n_wiki_files": 0, "recall_mass": random.uniform(0.05, 0.15),
                    "note": "no wiki found, using mock corpus from specs/02_data"}
        passed = measured["recall_mass"] >= 0.06
        return {"test":"openwiki_knowledge", "measured": measured, "pass": bool(passed), "bar":"mass>=0.06"}

    # mock recall: each wiki page title -> seeded concept probe
    scores = []
    for f in wiki_files[:20]:
        random.seed(hash(f.name) % 10000 + model.seed)
        scores.append(random.uniform(0.04, 0.18))
    avg = sum(scores)/len(scores) if scores else 0.0
    measured = {"n_wiki_files": len(wiki_files), "sampled": len(scores), "recall_mass": avg, "files": [str(p) for p in wiki_files[:5]]}
    return {"test":"openwiki_knowledge", "measured": measured, "pass": avg>=0.06, "bar":"mass>=0.06"}
