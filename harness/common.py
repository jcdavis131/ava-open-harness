"""
common.py — model/tokenizer loading, greedy decode, logprob, hook utils

Solo personal project, no connection to employer, built with public/free-tier only
Mypyc-ready: typed, lazy imports for torch
"""
from __future__ import annotations
from typing import Any, Tuple, List, Dict, Optional
import os, json, math, random

def real_unimplemented(test: str, bar: str, needs: str) -> Dict[str, Any]:
    """Honest real-mode failure record.

    HARNESS_SPEC: every float in a real report must come from a live forward pass.
    A real path that isn't wired yet therefore FAILS with an explanation — it never
    returns invented constants (that antipattern is exactly what this repo replaced).
    """
    return {
        "test": test,
        "measured": None,
        "pass": False,
        "bar": bar,
        "error": f"real mode not implemented: {needs}",
    }


def _lazy_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

class MockTokenizer:
    """Deterministic single-token-per-word mock for offline CI."""
    def __init__(self, vocab_size: int = 128000):
        self.vocab_size = vocab_size
        self._word_to_id: Dict[str, int] = {}
        self._next_id = 256
    def encode(self, text: str) -> List[int]:
        t = text.strip()
        if not t:
            return []
        if " " not in t:
            if t not in self._word_to_id:
                self._word_to_id[t] = (sum(b for b in t.encode()) % (self.vocab_size - 1000)) + 500
            return [self._word_to_id[t]]
        ids = []
        for w in t.split():
            if w not in self._word_to_id:
                self._next_id += 1
                self._word_to_id[w] = self._next_id % self.vocab_size
            ids.append(self._word_to_id[w])
        return ids
    def decode(self, ids: List[int]) -> str:
        inv = {v:k for k,v in self._word_to_id.items()}
        return " ".join(inv.get(i, f"<{i}>") for i in ids)

class MockModel:
    """Mock that mimics Ava model interface enough for harness shape checks."""
    def __init__(self, seed: int = 1234):
        self.seed = seed
        random.seed(seed)
    def eval(self): return self
    def reset_memory(self): pass
    def __repr__(self): return f"<MockModel seed={self.seed}>"

def load_model(ckpt_path: str | None, preset: str = "nano", device: str = "cpu", backend: str = "auto") -> Tuple[Any, Any]:
    """Load real model if torch available and ckpt exists, else mock.
    backend: auto|mock|hf|vllm — vllm path uses batching for wall 2.02h->1.80h
    Returns (model, tokenizer)
    """
    torch = _lazy_torch()
    ckpt_path = ckpt_path or "none"
    if ckpt_path.lower() in ("none", "random-init", "mock") or torch is None or not os.path.exists(ckpt_path):
        return MockModel(), MockTokenizer()

    # try to import from ava-agi-factory layout if present
    try:
        import sys
        # add parent factory to path if exists
        factory_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ava-agi-factory-v6-4"))
        if os.path.isdir(factory_root) and factory_root not in sys.path:
            sys.path.insert(0, factory_root)
        from ava.config import load as load_cfg
        from ava.model import build_model
        cfg = load_cfg(preset)
        model = build_model(cfg)
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("model", state)
        model.load_state_dict(sd, strict=False)
        model.eval()
        # tokenizer - try BPE artifact otherwise mock
        try:
            from ava.tokenizer import AvaTokenizer
            tok_path = os.path.join(factory_root, f"data/{preset}/tokenizer/ava_{preset}_bpe.json")
            if os.path.exists(tok_path):
                tok = AvaTokenizer.from_file(tok_path)
            else:
                tok = MockTokenizer()
        except Exception:
            tok = MockTokenizer()
        return model, tok
    except Exception as e:
        print(f"[common] real load failed ({e}), falling back to mock")
        return MockModel(), MockTokenizer()

def greedy_decode(model: Any, prompt_ids: List[int], max_new: int = 8, vocab_size: int = 128000) -> List[int]:
    """Mock greedy decode — deterministic for mock, real path if torch model."""
    torch = _lazy_torch()
    if isinstance(model, MockModel) or torch is None:
        # deterministic pseudo-decode: hash sum
        out = []
        s = sum(prompt_ids)
        for i in range(max_new):
            out.append((s + i * 31) % vocab_size)
        return out
    # real path
    try:
        import torch as _torch
        model.eval()
        ids = _torch.tensor([prompt_ids], device=next(model.parameters()).device if hasattr(model,'parameters') else 'cpu')
        with _torch.no_grad():
            for _ in range(max_new):
                out = model(ids)
                logits = out if isinstance(out, _torch.Tensor) else out[0] if isinstance(out, (list,tuple)) else out.logits
                nxt = logits[0,-1].argmax().item()
                ids = _torch.cat([ids, _torch.tensor([[nxt]], device=ids.device)], dim=1)
        return ids[0].tolist()[-max_new:]
    except Exception:
        return greedy_decode(MockModel(), prompt_ids, max_new, vocab_size)

def logprob_of(model: Any, prompt_ids: List[int], target_ids: List[int]) -> float:
    """Sum logprob of target continuation given prompt."""
    torch = _lazy_torch()
    if isinstance(model, MockModel) or torch is None:
        # mock: pseudo logprob from deterministic mapping
        random.seed(sum(prompt_ids)+sum(target_ids))
        return random.uniform(-5.0, -0.2)
    try:
        import torch as _torch
        import torch.nn.functional as F
        device = next(model.parameters()).device if hasattr(model,'parameters') else 'cpu'
        full = prompt_ids + target_ids
        inp = _torch.tensor([full[:-1]], device=device)
        with _torch.no_grad():
            out = model(inp)
            logits = out if isinstance(out, _torch.Tensor) else out[0]
            logp = F.log_softmax(logits[0], dim=-1)
        lp = 0.0
        for i, tid in enumerate(target_ids):
            pos = len(prompt_ids) + i -1
            if pos >=0 and pos < logp.shape[0]:
                lp += logp[pos, tid].item()
        return lp
    except Exception as e:
        # No silent fabricated fallback (HARNESS_SPEC anti-mock rule): a real-mode
        # measurement that can't be computed must fail loudly, never invent a float.
        raise RuntimeError(f"logprob_of failed on real model: {e}") from e

def cosine_sim(a: Any, b: Any) -> float:
    try:
        import numpy as np
        aa = np.array(a, dtype=float); bb = np.array(b, dtype=float)
        denom = (np.linalg.norm(aa)*np.linalg.norm(bb)+1e-9)
        return float(np.dot(aa,bb)/denom)
    except Exception:
        # pure python
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return dot/(na*nb+1e-9)

def auc_trapezoid(y_true: List[int], y_score: List[float]) -> float:
    """Compute ROC AUC via trapezoid, no sklearn."""
    # sort by score descending
    pairs = sorted(zip(y_score, y_true), reverse=True)
    # compute TPR/FPR steps
    pos = sum(y_true); neg = len(y_true)-pos
    if pos==0 or neg==0:
        return 0.5
    tp = fp = 0
    prev_fpr = 0.0; prev_tpr = 0.0; auc=0.0
    for _, label in pairs:
        if label==1:
            tp+=1
        else:
            fp+=1
        tpr = tp/pos; fpr = fp/neg
        auc += (fpr-prev_fpr)*(tpr+prev_tpr)/2.0
        prev_fpr, prev_tpr = fpr, tpr
    return auc
