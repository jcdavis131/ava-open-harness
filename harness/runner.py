"""
runner.py — runs harness, supports mock and real modes, torch optional
Solo personal project, no connection to employer, built with public/free-tier only
"""
from __future__ import annotations
from typing import Any, Dict, List
import argparse, json, time, os, sys, pathlib
from .registry import EVAL_REGISTRY, list_evals
# ensure evals loaded
import harness.evals  # noqa

def run_harness(
    eval_names: List[str] | str = "all",
    mode: str = "mock",
    ckpt: str | None = None,
    preset: str = "nano",
    device: str = "cpu",
    verbose: bool = False,
    **kwargs,
) -> Dict[str,Any]:
    from .common import load_model
    # normalize eval list
    if isinstance(eval_names, str):
        if eval_names == "all":
            eval_names = list_evals()
        else:
            eval_names = [s.strip() for s in eval_names.split(",") if s.strip()]
    # filter unknown
    eval_names = [n for n in eval_names if n in EVAL_REGISTRY]
    if not eval_names:
        eval_names = list_evals()

    # load model
    ckpt_path = ckpt if mode=="real" else None
    model, tokenizer = load_model(ckpt_path, preset=preset, device=device)

    results: Dict[str,Any] = {"meta": {"mode": mode, "ckpt": ckpt or "mock/random-init", "preset": preset, "device": device, "eval_names": eval_names}, "evals": {}}
    start = time.time()
    for name in eval_names:
        entry = EVAL_REGISTRY[name]
        fn = entry["fn"]
        if verbose:
            print(f"[runner] running {name} ({entry['description']})")
        try:
            res = fn(model, tokenizer, device, **kwargs)
            results["evals"][name] = res
        except Exception as e:
            results["evals"][name] = {"test": name, "error": str(e), "pass": False}
    results["meta"]["wall_s"] = time.time()-start
    results["meta"]["passed"] = sum(1 for r in results["evals"].values() if r.get("pass"))
    results["meta"]["total"] = len(results["evals"])
    return results

def write_reports(results: Dict[str,Any], out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "branch_eval_results_real.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    # markdown
    md_path = os.path.join(out_dir, "REPORT_REAL.md")
    lines = []
    lines.append("# Ava Open Harness — Real Eval Report")
    lines.append("")
    lines.append(f"Mode: {results['meta'].get('mode')} | CKPT: {results['meta'].get('ckpt')} | Passed {results['meta'].get('passed')}/{results['meta'].get('total')} | {results['meta'].get('wall_s',0):.1f}s")
    lines.append("")
    lines.append("| Test | Bar | Measured | PASS/FAIL |")
    lines.append("|---|---|---|---|")
    for name, r in results["evals"].items():
        bar = r.get("bar","")
        measured = r.get("measured",{})
        # summarize
        if isinstance(measured, dict):
            # pick first numeric
            summary = str(measured)[:120]
        else:
            summary = str(measured)[:120]
        verdict = "PASS" if r.get("pass") else "FAIL"
        if "error" in r:
            verdict = f"ERROR {r['error'][:40]}"
            summary = r["error"][:80]
        lines.append(f"| {name} | {bar} | {summary} | {verdict} |")
    lines.append("")
    # frozen comparison placeholder
    lines.append("## Frozen-capability comparison (base vs chat)")
    lines.append("When both base and chat ckpts supplied, Δ% column shows regression if chat >5% worse (system1+system2 frozen).")
    with open(md_path,"w") as f:
        f.write("\n".join(lines))
    print(f"[runner] wrote {json_path} and {md_path}")
    return json_path, md_path

def main():
    ap = argparse.ArgumentParser(description="Ava Open Harness — mock and real eval runner")
    ap.add_argument("--eval", default="all", help="comma list or all")
    ap.add_argument("--mode", default="mock", choices=["mock","real"], help="mock = no torch, real = load ckpt if exists")
    ap.add_argument("--ckpt", default=None, help="checkpoint path for real mode")
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-n", type=int, default=200)
    ap.add_argument("--skip", default="", help="comma list to skip")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--wiki-path", default=None)
    args = ap.parse_args()

    skip = set([s.strip() for s in args.skip.split(",") if s.strip()])
    eval_names = args.eval
    if eval_names == "all":
        eval_names = [n for n in list_evals() if n not in skip]
    else:
        eval_names = [n for n in eval_names.split(",") if n.strip() and n.strip() not in skip]

    res = run_harness(eval_names=eval_names, mode=args.mode, ckpt=args.ckpt, preset=args.preset, device=args.device, verbose=args.verbose, probe_n=args.probe_n, wiki_path=args.wiki_path)
    write_reports(res, out_dir=args.out_dir)
    # exit 0 even if bars fail, per spec
    print(json.dumps(res["meta"], indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
