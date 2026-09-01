"""Induce one SCFG per protocol. OPT-IN; output already cached.

For each of the 45 baseline roots, GPT-5.5 is shown three things and asked to write a
synchronous grammar as JSON:

  the fixed referent inventory   the 37 things a message can denote (ontology.py)
  the postmortem discussion      where the agents explicitly designed their code
  up to 28 (message -> meaning)  pairs from the development rounds

One call per root, no refinement pass. The grammar is then scored for parse coverage
and its DL(G) computed, giving the baseline table the resume analysis reads.

You do NOT need to run this: the 45 induced grammars and mdl_table_joint.csv ship in
data/mdl/. Re-running costs roughly $17 and, being an LLM, will not reproduce exactly.

  export FOUNDRY_API_KEY=...       # or OPENAI_API_KEY
  python scripts/mdl/induce_grammars.py --out-dir /tmp/reinduce
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mdl  # noqa: E402
import ontology as ont  # noqa: E402
import scfg  # noqa: E402
from common import DATA  # noqa: E402

MDL_DATA = DATA / "mdl"
PROMPT = DATA / "prompts/scfg_induce.txt"
AZURE_ENDPOINT = "https://azure-credits-2026.services.ai.azure.com/openai/v1"
MODEL = "gpt-5.5"

TRAIN_MAX = 14      # induce only from the language-development rounds
MAX_PAIRS = 28
# the joint grammar spans both agents, so all four categorical axes and R = 37
AXES_LEX = ("motif", "template", "face", "intensity")
SLOTS = ("motif", "template", "face", "intensity", "duration")
REF_INV = 37


def client():
    key = os.environ.get("FOUNDRY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set FOUNDRY_API_KEY (or OPENAI_API_KEY) to run the inducer")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=AZURE_ENDPOINT)


def inventory() -> str:
    lines = ["MOTIFS (use exactly these as motif values):"]
    lines += [f"  {k}" for k in ont.MOTIFS]
    lines.append("TEMPLATE IDs (use exactly these as template values):")
    lines += [f"  {tid}: {text}" for tid, text in ont.TEMPLATES.items()]
    lines.append("FACES: " + ", ".join(ont.FACES))
    lines.append("INTENSITIES: " + ", ".join(ont.INTENSITIES))
    return "\n".join(lines)


def message_meaning_pairs(root: str) -> list:
    """(coded message, the meaning it should denote) over the development rounds."""
    rounds = json.loads((MDL_DATA / f"grounded/veyru_{root}.json").read_text())["rounds"]
    out = []
    for r in rounds:
        if r["round"] > TRAIN_MAX:
            continue
        for a in r["attempts"]:
            if a.get("engineer_msg"):
                c = a.get("correct") or {}
                gt = {k: c[k] for k in ("template", "face", "intensity", "duration")
                      if c.get(k) not in (None, "")}
                if gt:
                    out.append((a["engineer_msg"], gt))
            if a.get("observer_msg") and a.get("motif"):
                out.append((a["observer_msg"], {"motif": a["motif"]}))
    return out


def coverage(grammar, pairs) -> dict:
    """How often the grammar parses a message, and how often the parse is right."""
    n = parsed = correct = 0
    for msg, gt in pairs:
        if not msg:
            continue
        n += 1
        m = scfg.parse(grammar, msg)
        if not m:
            continue
        parsed += 1
        correct += int(all(m[0].get(k) == v for k, v in gt.items()))
    return {"n": n, "coverage": parsed / n if n else 0,
            "joint_parsed": correct / parsed if parsed else 0,
            "joint_all": correct / n if n else 0}


def agent_model(root: str) -> str:
    p = MDL_DATA / f"labels/{root}.json"
    if not p.exists():
        return "?"
    return dict(x.split("=", 1) for x in json.loads(p.read_text()) if "=" in x).get("model", "?")


def induce_one(root: str, cli, system: str, out_dir: Path) -> dict:
    pairs = message_meaning_pairs(root)
    gfile = out_dir / f"grammar_joint_{root}.json"
    if gfile.exists():
        spec = json.loads(gfile.read_text())      # cached; delete to force re-induction
    else:
        sample = pairs[:: max(1, len(pairs) // MAX_PAIRS)][:MAX_PAIRS]
        user = (inventory()
                + "\n\n=== POSTMORTEM DISCUSSION ===\n"
                + (MDL_DATA / f"postmortems/{root}.txt").read_text()
                + "\n\n=== EXAMPLE messages (text -> gold meaning) ===\n"
                + "\n".join(f"  {m!r} -> {json.dumps(gt)}" for m, gt in sample))
        raw = cli.chat.completions.create(
            model=MODEL, timeout=600,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]).choices[0].message.content
        spec = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        gfile.write_text(json.dumps(spec, indent=2))

    s = coverage(scfg.Grammar(spec), pairs)
    m = mdl.compute(spec, [msg for msg, _ in pairs],
                    axes_lex=AXES_LEX, slots=SLOTS, ref_inv=REF_INV)
    return {"run_id": root, "model": agent_model(root),
            "coverage": round(s["coverage"], 3),
            "joint_parsed": round(s["joint_parsed"], 3),
            "joint_all": round(s["joint_all"], 3), **m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roots = (MDL_DATA / "protocol_roots.txt").read_text().split()
    system = PROMPT.read_text()
    cli = client()

    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(induce_one, r, cli, system, args.out_dir): r for r in roots}
        for i, fut in enumerate(as_completed(futures), 1):
            root = futures[fut]
            try:
                rows[root] = fut.result()
                r = rows[root]
                print(f"[{i}/{len(roots)}] {root} ({r['model']}): coverage={r['coverage']} "
                      f"DL(G)={r['DL_G']}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(roots)}] {root}: ERROR {type(e).__name__}: {e}", flush=True)

    import pandas as pd
    out = args.out_dir / "mdl_table_joint.csv"
    pd.DataFrame([rows[r] for r in roots if r in rows]).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(rows)} protocols)")


if __name__ == "__main__":
    main()
