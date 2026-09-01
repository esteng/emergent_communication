"""The GPT-5.5 metalinguistic annotation pass. OPT-IN; its output is already cached.

Labels every clarification candidate a swapped-in agent sent during the post-swap
rounds, grounded in that protocol's induced grammar so the judge knows how the code
realizes each slot:

  has_question   is this a METALINGUISTIC query (what does this symbol mean / decode
                 this), as opposed to a repeat request, a next-step question, or a
                 bare "?"
  target_scope   whole code, one constituent, or unidentifiable
  target_slot    template / face / duration / intensity / motif / bundle / ...
  compositional  does the queried target decompose into meaningful pieces?
  transparent    does the target's form relate to its meaning?

You do NOT need to run this: data/annotations/metaling_{instances,denominators}.csv
ship with the release and the downstream scripts read those. Re-running costs money
and, being an LLM judge, will not reproduce bit-for-bit.

  export FOUNDRY_API_KEY=...        # or OPENAI_API_KEY
  python scripts/transmission/annotate_metalinguistic.py --out-dir /tmp/reannotate
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402

GRAMMARS = DATA / "annotations/prompt_grammars"
PROMPT = DATA / "prompts/metalinguistic_annotation.txt"
AZURE_ENDPOINT = "https://azure-credits-2026.services.ai.azure.com/openai/v1"
MODEL = "gpt-5.5"

NEWCOMER_START = 15   # first post-swap round
CONTEXT_TURNS = 10    # preceding link turns shown to the judge
MAXLEN = 45           # candidates longer than this are not clarification requests
BATCH = 25
AGENT = {"field_observer": "Field Observer", "stabilization_engineer": "Engineer"}
KW = (r"mean|confirm|clarif|which|what|full|plain|spell|expand|repeat|again|unclear|"
      r"define|\btag\b|\bcode\b|format|understand|parse|resend|specif|explain|\bdesc\b|"
      r"guidance|advise|advice|next\?|how\b|sym")
SLOT_NAMES = {"MOTIF": "motif_symptom", "TMPL": "template", "PNUM": "template",
              "FACE": "face", "DUR": "duration", "INT": "intensity"}
SORT = ["run_id", "round_number", "substage", "message_index_in_substage"]


def client():
    key = os.environ.get("FOUNDRY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set FOUNDRY_API_KEY (or OPENAI_API_KEY) to run the annotator")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=AZURE_ENDPOINT)


def prompts() -> tuple[str, str]:
    text = PROMPT.read_text()
    system = text.split("# SYSTEM\n", 1)[1].split("\n\n# TASK", 1)[0].strip()
    task = text.split("filled per protocol)\n", 1)[1]
    return system, task


def grammar_summary(root: str) -> str:
    """Per-slot terminal realizations, so the judge sees how THIS protocol codes things."""
    f = GRAMMARS / f"grammar_joint_{root.split('/')[-1]}.json"
    if not f.exists():
        return "(no induced grammar available for this protocol)"
    g = json.loads(f.read_text())
    nonterminals = {r["lhs"] for r in g["rules"]}
    terms: dict[str, set] = {}
    for r in g["rules"]:
        if r["rhs"] and all(s not in nonterminals for s in r["rhs"]):
            slot = SLOT_NAMES.get(r["lhs"])
            if slot:
                terms.setdefault(slot, set()).update(t for t in r["rhs"] if t != "#NUM")
    lines = []
    for slot in ["motif_symptom", "template", "face", "duration", "intensity"]:
        if slot == "duration":
            lines.append("- duration: a number = that many seconds")
        elif terms.get(slot):
            lines.append(f"- {slot}: realized as {{{', '.join(sorted(terms[slot]))}}}")
    return "\n".join(lines)


def context_for(rootmsgs: pd.DataFrame, msg: str):
    """The turns preceding a representative occurrence of this message string."""
    hits = rootmsgs[rootmsgs.t == msg]
    if hits.empty:
        return [], "?"
    post = hits[hits.round_number >= NEWCOMER_START]
    hit = post.iloc[0] if len(post) else hits.iloc[0]
    run = rootmsgs[rootmsgs.run_id == hit.run_id].reset_index(drop=True)
    mask = ((run.round_number == hit.round_number) & (run.substage == hit.substage)
            & (run.message_index_in_substage == hit.message_index_in_substage))
    pos = int(run.index[mask][0])
    ctx = [{"agent": AGENT.get(r.message_agent, r.message_agent), "text": str(r.t)}
           for r in run.iloc[max(0, pos - CONTEXT_TURNS):pos].itertuples()]
    return ctx, AGENT.get(hit.message_agent, hit.message_agent)


def annotate_root(cli, system, task, root, items):
    labels_by_string = {}
    filled = task.format(grammar=grammar_summary(root))
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        blocks = []
        for j, (s, ctx, agent) in enumerate(chunk):
            lines = [f"=== ITEM {j} ==="]
            lines.append("conversation context (recent turns):" if ctx
                         else "conversation context: (none available)")
            lines += [f"  [{c['agent']}] {c['text'][:200]}" for c in ctx]
            lines.append(f"TARGET (sent by {agent}): {s!r}")
            blocks.append("\n".join(lines))
        r = cli.chat.completions.create(
            model=MODEL, timeout=300,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": filled + "\n\n" + "\n\n".join(blocks)}])
        txt = r.choices[0].message.content
        labels = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group(0))["labels"]
        if len(labels) != len(chunk):
            raise ValueError(f"{root}: got {len(labels)} labels for {len(chunk)} items")
        print(f"    batch {i // BATCH}: {r.usage.prompt_tokens} in / "
              f"{r.usage.completion_tokens} out tokens")
        labels_by_string.update({s: lab for (s, _, _), lab in zip(chunk, labels)})
    return labels_by_string


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "metaling_annotations_cache.json"

    rl = pd.read_csv(DATA / "learnability_run_level.csv")
    ml = pd.read_csv(DATA / "learnability_message_level_slim.csv")

    swap = rl[rl.phase == "replace_learned"][["run_id", "src_id", "history"]]
    m = ml[ml.run_id.isin(swap.run_id)].copy()
    m["t"] = m.message_text.astype(str).str.strip()
    m["root"] = m.run_id.map(dict(zip(swap.run_id, swap.src_id)))
    m["history"] = m.run_id.map(dict(zip(swap.run_id, swap.history)))

    cand = m[m.t.str.endswith("?") | m.t.str.contains(KW, case=False, regex=True, na=False)]
    cand = cand[(cand.t.str.len() <= MAXLEN) & (cand.t != "#NAME?")]

    system, task = prompts()
    cli = client()
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for root in sorted(m.root.unique()):
        if root in cache:
            continue
        strings = sorted(cand[cand.root == root].t.unique())
        rootmsgs = m[m.root == root].sort_values(SORT).reset_index(drop=True)
        items = [(s, *context_for(rootmsgs, s)) for s in strings]
        cache[root] = annotate_root(cli, system, task, root, items) if items else {}
        print(f"  {root}: {len(strings)} unique strings")
        cache_path.write_text(json.dumps(cache, indent=1))

    # join the labels onto every post-swap candidate occurrence
    inst = cand[cand.round_number >= NEWCOMER_START].copy()
    fields = ["has_question", "target_scope", "target_slot", "compositional", "transparent"]
    for f in fields:
        inst[f] = [(cache.get(r.root, {}).get(r.t) or {}).get(f) for r in inst.itertuples()]
    cols = ["run_id", "root", "history", "phase", "round_number", "message_agent", "t"] + fields
    inst[cols].to_csv(args.out_dir / "metaling_instances.csv", index=False)

    post = m[m.round_number >= NEWCOMER_START]
    den = (post.groupby(["root", "history"])
           .agg(link_msgs=("t", "size"), rounds=("round_number", "nunique"),
                runs=("run_id", "nunique")).reset_index())
    den.to_csv(args.out_dir / "metaling_denominators.csv", index=False)
    print(f"\n{len(inst)} candidates, {int((inst.has_question == True).sum())} questions"
          f" -> {args.out_dir}")


if __name__ == "__main__":
    main()
