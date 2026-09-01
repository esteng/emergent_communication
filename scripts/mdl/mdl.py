#!/usr/bin/env python3
"""Two-part description length of an emergent protocol: DL(D, G) = DL(G) + DL(D|G).

Every cost is a Shannon code length under a distribution estimated from the data or
fixed by the SaveVeyru scenario — there are no hand-tuned constants.

  char code p(c)   the protocol's own message characters, add-1 smoothed. Used to spell
                   out invented code-words, and as the fallback for messages the grammar
                   cannot parse.
  referent id      log2(R) to name which of the R fixed environment referents a lexical
                   entry denotes. R = 37 for the joint grammar (14 motifs + 14 templates
                   + 6 faces + 3 intensities).
  vocabulary |V|   the distinct symbols of THIS grammar; a structural rule of m symbols
                   costs m*log2|V|.

  DL(G)   = sum_lex [ char_cost(word) + log2 R ]     codebook
          + sum_struct |rule| * log2|V|              combinators
          + log2(#lex+1) + log2(#struct+1)           how many rules of each kind

There are two data codes here, and they are NOT interchangeable:

  dl_data_given_grammar        per-slot MLE over the meanings. Nearly independent of the
                               grammar's structure, so it is essentially coverage in
                               disguise. Kept only because it feeds the `MDL` column of
                               the baseline table.
  dl_data_given_grammar_pcfg   the DERIVATIONAL code, and the one the paper reports.
                               Rule-expansion probabilities are estimated by MLE on a
                               train window (no structural re-induction), then held-out
                               messages are charged -log2 P(derivation). Unparsed
                               messages fall back to the character code, so a language
                               whose speakers revert to English pays heavily.

Refs: Rissanen 1978; Grunwald 2007; Stolcke & Omohundro 1994; de Marcken 1996.
"""

import math, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import scfg

# defaults = engineer side; observer/joint pass their own via compute()
REF_INV = 14 + 6 + 3            # fixed referent inventory (templates+faces+intensities)
AXES_LEX = ("template", "face", "intensity")
SLOTS = ("template", "face", "intensity", "duration")
# inventory sizes per axis (fixed scenario ontology)
AXIS_SIZE = {"motif": 14, "template": 14, "face": 6, "intensity": 3}


def _char_cost(messages):
    """Shannon code length of a string under the corpus's char distribution (add-1)."""
    cnt = Counter(c for m in messages if m for c in m)
    tot, vocab = sum(cnt.values()), len(cnt)
    denom = tot + vocab + 1
    def cost(s):
        return sum(-math.log2((cnt[c] + 1) / denom) for c in s)
    return cost


def _lexicon(spec, axes_lex):
    lex, struct = [], []
    for r in spec["rules"]:
        mp = r.get("map")
        if isinstance(mp, dict) and len(mp) == 1 and list(mp)[0] in axes_lex:
            lex.append(("".join(r["rhs"]), list(mp)[0], list(mp.values())[0]))
        elif not isinstance(mp, dict):
            struct.append(r)
    return lex, struct


def dl_grammar(spec, ccost, axes_lex, ref_inv):
    lex, struct = _lexicon(spec, axes_lex)
    V = set()
    for r in spec["rules"]:
        V.add(r["lhs"]); V.update(r["rhs"])
        V.add(str(r.get("map")))
    vbits = math.log2(len(V)) if len(V) > 1 else 0.0
    L_lex = sum(ccost(tok) + math.log2(ref_inv) for tok, _, _ in lex)
    L_struct = sum((1 + len(r["rhs"]) + 1) * vbits for r in struct)   # LHS + RHS + map
    L_count = math.log2(len(lex) + 1) + math.log2(len(struct) + 1)
    return L_lex + L_struct + L_count, {"n_lex": len(lex), "n_struct": len(struct)}


def _proc(grammar, msg):
    m = scfg.parse(grammar, msg)
    return m[0] if m else None


def dl_data_given_grammar(grammar, messages, ccost, slots):
    procs = [(_proc(grammar, m), m) for m in messages if m]
    covered = [(p, m) for p, m in procs if p]
    counts = {ax: Counter() for ax in slots}
    for p, _ in covered:
        for ax in slots:
            if ax in p:
                counts[ax][p[ax]] += 1
    tot = {ax: sum(counts[ax].values()) for ax in slots}

    def code_bits(p):
        b = 0.0
        for ax in slots:
            if ax in p and tot[ax]:
                b += -math.log2(counts[ax][p[ax]] / tot[ax])   # MLE incl. duration
        return b

    bits = sum(code_bits(p) if p else ccost(m) for p, m in procs)
    return bits, {"n_msgs": len(procs), "n_covered": len(covered),
                  "coverage": len(covered) / len(procs) if procs else 0.0}


# ── PCFG data code (option 1): DL(D|G) = -log2 P(derivation) ─────────────────────
# Turns an ALREADY-INDUCED grammar into a PCFG by counting rule expansions on a
# TRAIN window of messages (no structural re-induction), then scores -log2 P on a
# held-out TEST window. The grammar's structure now drives the data cost: messages
# whose derivations use higher-probability rules encode more cheaply. Numeric leaves
# (#NUM / duration) are not a rule choice, so they keep an empirical value code.

def rule_key(r):
    import json as _json
    return (r["lhs"], tuple(r["rhs"]), _json.dumps(r.get("map"), sort_keys=True))


def parse_rules(g, msg):
    """Like scfg.parse, but returns (rule_keys, meaning) for the first full
    derivation (mirrors scfg's first-meaning-per-(sym,i) memoization), else None."""
    toks = scfg.tokenize(msg)
    if not toks:
        return None
    n = len(toks); memo = {}; inprog = set()

    def expand(rhs, k, i, acc, racc):
        if k == len(rhs):
            yield acc, i, racc; return
        for m, j, rl in P(rhs[k], i):
            yield from expand(rhs, k + 1, j, acc + [m], racc + rl)

    def P(sym, i):
        if (sym, i) in memo:
            return memo[(sym, i)]
        if (sym, i) in inprog:
            return []
        inprog.add((sym, i))
        res = {}
        if sym not in g.nt:
            if i < n:
                if sym == "#NUM" and toks[i].isdigit():
                    res[i + 1] = (toks[i], [])
                elif toks[i] == sym:
                    res[i + 1] = (sym, [])
        else:
            for rule in g.by_lhs[sym]:
                rk = rule_key(rule)
                for acc, j, racc in expand(rule["rhs"], 0, i, [], []):
                    if j not in res:
                        res[j] = (scfg._assemble(rule.get("map"), acc), [rk] + racc)
        inprog.discard((sym, i))
        out = [(mn, j, rl) for j, (mn, rl) in res.items()]
        memo[(sym, i)] = out
        return out

    for mn, j, rl in P(g.start, 0):
        if j == n:
            return rl, (mn if isinstance(mn, list) else [mn])
    return None


def fit_pcfg(grammar, train_messages, num_slot="duration", k=0.5):
    """Estimate per-LHS rule probabilities (add-k) + an empirical numeric-leaf code
    from TRAIN messages parsed under the fixed grammar. Returns a scorer object."""
    rule_cnt = {}
    num_cnt = Counter()
    seen = set()
    for m in train_messages:
        pr = parse_rules(grammar, m)
        if not pr:
            continue
        rks, meaning = pr
        for rk in rks:
            rule_cnt.setdefault(rk[0], Counter())[rk] += 1
            seen.add(rk)
        for proc in meaning:
            if num_slot in proc:
                num_cnt[proc[num_slot]] += 1
    rules_per_lhs = {lhs: [rule_key(r) for r in rs] for lhs, rs in grammar.by_lhs.items()}

    def rule_bits(rk):
        lhs = rk[0]; alts = rules_per_lhs[lhs]
        if len(alts) <= 1:
            return 0.0                              # deterministic, no choice to encode
        tot = sum(rule_cnt.get(lhs, {}).values())
        p = (rule_cnt.get(lhs, {}).get(rk, 0) + k) / (tot + k * len(alts))
        return -math.log2(p)

    def num_bits(v):
        tot = sum(num_cnt.values()); V = len(num_cnt) + 1     # +1 OOV
        p = (num_cnt[v] + k) / (tot + k * V) if tot else 1.0 / V
        return -math.log2(p)

    return {"rule_bits": rule_bits, "num_bits": num_bits, "seen": seen, "num_slot": num_slot}


def dl_data_given_grammar_pcfg(grammar, train_messages, test_messages, ccost,
                               num_slot="duration", k=0.5):
    """PCFG DL(D|G) on TEST: -log2 P(derivation) for covered messages (+ numeric-leaf
    code), literal char fallback for un-parseable ones. ccost = char-model fallback."""
    pf = fit_pcfg(grammar, train_messages, num_slot, k)
    n = cov = 0
    bits = cov_bits = 0.0
    used = unseen = 0
    for m in test_messages:
        if not m:
            continue
        n += 1
        pr = parse_rules(grammar, m)
        if not pr:
            bits += ccost(m); continue              # identical fallback to slot-MLE scheme
        cov += 1
        rks, meaning = pr
        b = sum(pf["rule_bits"](rk) for rk in rks)
        used += len(rks); unseen += sum(1 for rk in rks if rk not in pf["seen"])
        for proc in meaning:
            if num_slot in proc:
                b += pf["num_bits"](proc[num_slot])
        bits += b; cov_bits += b
    return bits, {"n_msgs": n, "n_covered": cov,
                  "coverage": cov / n if n else 0.0,
                  "DL_DgG_pcfg_cov": cov_bits,
                  "frac_rules_unseen": unseen / used if used else None}


def compute(grammar_spec, messages, axes_lex=AXES_LEX, slots=SLOTS, ref_inv=REF_INV):
    ccost = _char_cost(messages)               # char code from THIS protocol's messages
    g = scfg.Grammar(grammar_spec)
    dlg, gi = dl_grammar(grammar_spec, ccost, axes_lex, ref_inv)
    dld, di = dl_data_given_grammar(g, messages, ccost, slots)
    return {
        "DL_G": round(dlg, 1), "DL_D_given_G": round(dld, 1), "MDL": round(dlg + dld, 1),
        "MDL_per_msg": round((dlg + dld) / di["n_msgs"], 2) if di["n_msgs"] else None,
        **gi, **di,
    }
