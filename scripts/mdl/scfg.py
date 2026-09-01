#!/usr/bin/env python3
"""
scfg.py — parse a coded agent message into the meaning it denotes, using a small
synchronous context-free grammar. Pure computation, no API.

A grammar is JSON:
  {"start": "S",
   "rules": [
     {"lhs": "S",    "rhs": ["PROC"],            "map": "seq"},
     {"lhs": "S",    "rhs": ["PROC","+","S"],    "map": "seq"},   # cascade
     {"lhs": "PROC", "rhs": ["TMPL","FACE","ID"],"map": "merge"},
     {"lhs": "TMPL", "rhs": ["A"],               "map": {"template": "TONE_ALL"}},
     {"lhs": "FACE", "rhs": ["Lf"],              "map": {"face": "left"}},
     {"lhs": "ID",   "rhs": ["INT","DUR"],       "map": "merge"},
     {"lhs": "INT",  "rhs": ["g"],               "map": {"intensity": "gentle"}},
     {"lhs": "DUR",  "rhs": ["#NUM"],            "map": "duration"}
   ]}

Symbols that never appear as an lhs are terminals; they match a token literally,
except the special terminal "#NUM" which matches any all-digit token.

map values:
  "merge"    — merge child dicts into one procedure {template,face,intensity,duration}
  "seq"      — build an ordered list of procedure dicts (for cascades)
  "duration" — read the matched number into {"duration": <int>}
  {...}      — a literal partial-meaning dict (leaf)
  null       — contributes nothing (separators)

A message parses only if the WHOLE token sequence is generated (strict coverage).
Result of a parse is a list of procedure dicts.
"""

import re

SLOTS = ["template", "face", "intensity", "duration"]

# token = whitespace/punct split, then split letter|digit runs so "g12" -> "g","12",
# keeping multi-letter codes ("Lf") and separators ("+") intact.
_PUNCT = re.compile(r"[\s;,/\[\]()]+")
_LD = re.compile(r"[A-Za-z]+|\d+|[+|=]")


def tokenize(msg):
    if not msg:
        return []
    toks = []
    for chunk in _PUNCT.split(msg.strip()):
        toks.extend(_LD.findall(chunk))
    return toks


class Grammar:
    def __init__(self, spec):
        self.start = spec["start"]
        self.rules = spec["rules"]
        self.nt = {r["lhs"] for r in self.rules}
        self.by_lhs = {}
        for r in self.rules:
            self.by_lhs.setdefault(r["lhs"], []).append(r)

    @staticmethod
    def load(path):
        return Grammar(json.loads(Path(path).read_text()))


def _assemble(mp, kids):
    if isinstance(mp, dict):
        return dict(mp)
    if mp == "merge":
        d = {}
        for c in kids:
            if isinstance(c, dict):
                d.update(c)
        return d
    if mp == "seq":
        out = []
        for c in kids:
            if isinstance(c, dict):
                out.append(c)
            elif isinstance(c, list):
                out.extend(c)
        return out
    if mp == "duration":
        for c in kids:
            if isinstance(c, str) and c.isdigit():
                return {"duration": int(c)}
        return {}
    return None


def parse(g, msg):
    """Return the list-of-procedures meaning for a full parse, else None.

    Memoized chart parse: P(symbol, i) is computed once and returns one meaning per
    reachable end-position j. This makes ambiguous / recursive (optional-modifier)
    grammars parse in polynomial time instead of exploding the way naive backtracking
    did. A simple in-progress guard breaks left-recursion / epsilon cycles."""
    toks = tokenize(msg)
    if not toks:
        return None
    n = len(toks)
    memo = {}
    inprog = set()

    def expand(rhs, k, i, acc):
        if k == len(rhs):
            yield acc, i
            return
        for m, j in P(rhs[k], i):
            yield from expand(rhs, k + 1, j, acc + [m])

    def P(sym, i):
        if (sym, i) in memo:
            return memo[(sym, i)]
        if (sym, i) in inprog:
            return []                          # cycle guard (left recursion)
        inprog.add((sym, i))
        res = {}                               # end-position j -> first meaning
        if sym not in g.nt:                    # terminal
            if i < n:
                if sym == "#NUM" and toks[i].isdigit():
                    res[i + 1] = toks[i]
                elif toks[i] == sym:
                    res[i + 1] = sym
        else:
            for rule in g.by_lhs[sym]:
                for acc, j in expand(rule["rhs"], 0, i, []):
                    if j not in res:
                        res[j] = _assemble(rule.get("map"), acc)
        inprog.discard((sym, i))
        out = [(meaning, j) for j, meaning in res.items()]
        memo[(sym, i)] = out
        return out

    for meaning, j in P(g.start, 0):
        if j == n:
            return meaning if isinstance(meaning, list) else [meaning]
    return None
