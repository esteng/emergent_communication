"""Shared paths and plot styling for the morphology release.

Every path is relative to the release root, so scripts run from any working directory.
Mirrors the conventions of the release's top-level `scripts/common.py`.
"""
from pathlib import Path

import matplotlib
import matplotlib.font_manager as _font_manager

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "morphology"
CACHE = DATA / "cache"
RUNS = DATA / "transcripts"          # verbatim event logs, one dir per root
ENCODE_DECODE = DATA / "encode_decode"
ROBUSTNESS = DATA / "robustness"
JUDGE_VALIDATION = DATA / "judge_validation"
RUN_MANIFEST = DATA / "run_manifest.json"
PROMPTS = DATA / "prompts"                # one .txt per LLM prompt, loaded at call time

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CSV = RESULTS / "csv"

for _d in (FIGURES, TABLES, CSV):
    _d.mkdir(parents=True, exist_ok=True)


# Nimbus Roman is the paper's face; register it wherever the distribution puts it and fall
# back to a generic serif when it is not installed, so figures render anywhere.
for _font_root in (Path("/usr/share/fonts/urw-base35"),
                   Path("/usr/share/fonts/truetype/urw-base35")):
    for _font_file in _font_root.glob("NimbusRoman-*.otf"):
        _font_manager.fontManager.addfont(str(_font_file))
PAPER_FONT = (
    "Nimbus Roman"
    if any(f.name == "Nimbus Roman" for f in _font_manager.fontManager.ttflist)
    else "serif"
)


# --------------------------------------------------------------------------
# Model identity and palette, shared by every figure and table
# --------------------------------------------------------------------------
MODEL_IDS = ["claude-opus-4-7", "claude-sonnet-4-6", "gpt-5.4"]
MODEL_LABEL = {"claude-opus-4-7": "Claude Opus 4.7",
               "claude-sonnet-4-6": "Claude Sonnet 4.6",
               "gpt-5.4": "GPT-5.4"}
MODEL_SHORT = {"claude-opus-4-7": "Opus",
               "claude-sonnet-4-6": "Sonnet",
               "gpt-5.4": "GPT"}
MODEL_COLOR = {"claude-opus-4-7": "#2a78d6",
               "claude-sonnet-4-6": "#008300",
               "gpt-5.4": "#eda100"}


def by_label(mapping: dict) -> dict:
    """Re-key a per-model-id mapping by the model's full display label."""
    return {MODEL_LABEL[model_id]: value for model_id, value in mapping.items()}


def by_short(mapping: dict) -> dict:
    """Re-key a per-model-id mapping by the model's short display label."""
    return {MODEL_SHORT[model_id]: value for model_id, value in mapping.items()}


def use_paper_style() -> None:
    """Serif text, editable PDF fonts, no grid -- matches the figures in the paper."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["axes.grid"] = False


def save(fig, name: str) -> Path:
    """Write a figure into results/figures and report the release-relative path."""
    out = FIGURES / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")
    return out


def load_prompt(name: str) -> str:
    """Return the text of `data/morphology/prompts/<name>.txt`."""
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def load_manifest() -> dict:
    """The induced and tested run ids this release ships data for."""
    import json

    return json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
