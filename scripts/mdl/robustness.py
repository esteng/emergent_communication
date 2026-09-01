"""Robustness control — is the DL(D|G) result an artifact of the smoothing constant?

Re-runs the component computation and the root-level correlations across the add-k
Laplace constant used for rule-expansion probabilities, k in {0.1, 0.25, 0.5, 1.0, 2.0},
and reports the largest resulting change in Pearson r.

  python scripts/mdl/robustness.py
    -> results/tables/mdl_robustness_smoothing.csv
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import DATA, TABLES  # noqa: E402
from stats_and_partials import correlations, root_level  # noqa: E402

K_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0]
KEY_STATS = ["DL(G) ~ success", "DL(D|G) ~ success", "partial DL(D|G)|cov ~ success"]


def components_at(tmp: Path, k: float) -> pd.DataFrame:
    out = tmp / f"k{k}.csv"
    subprocess.run([sys.executable, str(HERE / "compute_components.py"),
                    "--k", str(k), "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(out)


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        print(f"smoothing sweep, k in {K_VALUES}:")
        for k in K_VALUES:
            r = correlations(root_level(components_at(Path(td), k))).set_index("stat").pearson_r
            rows.append({"k": k, **{s: round(r[s], 4) for s in KEY_STATS}})
            print("  k=%-5s " % k + "  ".join(f"{s.split(' ~')[0]}={r[s]:+.4f}" for s in KEY_STATS))

    df = pd.DataFrame(rows)
    out = TABLES / "mdl_robustness_smoothing.csv"
    df.to_csv(out, index=False)
    spread = {s: round(float(df[s].max() - df[s].min()), 4) for s in KEY_STATS}
    print(f"\nmax change in Pearson r across k: {spread}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
