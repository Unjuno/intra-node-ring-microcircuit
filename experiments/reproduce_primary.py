from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

DATASETS = ["digits", "breast_cancer", "wine", "synthetic_ring"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
SPECS = [
    ("MLP", "none", 0, "MLP"),
    ("Micro", "none", 3, "MicroNoEdges"),
    ("Micro", "ring", 3, "MicroRing"),
]


def exact_signflip_p(d: np.ndarray) -> float:
    observed = abs(float(d.mean()))
    vals = []
    for mask in range(1 << len(d)):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(len(d))])
        vals.append(abs(float((signs * d).mean())))
    return float(np.mean(np.asarray(vals) >= observed - 1e-15))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/local-primary")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--target-params", type=int, default=12000)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    out = Path(args.outdir)
    jobs = out / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})

    rows = []
    for dataset, seed, (model, topology, steps, label) in itertools.product(args.datasets, args.seeds, SPECS):
        result_path = jobs / f"{dataset}__{seed}__{label}.json"
        if not result_path.exists():
            cmd = [
                sys.executable, str(here / "run_single.py"),
                "--dataset", dataset, "--seed", str(seed), "--model", model,
                "--topology", topology, "--steps", str(steps),
                "--target-params", str(args.target_params),
                "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
                "--threads", "1", "--output", str(result_path),
            ]
            subprocess.run(cmd, env=env, timeout=args.timeout, check=True)
        row = json.loads(result_path.read_text(encoding="utf-8"))
        row["label"] = label
        rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw_primary.csv", index=False)
    summary = raw.groupby(["dataset", "label"]).agg(
        n=("accuracy", "size"), accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        nll_mean=("nll", "mean"), effective_rank_mean=("effective_rank", "mean"),
        normalized_effective_rank_mean=("normalized_effective_rank", "mean"),
        latency_us_mean=("latency_us_per_sample", "mean"),
    ).reset_index()
    summary.to_csv(out / "primary_summary.csv", index=False)

    stats = []
    for dataset in args.datasets:
        pivot = raw[raw.dataset.eq(dataset)].pivot(index="seed", columns="label", values="accuracy")
        for a, b in [("MicroRing", "MLP"), ("MicroNoEdges", "MLP"), ("MicroRing", "MicroNoEdges")]:
            d = (pivot[a] - pivot[b]).dropna().to_numpy()
            mean = float(d.mean())
            sd = float(d.std(ddof=1))
            se = sd / math.sqrt(len(d))
            critical = float(t.ppf(0.975, len(d) - 1))
            stats.append({
                "dataset": dataset, "comparison": f"{a}-{b}", "n": len(d),
                "mean_difference": mean, "ci95_low": mean - critical * se,
                "ci95_high": mean + critical * se,
                "cohen_dz": mean / sd if sd > 0 else None,
                "exact_signflip_p": exact_signflip_p(d),
            })
    pd.DataFrame(stats).to_csv(out / "primary_paired_stats.csv", index=False)


if __name__ == "__main__":
    main()
