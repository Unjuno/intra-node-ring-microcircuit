from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
DATASETS = ["digits", "synthetic_ring"]
CONDITIONS = ["MLP", "Composite", "ScalarMicroT1", "ScalarMicroT3"]


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(values.mean())
    count = 0
    for signs in itertools.product([-1, 1], repeat=len(values)):
        permuted = values * np.asarray(signs)
        if abs(permuted.mean()) >= observed - 1e-15:
            count += 1
    return count / (2 ** len(values))


def paired_row(frame: pd.DataFrame, dataset: str, left: str, right: str) -> dict:
    pivot = frame[frame.dataset == dataset].pivot(
        index="seed", columns="condition", values="accuracy"
    )
    differences = (pivot[left] - pivot[right]).dropna().to_numpy()
    n = len(differences)
    mean = differences.mean()
    std = differences.std(ddof=1)
    error = std / math.sqrt(n)
    critical = stats.t.ppf(0.975, n - 1)
    return {
        "dataset": dataset,
        "contrast": f"{left} - {right}",
        "n": n,
        "mean_pp": 100 * mean,
        "ci_low_pp": 100 * (mean - critical * error),
        "ci_high_pp": 100 * (mean + critical * error),
        "cohen_dz": mean / std if std else float("inf"),
        "exact_p": exact_sign_flip(differences),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/local-scalar-followup")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    worker = Path(__file__).with_name("run_scalar_control.py")
    for dataset in DATASETS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                output = rawdir / f"{dataset}__{seed}__{condition}.json"
                if output.exists():
                    continue
                subprocess.run(
                    [
                        args.python,
                        str(worker),
                        "--dataset",
                        dataset,
                        "--seed",
                        str(seed),
                        "--condition",
                        condition,
                        "--output",
                        str(output),
                    ],
                    check=True,
                )

    rows = [json.loads(path.read_text()) for path in sorted(rawdir.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "raw.csv", index=False)
    summary = frame.groupby(["dataset", "condition"]).agg(
        n=("accuracy", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        nll_mean=("nll", "mean"),
        normalized_effective_rank_mean=("normalized_effective_rank", "mean"),
        latency_us_mean=("latency_us_per_sample", "mean"),
        params_mean=("params", "mean"),
    ).reset_index()
    summary.to_csv(outdir / "summary.csv", index=False)

    contrasts = []
    for dataset in DATASETS:
        for left, right in [
            ("ScalarMicroT1", "MLP"),
            ("ScalarMicroT3", "MLP"),
            ("ScalarMicroT1", "ScalarMicroT3"),
            ("ScalarMicroT1", "Composite"),
        ]:
            contrasts.append(paired_row(frame, dataset, left, right))
    pd.DataFrame(contrasts).to_csv(outdir / "paired_stats.csv", index=False)


if __name__ == "__main__":
    main()
