from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from core import (
    MicrocircuitNet,
    choose_matched_widths,
    count_params,
    effective_rank,
    evaluate,
    instantiate,
    latency_us,
    load_bundle,
    subnode_diversity,
    train_one,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--model", required=True, choices=["MLP", "Micro"])
    p.add_argument("--topology", default="ring")
    p.add_argument("--steps", default=3, type=int)
    p.add_argument("--target-params", default=12000, type=int)
    p.add_argument("--k", default=5, type=int)
    p.add_argument("--max-epochs", default=40, type=int)
    p.add_argument("--patience", default=6, type=int)
    p.add_argument("--threads", default=1, type=int)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    data = load_bundle(args.dataset)
    h, r, pm, pr = choose_matched_widths(data.input_dim, data.n_classes, args.target_params, args.k)
    model = instantiate(args.model, data, h, r, args.k, args.steps, args.topology, args.seed)
    model, stats = train_one(model, data, args.seed, args.max_epochs, args.patience, 128)
    acc, nll, features = evaluate(model, data.x_test, data.y_test)
    er, er_norm = effective_rank(features)
    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": args.model,
        "topology": args.topology,
        "steps": args.steps,
        "mlp_hidden": h,
        "macro_nodes": r,
        "mlp_params": pm,
        "micro_params": pr,
        "params": count_params(model),
        "accuracy": acc,
        "nll": nll,
        "effective_rank": er,
        "normalized_effective_rank": er_norm,
        "latency_us_per_sample": latency_us(model, data.input_dim, repeats=10),
        **stats,
    }
    row["subnode_diversity"] = (
        subnode_diversity(model, data.x_test) if isinstance(model, MicrocircuitNet) else None
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)
    # Some CPU numerical-library builds delay interpreter shutdown. The result
    # is already fsynced above; a direct exit keeps one-job workers deterministic.
    os._exit(0)


if __name__ == "__main__":
    main()
