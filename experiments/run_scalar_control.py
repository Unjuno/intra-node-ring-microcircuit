from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

from core import (
    MLP,
    count_params,
    effective_rank,
    evaluate,
    latency_us,
    load_bundle,
    set_seed,
    train_one,
)


class CompositeMLP(nn.Module):
    """Two-layer MLP with the fixed composite activation GELU(tanh(z))."""

    def __init__(self, input_dim: int, hidden: int, classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, classes)

    @staticmethod
    def activation(z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(torch.tanh(z))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        logits = self.out(h)
        return (logits, h) if return_features else logits


class Deep4MLP(nn.Module):
    """Four-hidden-layer GELU MLP used as a depth control."""

    def __init__(self, input_dim: int, hidden: int, classes: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden)]
            + [nn.Linear(hidden, hidden) for _ in range(3)]
        )
        self.out = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = x
        for layer in self.layers:
            h = torch.nn.functional.gelu(layer(h))
        logits = self.out(h)
        return (logits, h) if return_features else logits


class ScalarMicroLayer(nn.Module):
    """Learned scalar state-update activation with no graph or dormant weights."""

    def __init__(self, input_dim: int, width: int, steps: int):
        super().__init__()
        self.steps = steps
        self.input_proj = nn.Linear(input_dim, width)
        self.raw_self = nn.Parameter(torch.empty(width))
        self.raw_drive = nn.Parameter(torch.ones(width))
        self.output_bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.raw_self, 0.15, 0.08)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drive = torch.tanh(self.input_proj(x))
        state = drive
        self_gain = 0.85 * torch.tanh(self.raw_self)
        drive_gain = 1.25 * torch.sigmoid(self.raw_drive)
        for _ in range(self.steps):
            state = torch.tanh(self_gain * state + drive_gain * drive)
        return state + self.output_bias


class ScalarMicroNet(nn.Module):
    def __init__(self, input_dim: int, width: int, classes: int, steps: int):
        super().__init__()
        self.l1 = ScalarMicroLayer(input_dim, width, steps)
        self.l2 = ScalarMicroLayer(width, width, steps)
        self.out = nn.Linear(width, classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = torch.nn.functional.gelu(self.l1(x))
        h = torch.nn.functional.gelu(self.l2(h))
        logits = self.out(h)
        return (logits, h) if return_features else logits


def mlp_params(input_dim: int, hidden: int, classes: int) -> int:
    return hidden * hidden + hidden * (input_dim + classes + 2) + classes


def deep4_params(input_dim: int, hidden: int, classes: int) -> int:
    return 3 * hidden * hidden + hidden * (input_dim + classes + 4) + classes


def scalar_micro_params(input_dim: int, width: int, classes: int) -> int:
    return width * width + width * (input_dim + classes + 8) + classes


def closest_width(fn, target: int) -> int:
    return min(range(1, 512), key=lambda width: abs(fn(width) - target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["digits", "synthetic_ring", "wine", "breast_cancer"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--condition",
        choices=["MLP", "Composite", "Deep4", "ScalarMicroT1", "ScalarMicroT3"],
        required=True,
    )
    parser.add_argument("--target-params", type=int, default=30_000)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data = load_bundle(args.dataset)
    set_seed(args.seed)

    if args.condition in {"MLP", "Composite"}:
        hidden = closest_width(
            lambda width: mlp_params(data.input_dim, width, data.n_classes),
            args.target_params,
        )
        model = (
            MLP(data.input_dim, hidden, data.n_classes)
            if args.condition == "MLP"
            else CompositeMLP(data.input_dim, hidden, data.n_classes)
        )
    elif args.condition == "Deep4":
        hidden = closest_width(
            lambda width: deep4_params(data.input_dim, width, data.n_classes),
            args.target_params,
        )
        model = Deep4MLP(data.input_dim, hidden, data.n_classes)
    else:
        steps = 1 if args.condition.endswith("T1") else 3
        hidden = closest_width(
            lambda width: scalar_micro_params(data.input_dim, width, data.n_classes),
            args.target_params,
        )
        model = ScalarMicroNet(data.input_dim, hidden, data.n_classes, steps)

    model, training = train_one(
        model,
        data,
        args.seed,
        args.max_epochs,
        args.patience,
        128,
    )
    accuracy, nll, features = evaluate(model, data.x_test, data.y_test)
    effective, normalized_effective = effective_rank(features)
    row = {
        "dataset": args.dataset,
        "seed": args.seed,
        "condition": args.condition,
        "hidden": hidden,
        "target_params": args.target_params,
        "params": count_params(model),
        "budget_error_pct": 100 * (count_params(model) - args.target_params) / args.target_params,
        "accuracy": accuracy,
        "nll": nll,
        "effective_rank": effective,
        "normalized_effective_rank": normalized_effective,
        "latency_us_per_sample": latency_us(model, data.input_dim, repeats=5),
        **training,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
