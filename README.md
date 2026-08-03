# Intra-Node Ring Microcircuit

A research prototype that replaces each scalar hidden unit in a fully connected network with a small recurrent microcircuit. Each macro-unit receives the ordinary fully connected input, distributes it over five internal scalar subnodes, performs shared recurrent updates, and reads the internal state back to one scalar macro-output.

> **Status: exploratory and falsifiable.** The current evidence does **not** establish general superiority over MLPs, ring-specific superiority, computational efficiency, or world-first novelty.

## Current 10-seed evidence

The confirmatory replication matches trainable FP32 parameter counts for each dataset and compares:

- `MLP`: two-hidden-layer fully connected baseline;
- `MicroNoEdges`: the same five-subnode macro-unit without internal edges;
- `MicroRing`: the five-subnode macro-unit with a fixed ring and three shared recurrent steps.

| Dataset | MLP | MicroNoEdges | MicroRing | MicroRing − MLP, 95% CI |
|---|---:|---:|---:|---:|
| Breast cancer | 95.79% | 98.07% | 97.98% | +2.19 pp `[+1.40,+2.99]` |
| Digits | 96.58% | 96.06% | 96.47% | −0.11 pp `[−0.90,+0.67]` |
| Synthetic ring teacher | 85.72% | 84.50% | 84.51% | −1.21 pp `[−1.94,−0.48]` |
| Wine | 96.94% | 90.56% | 90.28% | −6.67 pp `[−9.51,−3.83]` |

The breast-cancer gain is evidence for the **micro-unit parameterization**, not for the ring: `MicroRing − MicroNoEdges = −0.09 pp`, with a confidence interval spanning zero. On Digits, the ring improves over the no-edge micro-unit by `+0.42 pp`, but this ring-specific effect does not survive correction across datasets.

## What is supported

- The architecture is implementable and trainable on CPU.
- Equal trainable-parameter budgets can be enforced exactly or nearly exactly.
- Internal recurrence can alter results relative to the same micro-unit with edges disabled.
- Effects are strongly task-dependent.

## What is not supported

- General accuracy superiority over ordinary MLPs.
- A ring-specific advantage over other internal topologies.
- Increased effective representation dimension in every task.
- Better robustness, latency, or FLOP efficiency.
- Causal interpretability or anomaly localization; those remain testable hypotheses.

## Reproduce the primary experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/reproduce_primary.py --outdir results/local-primary
```

The coordinator launches each training run in a separate process because some CPU numerical-library builds delay interpreter shutdown after repeated PyTorch/scikit-learn jobs.

## Repository map

```text
experiments/                 exact CPU experiment implementation
results/2026-08-03/          complete 10-seed aggregate results
docs/CLAIMS.md               permitted and prohibited claims
docs/EVIDENCE_2026-08-03.md  statistical interpretation
docs/EXPERIMENT_PLAN.md      next confirmatory tests
docs/IMPACT.md               what would constitute meaningful impact
```

## License

Code is licensed under Apache-2.0. Original documentation and figures may be reused under the repository license unless a file states otherwise. Third-party datasets retain their original terms.
