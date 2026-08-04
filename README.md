# Intra-Node Ring Microcircuit

An exploratory research repository testing whether ordinary scalar hidden units benefit from internal microstructure. The project began with five-subnode recurrent ring cells, but the evidence now points away from ring topology and toward a simpler learned scalar micro-activation.

> **Status: completed exploratory study.** The tested ring topology did not show a robust advantage. A smaller learned scalar state-update activation improved two tasks under closely matched parameter budgets, but general superiority and novelty are not established.

## Main conclusion

The experiments separated four hypotheses:

1. ring topology;
2. recurrence depth;
3. multiple internal subnodes;
4. a learned nonlinear transformation inside each scalar unit.

The first three were not required for the strongest positive result. A dedicated scalar micro-activation with one learned update was sufficient.

### Ten-seed follow-up at approximately 30,000 trainable parameters

| Dataset | MLP | Fixed `GELU(tanh(z))` | ScalarMicro T=1 | ScalarMicro T=3 |
|---|---:|---:|---:|---:|
| Digits | 96.67% | 97.72% | **98.28%** | 98.31% |
| Synthetic cyclic teacher | 85.89% | 86.17% | **87.22%** | 86.58% |

Paired results for `ScalarMicro T=1`:

- Digits versus MLP: `+1.61 pp`, 95% CI `[+1.19,+2.03]`, exact sign-flip `p=0.001953`, wins `10/10`.
- Synthetic task versus MLP: `+1.33 pp`, 95% CI `[+0.75,+1.90]`, exact sign-flip `p=0.001953`, wins `10/10`.
- Digits versus the fixed composite activation: `+0.56 pp`, 95% CI `[+0.26,+0.85]`.
- Synthetic task versus the fixed composite activation: `+1.05 pp`, 95% CI `[+0.31,+1.79]`.

One update and three updates were indistinguishable on Digits. On the synthetic task, one update was significantly better than three. More recurrence is therefore not the explanation.

## What happened to the ring hypothesis?

The primary four-dataset replication compared:

- `MLP`;
- `MicroNoEdges`, a five-subnode unit without internal edges;
- `MicroRing`, the same unit with ring communication.

Results were heterogeneous. The ring did not show a robust advantage over the no-edge control, and a topology screen did not identify Ring as uniquely superior to Chain, Star, Complete, or Random graphs.

At a larger fixed budget on Digits, ten paired seeds gave:

- `k5_no_edges - MLP`: `+1.22 pp`;
- `k5_ring - k5_no_edges`: `-0.06 pp`, 95% CI spanning zero.

Thus the positive result is not evidence for ring topology.

## Supported statements

- The ring and no-edge micro-unit implementations are trainable and reproducible on CPU.
- Ring-specific superiority was not established.
- Multiple internal subnodes were not necessary for the strongest positive result.
- A learned scalar state-update activation improved Digits and the controlled synthetic task under closely matched trainable-parameter budgets.
- The learned activation is slower than a standard MLP.
- Effects remain task-dependent: Breast Cancer and Wine did not provide confirmatory generalization.

## Unsupported statements

- “Ring neurons outperform MLPs.”
- “Internal microcircuits generally improve neural networks.”
- “The method is compute-efficient, robust, causal, biologically faithful, or world-first.”
- “The positive result establishes novelty over adaptive activation-function research.”

## Reproduce

Primary ring/no-edge experiment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/reproduce_primary.py --outdir results/local-primary
```

Scalar micro-activation follow-up:

```bash
python experiments/reproduce_scalar_followup.py \
  --outdir results/local-scalar-followup
```

## Repository map

```text
experiments/core.py                         shared datasets, models, and metrics
experiments/reproduce_primary.py            primary four-dataset replication
experiments/run_scalar_control.py            scalar activation worker
experiments/reproduce_scalar_followup.py     ten-seed mechanism follow-up
docs/EVIDENCE_2026-08-03.md                 primary evidence report
docs/FOLLOWUP_2026-08-03.md                 mechanism isolation and stopping report
docs/CLAIMS.md                              public claim boundary
results/2026-08-03/                         aggregate statistics
```

## Research status

The original ring question has reached a stopping point. The scientifically justified next project, if pursued, is comparison of the scalar micro-activation against modern adaptive and learnable activation functions. It is not further tuning of ring topologies.

## License

Code is licensed under Apache-2.0. Third-party datasets retain their original terms.
