# Next confirmatory experiment plan

## Goal

Determine whether any benefit is caused by the ring topology itself rather than by extra internal state, recurrence depth, optimization noise, or additional FLOPs.

## H — falsifiable hypotheses

### H1: topology specificity

At matched trainable parameters and matched recurrent steps, Ring exceeds Chain, Star, Complete, and degree-matched Random topologies by at least 0.5 percentage points on a preregistered task set.

### H2: compute-matched utility

At matched measured CPU latency or matched multiply-accumulate budget, Ring is non-inferior to MLP within a margin of 0.25 percentage points.

### H3: intervention localization

When one internal subnode or edge is corrupted, a diagnostic head identifies the corrupted location at least 10 percentage points more accurately than a parameter-matched MLP diagnostic baseline.

## T — minimum test

- 10 datasets: at least 5 tabular, 3 image-derived fixed-vector tasks, and 2 controlled dynamical tasks.
- 20 paired seeds after the architecture and hyperparameter search is frozen.
- Topologies: None, Ring, Chain, Star, Complete, degree-matched Random.
- Shared-step sweep: `T ∈ {0,1,2,3,5,8}` conducted only on a development split; one T selected before final testing.
- Controls: parameter-matched MLP, FLOP-matched MLP, recurrent MLP without micro-nodes.
- Metrics: accuracy/AUROC, NLL, calibration error, effective rank, subnode diversity, latency, peak memory, and intervention localization.
- Multiple testing: Holm correction within each declared hypothesis family.

## D — decision rules

- **PASS H1:** lower simultaneous 95% CI for Ring minus every topology control exceeds +0.5 pp on the preregistered aggregate.
- **PASS H2:** Ring meets the non-inferiority margin under compute matching and improves at least one secondary efficiency metric.
- **PASS H3:** lower 95% CI for localization improvement exceeds +10 pp.
- **FAIL:** upper CI is at or below zero for H1/H3, or Ring violates compute non-inferiority for H2.
- **UNCERTAIN:** all other outcomes.

## C — principal alternative explanations

1. Internal state count, not topology, drives performance.
2. Repeated nonlinear computation, not the ring, drives performance.
3. Larger activation memory acts as hidden capacity despite parameter matching.
4. Hyperparameter selection favors the proposed model.
5. Gains are dataset-specific and vanish under compute matching.
6. Intervention localization is trivial because the corruption leaks directly into observables.

## U — uncertainty control

- Freeze preprocessing, split seeds, optimizer, stopping rule, and model-selection protocol before the final run.
- Publish all failed runs and raw seed-level results.
- Report both static parameter bits and dynamic activation-memory bits.
- Record CPU model, thread count, PyTorch version, batch size, and timing repetitions.
