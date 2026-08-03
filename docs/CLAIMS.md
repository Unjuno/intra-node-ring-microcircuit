# Claim boundary

## Supported factual statements

1. A fully connected macro-layer can be built from units containing recurrent scalar subnodes.
2. The primary 2026-08-03 replication completed 10 paired seeds on four datasets under exact or near-exact trainable FP32 parameter matching.
3. The primary replication did not establish a ring-specific advantage. Results were heterogeneous across datasets, and Ring was not robustly better than the same five-subnode unit with internal edges disabled.
4. A later fixed-budget Digits experiment found that no-edge micro-units with `k=1`, `k=2`, and `k=5` outperformed an approximately 30,000-parameter MLP across ten paired seeds.
5. Because `k=1` has no multi-state topology, the strongest positive Digits result does not require multiple subnodes or graph message passing.
6. A dedicated scalar micro-activation with no dormant graph parameters improved Digits by `+1.61 pp` and the synthetic cyclic-interaction task by `+1.33 pp` versus parameter-matched MLPs across ten paired seeds. Both comparisons won 10/10 seeds and had exact sign-flip `p=0.001953`.
7. A fixed `GELU(tanh(z))` activation explained part, but not all, of the improvement. A four-hidden-layer GELU MLP did not reliably reproduce the Digits gain.
8. One learned scalar update was sufficient. Three updates did not improve Digits and were worse than one update on the synthetic task.
9. All tested micro variants were slower than the baseline MLP in the measured CPU implementation.

## Supported interpretation with qualification

- The original Ring hypothesis was not supported.
- The evidence instead supports a task-dependent benefit from replacing a fixed pointwise activation with a small learned scalar state-update function.
- The current positive result is closer to adaptive or learnable activation-function research than to graph-topology research.
- The result is replicated on Digits and one controlled synthetic task, not established as a general-purpose improvement.

## Not supported

- “The ring cell generally improves MLP performance.”
- “Ring topology caused the positive results.”
- “Multiple internal states are necessary.”
- “The representation space always expands.”
- “More recurrent steps are better.”
- “The method is more compute-efficient or faster.”
- “The model is robust to noise or failures.”
- “The architecture enables causal interpretation or anomaly localization.”
- “The architecture is a faithful brain model.”
- “This is the first learnable or adaptive activation function.”

## Required wording for public summaries

Use:

> We began by testing ring-shaped recurrent microcircuits inside fully connected units. Ring-specific superiority was not established. Mechanism controls instead isolated a simpler learned scalar state-update activation that improved Digits and a controlled synthetic task under closely matched parameter budgets, at higher CPU latency. Generality and novelty relative to adaptive activation functions remain open.

Do not use:

> A new ring neuron that outperforms MLPs.
