# Claim boundary

## Supported factual statements

1. A fully connected macro-layer can be built from units that each contain five recurrent scalar subnodes.
2. The implementation can share internal parameters across recurrent steps, increasing computation depth without increasing trainable parameter count per added step.
3. In the 2026-08-03 replication, 10 paired seeds were completed for four datasets under matched trainable FP32 parameter counts.
4. `MicroRing` significantly outperformed the MLP on breast cancer, was statistically indistinguishable on Digits, and underperformed on Wine and the synthetic ring-teacher task.
5. `MicroRing` was materially slower than MLP on the tested CPU implementation.

## Supported interpretation with qualification

- The internal micro-unit parameterization can be useful on some tasks.
- Ring recurrence can contribute beyond merely adding internal subnodes on some tasks, but ring-specific evidence is not yet robust across datasets.
- The architecture offers explicit internal intervention sites that ordinary scalar units do not expose; usefulness for anomaly localization remains untested.

## Not supported

- “The ring cell generally improves MLP performance.”
- “The ring topology is the cause of the breast-cancer improvement.”
- “The representation space always expands.”
- “The method is more compute-efficient or faster.”
- “The model is robust to noise or failures.”
- “The architecture is a faithful brain model.”
- “This is the first neural unit with an internal microcircuit.”

## Required wording for public summaries

Use:

> We investigate a specific ring-shaped recurrent microcircuit as a replacement for scalar hidden units. Ten-seed, parameter-matched results are heterogeneous: one dataset improves, one is neutral, and two degrade. Ring-specific superiority remains unestablished.

Do not use:

> A new ring neuron that outperforms MLPs.
