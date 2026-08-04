# What would be impactful?

## Level 1 — reproducibility impact

A stable open implementation with complete negative and positive results prevents repeated rediscovery and provides a clean benchmark for structured neural units.

## Level 2 — architectural impact

If Ring consistently exceeds no-edge, chain, star, complete, and random controls under parameter and compute matching, it would demonstrate a topology-specific inductive bias inside individual neural units.

## Level 3 — diagnostic impact

If subnode/edge interventions enable reliable anomaly localization, the architecture could provide inspectable failure sites unavailable in ordinary scalar units. Relevant applications include fault diagnosis, sensor networks, safety monitoring, and mechanistic analysis.

## Level 4 — systems impact

If recurrent microcircuits can be fused or parallelized so that measured latency approaches MLP latency, the structure may become useful for edge inference or specialized hardware. The current CPU implementation is far from this threshold.

## Level 5 — theoretical impact

A theorem relating topology, shared recurrent depth, stability, and distinguishable function classes under a fixed parameter budget would elevate the work from an empirical cell design to a general principle of structured neural units.

## Non-impactful outcomes

- A gain on one small dataset without topology controls.
- A gain caused solely by more FLOPs or activation memory.
- Higher coordinate dimension without task-relevant information.
- Biological analogies without predictive or diagnostic consequences.
