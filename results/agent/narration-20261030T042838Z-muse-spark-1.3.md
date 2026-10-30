The main result is a slow bias. This means the model favors slow, long-range patterns over fast ones. We add a small wave after the first layer and track how strongly later heads follow it across periods. The sealed test gave C1: T with lowpass tilt. Response at the shortest period was only 35 percent [F12] of the maximum. The shape repeated across book sets.

The second point is built-in knobs cannot fix it. This means the bias resists simple tuning. Gain and span tests gave C2: K0 and head scaling gave C3: K0. Both moved tilt in a consistent direction but by too little. Window and position tests gave C4: D0 and C4: D0. The window shift was -0.02324 [F76] against 0.03428 [F77] needed for a dial. Its worst perplexity ratio was 1.472 [F78]. Text controls gave C4: A.

The third point is a trained dial almost works. This means control is possible with small cost. Training gave C5: L0 because it missed its bar narrowly. It reached -24 percent [F92]. Worst perplexity ratio stayed at 1.008 [F97]. Behavior gave C6: B0. Topic switch loss rose only 1.77 percent [F104]. The effect is real but small.

The fourth point is the dial lives in layer 2. This means a small site drives most of the change. Localization gave C7: M1. Layer 2 q, k and v alone carried 80 percent [F113]. The other modules carried 22 percent [F114]. One matrix work proposed by Muse Wonder gave C8: O1. Layer 2 v alone gave 57 percent [F125]. Necessity work proposed by Muse Wonder gave C9: N1. Without that matrix the dial kept only 53 percent [F168]. Its own heads changed by -0.727 [F121]. Later layers changed only a little. A downstream control passed narrowly at 0.8044 [F177].

The final point is exploratory probes map the bias. This means we see where it sits and what it does. Exploratory head maps show median range 32 [F148] tokens and upper range 614.4 [F149] tokens. Exploratory behavior showed a 5 percent [F179] rise after a topic switch. The rise is measurable. Generated text showed 0 [F183] old book names in every condition. So the effect does not show in visible text. Open work is why slow bias helps modeling and how to remove it fully.

---
Verification: every number matches a cited fact.

Sources:
- [F12] C1 set B: gain at the shortest period, relative to the maximum: `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.per_set.B.min_normalized`
- [F76] C4 local sliding-window size: change in tilt from low to high: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window.delta`
- [F77] C4 local sliding-window size: change needed for a dial: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window.needed`
- [F78] C4 local sliding-window size: worst perplexity ratio: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window`
- [F92] C5 relative tilt change at scale 2: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.relative_change`
- [F97] C5 worst perplexity ratio over scales: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.worst_ppl_ratio`
- [F104] C6 relative rise in topic-switch excess loss at scale 2 (10 fresh books, 90 pairs): `results/derived/c6v2-20261029T060851Z/c6_summary.json#C6.relative_rise`
- [F113] C7 share of the change from layer 2 q, k, v alone (3 of 32 modules): `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.share_QKV2`
- [F114] C7 share of the change from the other 29 modules: `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.share_REST`
- [F121] C7 relative tilt change over layer 2's own heads, layer 2 q, k, v only: `results/derived/c7v1-20261029T154201Z/c7_summary.json#diagnostic_groups.layer2`
- [F125] C8 share of the full change from layer 2 v_proj alone (1 of 32 modules): `results/derived/c8v1-20261029T222946Z/c8_summary.json#C8.share_V2`
- [F148] EXPLORATORY head map: median listening range of a head (tokens): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.corner_percentiles`
- [F149] EXPLORATORY head map: 90th percentile listening range (tokens): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.corner_percentiles`
- [F168] C9 share of the full change kept by the full dial without layer 2 v_proj, layers 2 to 51: `results/derived/c9v1-20261030T023935Z/c9_summary.json#C9.retained_NOV2`
- [F177] C9 downstream: share kept without layer 4 v_proj (control, bar 0.80): `results/derived/c9v1-20261030T023935Z/c9_summary.json#C9.downstream.retained_NOV4`
- [F179] EXPLORATORY behavior probe (10 book pairs, sets B and C from token 80,000): relative rise in loss on the new book after a switch, the trained dial at scale 2: `results/raw/probe-behavior-20261030T022546Z/behavior.json#summary.S2`
- [F183] EXPLORATORY behavior probe: old-book names in all generated text, every condition: `results/raw/probe-behavior-20261030T022546Z/behavior.json#summary`
