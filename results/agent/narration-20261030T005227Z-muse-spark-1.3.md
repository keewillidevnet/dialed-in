The model favors slow patterns. That means long range signals pass best. We add a small sine wave after an early layer and measure how strongly later heads follow it for periods from 2 [F4] to 4096 [F5] tokens. The sealed shape test for C1 returned T. The shortest period reaches only 35 percent [F8] of peak response.

No built in setting could turn this down. That means the bias is stubborn. Changing global and local gains left tilt flat for C2, which returned K0. Boosting chosen heads also failed for C3, which returned K0. Changing window size and rotary base failed for C4, which returned D0 for both levers. The text control for C4 returned A.

A trained dial can flatten tilt but missed its goal with little behavior change. That means control is possible at small cost. The sealed training test for C5 returned L0, with tilt down by only -24 percent [F92] at full strength. General cost stayed tiny at 1.008 [F97]. The sealed behavior test for C6 returned B0, with only a 2 percent [F104] rise in topic switch loss.

The dial acts through a tiny early site. That means an early tweak spreads widely. The sealed localization test for C7 returned M1, with about 80 percent [F113] of the full change from one early attention block alone. The follow up proposed by Muse Wonder for C8 returned O1, with about 57 percent [F125] from a single value matrix alone.

The exploratory head map suggests a mix of ranges. That means average tilt hides variety. Exploratory fits show median range of 32 [F148] tokens with a long tail to 614.4 [F149] tokens. These findings are exploratory only. Open work is why the bias resists built in controls, whether stronger training can hold safely, and how range mix shapes real use.

---
Verification: every number matches a cited fact.

Sources:
- [F4] C1 drive periods N, shortest: `results/derived/c1v5-20261027T152040Z/c1_summary.json#grid`
- [F5] C1 drive periods N, longest: `results/derived/c1v5-20261027T152040Z/c1_summary.json#grid`
- [F8] C1 set A: gain at the shortest period, relative to the maximum: `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.per_set.A.min_normalized`
- [F92] C5 relative tilt change at scale 2: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.relative_change`
- [F97] C5 worst perplexity ratio over scales: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.worst_ppl_ratio`
- [F104] C6 relative rise in topic-switch excess loss at scale 2 (10 fresh books, 90 pairs): `results/derived/c6v2-20261029T060851Z/c6_summary.json#C6.relative_rise`
- [F113] C7 share of the change from layer 2 q, k, v alone (3 of 32 modules): `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.share_QKV2`
- [F125] C8 share of the full change from layer 2 v_proj alone (1 of 32 modules): `results/derived/c8v1-20261029T222946Z/c8_summary.json#C8.share_V2`
- [F148] EXPLORATORY head map: median listening range of a head (tokens): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.corner_percentiles`
- [F149] EXPLORATORY head map: 90th percentile listening range (tokens): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.corner_percentiles`
