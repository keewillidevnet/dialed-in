The lab injected a sine wave after an early layer and recorded head gain across drive periods from 2 [F4] to 4096 [F5].
C1 used 15 [F3] books. Gain rose monotonically with period. Spearman rho was 1 [F7]. Shortest period gain was 0.3525 [F8] of maximum. Tilt slope was 0.1334 [F9]. Repeat sets matched with shape agreement 1 [F18]. Outcome was T [F1] with LP [F2] direction.
C2 tested global and local layer gains. Outcome was K0 [F21]. Global knob shift changed tilt by 0.005953 [F28]. Local knob shift changed tilt by 0.01367 [F29]. 14 [F45] books went steeper under global up, and 15 [F46] under local up. Reproduction matched with largest difference 0 [F48].
C3 boosted selected heads. Outcome was K0 [F49]. Selected shift was -0.00217 [F58]. Control shift was -0.009814 [F59]. Only 5 [F70] books went steeper with selected up, while 0 [F71] did with random up.
C4 changed window size and RoPE base. Outcomes were D0 [F74] and D0 [F79]. Window shift changed tilt by -0.02324 [F76] against needed 0.03428 [F77]. RoPE shift changed tilt by 0.01118 [F81] against needed 0.03428 [F82]. Worst perplexity ratios were 1.472 [F78] and 1.119 [F83]. Text control outcome was A [F84]. Shuffled tokens kept 0.9703 [F85] of tilt. Repeated token kept 0.5066 [F86].
C5 trained 32 [F88] low rank modules of rank 8 [F88]. Outcome was L0 [F87]. Tilt fell from 0.1371 [F89] to 0.1042 [F91]. Relative change was -0.2402 [F92] with interval -0.2621 [F93] to -0.2183 [F94]. This missed the preregistered flattening threshold. Spearman against scale was -1 [F95]. All 5 [F96] books agreed. Worst perplexity ratio was 1.008 [F97].
Exploratory recall stayed at 20 [F100] correct. Topic switch excess loss rose from 509.6 [F101] to 567.5 [F102] in exploration. Sealed C6 outcome was B0 [F103]. Sealed rise was 0.0177 [F104] with interval 0.005412 [F105] to 0.02959 [F106]. 8 [F107] books rose. Random adapter rise was -0.0006696 [F108]. Perplexity ratio was 1.009 [F110].
C7 outcome was M1 [F111]. Full change was -0.268 [F112]. Share from layer 2 [F113] q k v alone was 0.7971 [F113]. Share from the rest was 0.2191 [F114]. Interval was 0.7818 [F115] to 0.8115 [F116]. All 5 [F119] books agreed. Perplexity ratio was 1.004 [F120]. Change over that layer own heads was -0.727 [F121].
Exploratory mapping resolved 458 [F130] heads. Median listening range was 32 [F131] tokens. Upper percentile was 614.4 [F132] tokens. Median tilt was 0.1755 [F133] in local layers and 0.04079 [F134] in global layers. Muting short range heads raised loss by 1.347 [F140] versus control max 0.3244 [F141]. Muting long range heads raised loss by 0.2048 [F143] versus control max 0.4677 [F144].

---
Verification: every number matches a cited fact.

Sources:
- [F1] C1 outcome (sealed rules: R resonance, HP/LP cutoff, T tilt, F flat, S structured, U inconclusive): `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.outcome`
- [F2] C1 tilt direction: `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.tilt_direction`
- [F3] C1 books: `results/derived/c1v5-20261027T152040Z/c1_summary.json#books`
- [F4] C1 drive periods N, shortest: `results/derived/c1v5-20261027T152040Z/c1_summary.json#grid`
- [F5] C1 drive periods N, longest: `results/derived/c1v5-20261027T152040Z/c1_summary.json#grid`
- [F7] C1 set A: Spearman rho of gain vs N: `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.per_set.A.rho_gain_vs_N`
- [F8] C1 set A: gain at the shortest period, relative to the maximum: `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.per_set.A.min_normalized`
- [F9] C1 set A: tilt (slope of ln gain vs ln N, computed from the curve): `results/derived/c1v5-20261027T152040Z/c1_summary.json#curves.A.whole`
- [F18] C1 shape agreement between sets A-B (Spearman rho): `results/derived/c1v5-20261027T152040Z/c1_summary.json#C1.shape_replication.A-B.rho`
- [F21] C2 outcome (sealed rules: K1 tunable and specific, K2 tunable not specific, K0 not tunable, U inconclusive): `results/derived/c2v1-20261027T201901Z/c2_summary.json#C2.outcome`
- [F28] C2 set A: global-layer knob, change in tilt from 0.5 to 1.5: `results/derived/c2v1-20261027T201901Z/c2_summary.json#C2.per_set.A.delta_global`
- [F29] C2 set A: local-layer knob, change in tilt from 0.5 to 1.5: `results/derived/c2v1-20261027T201901Z/c2_summary.json#C2.per_set.A.delta_local`
- [F45] C2 books: global_up_steeper: `results/derived/c2v1-20261027T201901Z/c2_summary.json#book_signs.global_up_steeper`
- [F46] C2 books: local_up_steeper: `results/derived/c2v1-20261027T201901Z/c2_summary.json#book_signs.local_up_steeper`
- [F48] C2 reproduction of C1: largest relative difference: `results/derived/c2v1-20261027T201901Z/c2_summary.json#diagnostic_G1_vs_C1v5.max_relative_difference`
- [F49] C3 outcome (sealed rules: K1 tunable and specific, K2 tunable not specific, K0 not tunable, U inconclusive): `results/derived/c3v1-20261028T014250Z/c3_summary.json#C3.outcome`
- [F58] C3 set B: selected heads, change in tilt from 0 to 2: `results/derived/c3v1-20261028T014250Z/c3_summary.json#C3.per_set.B.delta_selected`
- [F59] C3 set B: size-matched control heads, change in tilt from 0 to 2: `results/derived/c3v1-20261028T014250Z/c3_summary.json#C3.per_set.B.delta_random`
- [F70] C3 books: selected_up_steeper: `results/derived/c3v1-20261028T014250Z/c3_summary.json#book_signs.selected_up_steeper`
- [F71] C3 books: random_up_steeper: `results/derived/c3v1-20261028T014250Z/c3_summary.json#book_signs.random_up_steeper`
- [F74] C4 local sliding-window size lever outcome (sealed rules: D1 a dial, D0 no dial): `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window.outcome`
- [F76] C4 local sliding-window size: change in tilt from low to high: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window.delta`
- [F77] C4 local sliding-window size: change needed for a dial: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window.needed`
- [F78] C4 local sliding-window size: worst perplexity ratio: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.window`
- [F79] C4 local RoPE base lever outcome (sealed rules: D1 a dial, D0 no dial): `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.rope.outcome`
- [F81] C4 local RoPE base: change in tilt from low to high: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.rope.delta`
- [F82] C4 local RoPE base: change needed for a dial: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.rope.needed`
- [F83] C4 local RoPE base: worst perplexity ratio: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.rope`
- [F84] C4 text control outcome (A: the tilt belongs mostly to the model): `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.text.outcome`
- [F85] C4 tilt with shuffled tokens, as a fraction of the book's tilt: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.text.shuffled_over_base`
- [F86] C4 tilt with one repeated token, as a fraction of the book's tilt: `results/derived/c4v1-20261028T065820Z/c4_summary.json#C4.text.constant_over_base`
- [F87] C5 trained-dial outcome (sealed rules: L1 a dial, L2 a dial with damage, L0 no dial; needs a 25 percent flattening at scale 2): `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.outcome`
- [F88] C5 adapter: 32 low-rank modules in layers 2 to 5, rank: `criteria/sealed/c5-glimmer-trained-dial-v1.yaml#training.rank`
- [F89] C5 tilt on set A at dial scale 0: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.tilts.S0`
- [F91] C5 tilt on set A at dial scale 2: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.tilts.S2`
- [F92] C5 relative tilt change at scale 2: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.relative_change`
- [F93] C5 relative tilt change at scale 2, 95 percent interval low: `results/derived/c5v1-20261029T025343Z/c5_summary.json#bootstrap_95.relative_change_S2`
- [F94] C5 relative tilt change at scale 2, 95 percent interval high: `results/derived/c5v1-20261029T025343Z/c5_summary.json#bootstrap_95.relative_change_S2`
- [F95] C5 Spearman of tilt against scale: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.spearman`
- [F96] C5 books agreeing (of 5): `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.books_agree`
- [F97] C5 worst perplexity ratio over scales: `results/derived/c5v1-20261029T025343Z/c5_summary.json#C5.worst_ppl_ratio`
- [F100] C5 exploratory recall: passkey trials correct at scale 2 (of 20): `results/derived/c5v1-20261029T025343Z/c5_summary.json#behavior.recall.S2`
- [F101] C5 exploratory topic-switch excess loss at scale 0 (nats): `results/derived/c5v1-20261029T025343Z/c5_summary.json#behavior.switch.S0`
- [F102] C5 exploratory topic-switch excess loss at scale 2 (nats): `results/derived/c5v1-20261029T025343Z/c5_summary.json#behavior.switch.S2`
- [F103] C6 topic-switch outcome (sealed rules: B1 confirmed needs a 5 percent rise; B0 not confirmed): `results/derived/c6v2-20261029T060851Z/c6_summary.json#C6.outcome`
- [F104] C6 relative rise in topic-switch excess loss at scale 2 (10 fresh books, 90 pairs): `results/derived/c6v2-20261029T060851Z/c6_summary.json#C6.relative_rise`
- [F105] C6 relative rise, 95 percent interval low: `results/derived/c6v2-20261029T060851Z/c6_summary.json#bootstrap_95_relative_rise`
- [F106] C6 relative rise, 95 percent interval high: `results/derived/c6v2-20261029T060851Z/c6_summary.json#bootstrap_95_relative_rise`
- [F107] C6 target books rising (of 10): `results/derived/c6v2-20261029T060851Z/c6_summary.json#C6.books_agree`
- [F108] C6 random adapter of matched size: relative rise at scale 2: `results/derived/c6v2-20261029T060851Z/c6_summary.json#control_random_adapter.relative_rise`
- [F110] C6 perplexity ratio at scale 2: `results/derived/c6v2-20261029T060851Z/c6_summary.json#secondary.clean_perplexity_ratio.S2`
- [F111] C7 localization outcome (sealed rules: M1 localized needs layer 2 q, k, v alone at 60 to 140 percent of the full dial's tilt change, the other 29 modules within 30 percent, 4 of 5 books): `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.outcome`
- [F112] C7 full dial: relative tilt change on set B at scale 2: `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.full_change`
- [F113] C7 share of the change from layer 2 q, k, v alone (3 of 32 modules): `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.share_QKV2`
- [F114] C7 share of the change from the other 29 modules: `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.share_REST`
- [F115] C7 share_QKV2, 95 percent interval low: `results/derived/c7v1-20261029T154201Z/c7_summary.json#bootstrap_95.share_QKV2`
- [F116] C7 share_QKV2, 95 percent interval high: `results/derived/c7v1-20261029T154201Z/c7_summary.json#bootstrap_95.share_QKV2`
- [F119] C7 books agreeing (of 5): `results/derived/c7v1-20261029T154201Z/c7_summary.json#C7.books_agree_QKV2`
- [F120] C7 perplexity ratio with only layer 2 q, k, v on: `results/derived/c7v1-20261029T154201Z/c7_summary.json#perplexity_ratio_median.QKV2`
- [F121] C7 relative tilt change over layer 2's own heads, layer 2 q, k, v only: `results/derived/c7v1-20261029T154201Z/c7_summary.json#diagnostic_groups.layer2`
- [F130] EXPLORATORY head map: heads resolved individually (of 1,600): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.heads_resolved`
- [F131] EXPLORATORY head map: median listening range of a head (tokens): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.corner_percentiles`
- [F132] EXPLORATORY head map: 90th percentile listening range (tokens): `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary.corner_percentiles`
- [F133] EXPLORATORY head map: median head tilt, local layers: `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary`
- [F134] EXPLORATORY head map: median head tilt, global layers: `results/derived/headmap-c1v5-20261027T152040Z/headmap.json#summary`
- [F140] EXPLORATORY band mute: loss change, all tokens, muting the short-range heads: `results/raw/probe-bandmute-20261029T170639Z/bandmute.json#summary.relative_change.short`
- [F141] EXPLORATORY band mute: loss change, all tokens, largest of the short band's random controls: `results/raw/probe-bandmute-20261029T170639Z/bandmute.json#summary.relative_change`
- [F143] EXPLORATORY band mute: loss change, all tokens, muting the long-range heads: `results/raw/probe-bandmute-20261029T170639Z/bandmute.json#summary.relative_change.long`
- [F144] EXPLORATORY band mute: loss change, all tokens, largest of the long band's random controls: `results/raw/probe-bandmute-20261029T170639Z/bandmute.json#summary.relative_change`
