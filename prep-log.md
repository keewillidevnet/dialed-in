# Prep log

Dated record of all work done before the Oct 26, 2026 kickoff. Planning, tooling, and setup only. No experiment code, sweeps, adapter training, or front end before kickoff.

| Date | What was done | Experiment code? |
| --- | --- | --- |
| 2026-10-20 | Design doc drafted with pre-registered checks C1 to C3 | No |
| 2026-10-21 | Requested access to the Meta Llama 3.1 collection on Hugging Face | No |
| 2026-10-21 | Lab tooling: CLAUDE.md, rules, protection hooks, seal script | No |
| 2026-10-21 | Llama 3.1 collection access approved on Hugging Face | No |
| 2026-10-22 | RunPod: 100 GB network volume, RTX PRO 6000 pod, image runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404 | No |
| 2026-10-22 | Downloaded Llama-3.1-8B-Instruct @ 0e9e39f249a1 and Llama-3.1-8B @ d04e592bb4f6 to /workspace/weights, verified 4 shards each | No |
| 2026-10-22 | Smoke test passed on both models: config, hooks on all 32 layers, attentions, BF16 outputs. NF4 load works. Report: /workspace/env/smoke_20261022T184651Z.json | No |
| 2026-10-22 | Capacity check, random tokens. Eager: 8K fits (36.5 GiB), 16K out of memory; with attention weights 4K fits (51.7 GiB). SDPA: 8K 0.65 s 15.9 GiB, 16K 0.97 s 16.8 GiB, 32K 2.33 s 18.6 GiB, 64K 6.25 s 22.2 GiB | No |
| 2026-10-26 | After kickoff, read the official rules. Switched the subject model to Muse Glimmer-30B, an eligible Meta open-weight model. Removed base Llama-3.1-8B from the volume | No |
| 2026-10-26 | Downloaded meta-models/Muse-Glimmer-30B @ cec94fcfd651 (56 GB, 2 shards) | No |
| 2026-10-26 | Inspected Glimmer (transformers 5.21.0): layer layout, gate location, residual order, capacity 8K to 64K. Report: /workspace/env/glimmer_inspect_20261027T005358Z.json | No |
| 2026-10-26 | Building period starts Oct 27, 12:00 AM ET. All experiment code is written from then on | No |
