# Dialed In

Measures the frequency response of Muse Glimmer-30B by injecting pulses every N tokens, then tunes that response live by scaling its global attention layers. Entry for the Meta Global AI Developer Hackathon (Indie track, Theme 2). Build window: Oct 26 to Nov 4, 2026, submission due 9 PM ET.

The design doc is the source of truth for scope and pre-registered checks: https://claude.ai/artifact/2LoX5QTFVLJk96vtX5TGtM

## Hard rules

1. **Event work only.** Official rules: everything shown must be built during the event (building period Oct 27 to Nov 4), and judges must be able to tell what was built when. Files made before kickoff (CLAUDE.md, rules, hooks, `scripts/seal.sh`, `scripts/verify_seals.py`, `scripts/setup_pod.sh`, `scripts/smoke_test.py`) are setup, listed in `prep-log.md`, and never presented as part of the product.
2. **Sealed criteria are never edited.** Files in `criteria/sealed/` are locked by Keenan with `scripts/seal.sh`. Claude may read them, never write, move, or re-seal them. A changed criterion means a new draft and a new seal, with the old one kept.
3. **Metrics live in one module.** Every metric is imported from `src/dialedin/metrics.py` and never redefined inline. That file is protected; propose changes in chat for Keenan to apply.
4. **Raw results are immutable.** Nothing in `results/raw/` is edited after it is written. Derived numbers go in `results/derived/` and cite the raw file they came from.
5. **Numbers trace to primary JSON.** Any number cited in a doc, slide, or commit message must come from a file in `results/`, never from memory or tracker notes.
6. **Muse is the AI scientist; Claude is lab tooling.** No Anthropic SDK imports or Claude API calls anywhere under `src/` or the front end. Muse Spark (Meta Model API) handles proposing, sealing requests, and narration.
7. **BF16 everywhere.** Every sealed run (C1, C1b, C2, C3), the knob, and any stretch adapter use BF16 with no quantization config. A stretch adapter is trained and run at the same precision. NF4 is allowed only as a separately labeled C1 side check, run last, never mixed into C2 or C3, and it cannot rescue a failed C1. Every run asserts `model.dtype == torch.bfloat16` and no `quantization_config` before Phase A, and writes its precision into the result JSON.
8. **Fork, never rebuild.** A new experiment starts as a literal copy of the most recent successful notebook or script, on its own worktree branch.

## Platform

- RunPod RTX PRO 6000 Blackwell (96 GB). Image `runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404`: PyTorch 2.8.0, CUDA 12.8.1. The image ships without transformers or huggingface_hub; install pinned versions per session and record actual versions with every result. Verified on Oct 22: Python 3.12.3, torch 2.8.0+cu128, transformers 5.9.0, peft 0.19.1, bitsandbytes 0.49.2, accelerate 1.15.0, huggingface_hub 1.33.0, compute capability 12.0. Glimmer needs a newer transformers than 5.9.0; verified on Oct 26 with transformers 5.21.0. Install line: `pip install -q "transformers==5.21.0" "accelerate==1.15.0" "torchvision==0.23.0" "huggingface_hub==1.33.0"`.
- Subject model: `meta-models/Muse-Glimmer-30B` (Apache 2.0), revision pinned in `/workspace/weights/DOWNLOAD_LOG.txt` at download. About 60 GB in BF16. Llama-3.1-8B-Instruct (`0e9e39f249a1`) stays on the volume as an infrastructure reference only; it is not part of the submission.
- Weights live at `/workspace/weights/<model name>` on the network volume. Load from there, never from the Hub. Set `HF_HOME=/workspace/.cache/huggingface`.
- Record precision and hardware with every result. Never compare results across precision or hardware without saying so. All sealed runs use the same GPU model.

## Model loading

- Sealed runs use `attn_implementation="sdpa"`. The sweep reads head outputs through hooks on the attention output projection, which SDPA supports. Glimmer's long-input capacity must be measured on day 1 before the criteria are sealed (the Oct 22 numbers were for Llama-3.1-8B). Never mix attention modes inside or across sealed runs; the numbers differ slightly.
- `attn_implementation="eager"` only for exploratory work that needs attention weights. SDPA never materializes them.
- Use `dtype=`, not the deprecated `torch_dtype=`.
- `output_attentions=True` goes on the forward call, not `from_pretrained()`.
- HF auth: `from huggingface_hub import get_token, notebook_login, whoami`. Check `get_token()` and env vars before `notebook_login()`.
- Glimmer: 52 decoder layers in a repeating [local, local, local, global] pattern (13 global). Local layers use a 2,048-token sliding window with RoPE; global layers use full attention with no position encoding. 32 query heads, 2 key-value heads, head dim 128, hidden size 6,656, gated attention, vocab 202,048. Loads with `AutoModelForMultimodalLM` in about 134 s, 55.5 GiB. Decoder layers: `model.language_model.layers`; global layers are indices 3, 7, 11, ..., 51 (`self_attn.is_local_attention == False`). Attention modules: `q_proj` (6656 to 4096), `k_proj` and `v_proj` (6656 to 256), `gate_proj` (6656 to 4096, the output gate), `o_proj` (4096 to 6656). Each layer has sandwich norms: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`.
- Head metrics are read with a forward pre-hook on `o_proj`: its input is the gated head outputs, shape [batch, tokens, 4096] = 32 heads x 128. Every result is tagged with its layer index and local or global.
- The knob scales the OUTPUT of `post_attention_layernorm` in global layers, right before the residual add. Never scale `o_proj` output: the post-attention RMSNorm would cancel any g other than 0. Confirm the residual order in the layer's forward source before the first knob run. At g = 1 outputs must match the unhooked model exactly.
- Capacity, decoder only, SDPA, BF16 (Oct 26): 8K 1.87 s 56.9 GiB; 16K 4.61 s 58.4 GiB; 32K 12.68 s 61.8 GiB; 64K 39.28 s 72.0 GiB. Storing every head output at 64K would add about 28 GB, so hooks reduce on the fly and keep only per-head summaries.

## Experiment discipline

- Each run has a sealed criteria file before it starts. The run script refuses to start if the referenced seal's hash does not match `criteria/SEALS.txt`.
- Linearity check runs first. Sham (zero-amplitude) and norm-matched random pulses run as controls in every sweep.
- Rank statistics (Spearman rho) for small-n comparisons. Do not report Cohen's d at n below 10.
- Real text prompts only, never toy text.
- Per-run spend and runtime cap set in the config.
- Upload adapters, checkpoints, and raw results to private durable storage before any pod teardown.

## IP and disclosure

- Nothing from prior private research goes into this repo: no head lists, layer zones, adapter configs, or results from Mistral or Phi experiments.
- Do not name the prior research program in any file, commit, or public text.
- The rules require a public repo at submission. Keep it private until Keenan confirms the provisional is filed or the attorney has cleared what may be shown.
- Glimmer is Apache 2.0: keep its LICENSE and NOTICE text with any redistributed weights or derivatives.

## Writing

- No em dashes anywhere, including code comments and commit messages.
- No "not X, it's Y" constructions.
- Short declarative sentences. Plain words.

## Layout

```
criteria/drafts/     proposed criteria, editable
criteria/sealed/     locked criteria, read-only
criteria/SEALS.txt   hash log written by scripts/seal.sh
src/dialedin/        harness, agent loop, knob, metrics (built during the event)
notebooks/           exploratory notebooks (built during the event)
results/raw/         immutable run output
results/derived/     analysis that cites raw files
prep-log.md          dated log of all pre-kickoff setup
```

Protected paths are listed in `.claude/protected.txt` and enforced by hooks in `.claude/hooks/`.
