---
paths:
  - "notebooks/**"
  - "**/*.ipynb"
---

# Notebook rules

Cell order on a fresh pod:

1. Title and run ID
2. `!pip install transformers bitsandbytes peft accelerate scipy scikit-learn huggingface_hub matplotlib -q`
3. Preflight: weights present on the volume, library versions match CLAUDE.md, precision recorded, adapter files present when used
4. HF auth (`get_token()` first)
5. Imports, with all metrics from `dialedin.metrics`
6. CONFIG cell: model ID, precision, seed list, interval grid, spend cap, and the path of the sealed criteria file
7. Seal check: hash the referenced criteria file and compare to `criteria/SEALS.txt`; stop on mismatch
8. Phase A: linearity check and sham control
9. Calibration gate: stop unless Phase A passes
10. Phase B and later

Every notebook writes its raw output to `results/raw/<run_id>/` and never overwrites an existing run folder.
