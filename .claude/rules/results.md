---
paths:
  - "results/**"
  - "docs/**"
  - "*.md"
---

# Results and reporting rules

- Every number in prose names the file it came from, as `results/raw/<run_id>/<file>.json`.
- PASS or FAIL is computed by harness code against the sealed thresholds. Never restate a verdict the harness did not produce.
- A failed check is reported as failed, with its numbers. Do not reframe a pre-registered criterion after seeing data.
- When two runs differ in precision, hardware, model, or adapter strength, say so beside the comparison.
- Plot titles state the finding, and axes carry units.
