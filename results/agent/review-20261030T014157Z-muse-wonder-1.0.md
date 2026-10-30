Reviewed files: criteria/drafts/c9-glimmer-necessity-v1.yaml, src/dialedin/run_c9.py, src/dialedin/metrics_c9.py, src/dialedin/analyze_c9.py, tests/test_c9.py

Verdict: NOT READY

1. Blocker, criteria/drafts/c9-glimmer-necessity-v1.yaml budget.pre_seal_smoke line 296 and src/dialedin/run_c9.py lines 118 to 130 and 718. The field still holds SMOKE_RESULT. The runner refuses any sealed spec containing that placeholder, so sealing this text unchanged would make the run unrunnable before STARTED.json. Fix: run the smoke test on the final spec and code, replace SMOKE_RESULT with the actual smoke result, then seal and rerun the draft spec test.

2. Major, criteria line 239 and src/dialedin/analyze_c9.py lines 264 to 265. The criteria require a partial look to be written to a directory ending in PARTIAL. The code builds a directory named with run id, then PARTIAL, then a timestamp, so the directory name ends in the timestamp. Fix: name partial output directories so the final name ends in PARTIAL while keeping each partial directory unique.

3. Major, criteria line 239 and src/dialedin/analyze_c9.py lines 98 to 105 and 140 to 153. A partial analysis with zero book files will crash when set_tilts stacks empty curve lists, and group_diagnostic at line 120 also assumes books[0] exists. The criteria say an incomplete run gets a partial diagnostic with C9.outcome null. Fix: add an explicit zero book partial branch that returns outcome null with the warning, or refuse clearly before any stacking.

4. Major, criteria questions lines 50 to 51, pass_criteria lines 220 to 225, and src/dialedin/metrics_c9.py lines 84 to 99. The question says both windows are judged only if V2 passes both the main gate and the downstream gate. The operative rules judge the main window on the main gate alone and can return N1L when the downstream gate fails. Fix: rewrite the question to match the operative rules, or change the outcome code if both gates were intended to gate the main judgment.

5. Minor, criteria questions lines 47 to 49, pass_criteria line 221, and src/dialedin/metrics_c9.py lines 29 to 36. The question book clause omits the requirement that FULL flatten that book. The operative rule requires FULL minus BASE to be negative for the book before ordering NOV2 against NOG2. Fix: add that condition to the question text.

6. Minor, criteria lines 275 to 276 and src/dialedin/run_c9.py lines 483 to 488 and 523 to 530. A continued run with an earlier failure record must have the reason stamped into phaseA.json with the list of earlier failure records. The normal phaseA record does this, but the setup check failure record includes restart_reason and omits earlier_failures. Fix: add the same earlier_failures field to the setup check failure phaseA.json.
