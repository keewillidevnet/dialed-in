# Criteria drafts

Write proposed criteria here. When Keenan signs off, he runs:

    scripts/seal.sh criteria/drafts/<name>.yaml

That copies the file to `criteria/sealed/`, makes it read-only, and logs its hash in `criteria/SEALS.txt`. Commit both right after sealing.

Each draft states, before any data exists: the question, the model and precision, the interval grid, prompt sets and seeds, the statistic, the pass threshold, and what happens on fail.
