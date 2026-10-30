Critique of proposal-20261030T043046Z-muse-wonder-1.0.json

1. Downstream denominator is small with no guard. The cited full change over layers 5 to 51 is -0.05161 [F175] on different text. A near zero per book full drop makes the share unstable. Fix: add a no call rule that requires a minimum full drop magnitude per book and overall, else result is inconclusive.

2. Necessity bar has no book level replication. Pass requires mean share without layer 2 q,k,v at most 0.40 but allows one book to drive the mean. Fix: require at least 4 of 5 books to show share without below share with, or below 0.40.

3. Sufficiency bar has no precision rule. With 5 books and a small denominator the mean 0.60 can be noisy. Fix: require the 95 percent interval for the layer 2 share to lie clearly above the layer 4 share, or pre register a maximum interval width.

4. Burn in accounting is unclear. The range 90,000 to 122,767 is exactly 32,768 tokens. Fix: state that the first 8,192 tokens are burn in and only the remainder is scored, consistent with the harness.

5. Training overlap check is not stated. The adapter trained on books 1260, 345, 37106 and 174 at tokens 40,960 to 47,103. Fix: confirm in the smoke run that the 5 measured segments avoid those book token ranges.

6. Propagation distance is not matched. Both interventions are local and upstream of layer 5, but layer 2 must pass through global layer 3 while layer 4 does not. Fix: keep layer 4 as the closest possible local control but state this path difference as a limit on specificity claims.

Quoted numbers match: 0.7971 [F113], 0.3816 [F138], and -0.05161 [F175] are cited with the correct share denominators.

Verdict: FIXABLE
