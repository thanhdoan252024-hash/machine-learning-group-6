# 04 — Stage Gates and QA

| Gate | Must pass before next stage |
|---|---|
| P01 | X=(5620,64), no NaN/Inf, no `Unnamed` index feature |
| P02 | distance symmetric/diag0/nonnegative + brute-force match |
| P03 | probabilities sum1, self0, beta/perplexity behavior |
| P04 | achieved perplexity, P symmetry/diag0/sum1 |
| P05 | Q symmetry/sum1, KL(P||P)=0, translation invariance |
| P06 | analytic=brute-force; finite-difference relative error acceptable |
| P07 | optimizer finite, non-collapse, restored KL=best KL |
| P08 | synthetic local structure > random baseline |
| P09 | every grid config valid; no winner yet |
| P10 | T/C identity test=1; all metrics in [0,1] |
| P11 | all candidate×seed runs + structural stability |
| P12 | final full embedding and exported artifacts |
| P13 | package audit + ZIP integrity |
