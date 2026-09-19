# Final t-SNE Case Study Report

## Objective
Implement exact t-SNE from scratch with NumPy on UCI Optdigits, verify mathematical correctness, evaluate neighborhood preservation, test stochastic stability, and produce a final 2D visualization of all 5,620 samples.

## Completed protocol
P00–P08 validate data integrity, core mathematics, gradient correctness, optimizer behavior, and synthetic structure. P09 evaluates six perplexities. P10 evaluates multi-scale Trustworthiness/Continuity on the deterministic 1,000-sample screening subset. P11 evaluates candidate stability across three seeds. P12 runs the selected primary configuration on all 5,620 samples **and directly evaluates the final full-data embedding with Trustworthiness/Continuity at k={5,10,20,50}**. P13 audits and packages the artifacts.

## Final selection
- Primary perplexity: **40**
- Alternative perplexity: **30**

The primary is selected by the pre-declared lexicographic protocol prioritizing robust neighborhood preservation; p=20 has stronger worst seed stability, so p=40 is not presented as universally best or globally optimal.

## Full-run result
- Input: `5620 × 64`
- Embedding: `5620 × 2`
- Perplexity: `40`
- Random seed: `42`
- Initial KL: `4.806500153`
- Final/restored KL: `1.081797712`
- Iterations: `1000`
- Reference runtime: `1808.58 s` (`30.14` minutes)

## Direct quality evaluation of the final 5,620-sample embedding

| k | Trustworthiness | Continuity |
|---:|----------------:|-----------:|
| 5  | 0.996845 | 0.992437 |
| 10 | 0.994644 | 0.989400 |
| 20 | 0.991689 | 0.984876 |
| 50 | 0.986279 | 0.977478 |

Mean Trustworthiness is `0.992364`, mean Continuity is `0.986048`, and the worst score across the eight reported full-data neighborhood metrics is `0.977478`. These metrics directly evaluate the final artifact rather than inferring its quality only from the screening subset.

## Validation
Independent audits reproduced P10/P11 metrics from stored embeddings, recomputed the new P12 full-data Trustworthiness/Continuity table from the saved final embedding, and independently rebuilt P12's full high-dimensional probability distribution to recompute the final KL. All audits passed.

## Limitations
Exact t-SNE has O(N²) memory/time cost; this implementation is non-parametric and has no simple unseen-sample transform; embeddings remain stochastic; and global 2D distances/areas should not be interpreted literally. The full-data T/C scores validate local-neighborhood preservation, not global geometry.
