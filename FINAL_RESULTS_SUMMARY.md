# FINAL RESULTS — t-SNE From Scratch on UCI Optdigits

## Status

**READY FOR SUBMISSION** after completed P00–P13 validation.

## Dataset

- UCI Optdigits
- 5,620 samples
- 64 pixel features
- 10 labels used only for stratified screening/post-hoc visualization, never for fitting

## P09 — Perplexity sensitivity

|   Perplexity |   Final Restored KL |   Runtime Seconds |
|-------------:|--------------------:|------------------:|
|     5.000000 |            0.903895 |         18.561123 |
|    10.000000 |            0.779650 |         19.237640 |
|    20.000000 |            0.658250 |         19.083580 |
|    30.000000 |            0.633404 |         19.410990 |
|    40.000000 |            0.571283 |         18.667512 |
|    50.000000 |            0.557010 |         19.519004 |

KL is retained as an optimization diagnostic and is **not** used as the cross-perplexity winner criterion.

## P10 — Neighborhood preservation

|   Perplexity |      T@5 |      C@5 |     T@50 |     C@50 |   Worst Neighborhood Score |
|-------------:|---------:|---------:|---------:|---------:|---------------------------:|
|     5.000000 | 0.989075 | 0.976439 | 0.946396 | 0.922348 |                   0.922348 |
|    10.000000 | 0.991768 | 0.984870 | 0.960646 | 0.934227 |                   0.934227 |
|    20.000000 | 0.991901 | 0.987188 | 0.966515 | 0.943934 |                   0.943934 |
|    30.000000 | 0.991574 | 0.984373 | 0.967537 | 0.946435 |                   0.946435 |
|    40.000000 | 0.992413 | 0.987027 | 0.968599 | 0.949436 |                   0.949436 |
|    50.000000 | 0.991535 | 0.986856 | 0.967567 | 0.949171 |                   0.949171 |

## P11 — Multi-seed robustness

|   Perplexity |   Mean Neighborhood Score |   Robust Worst Neighborhood |   Mean Seed Stability |   Worst Seed Stability |
|-------------:|--------------------------:|----------------------------:|----------------------:|-----------------------:|
|    20.000000 |                  0.974909 |                    0.938746 |              0.839395 |               0.816950 |
|    30.000000 |                  0.976747 |                    0.946435 |              0.822436 |               0.797400 |
|    40.000000 |                  0.977271 |                    0.949436 |              0.825065 |               0.786400 |

**PRIMARY = 40**  
**ALTERNATIVE = 30**

The primary is selected among the evaluated candidates using the pre-declared robust-neighborhood/stability protocol; it is not claimed to be a global optimum.

## P12 — Full 5,620-sample exact t-SNE run

- Input: `5620 × 64`
- Output: `5620 × 2`
- Perplexity: `40`
- Initial KL: `4.806500153`
- Final/restored KL: `1.081797712`
- Actual iterations: `1000`
- Runtime: `1808.58 s` (`30.14` min)
- Final embedding: `outputs/embeddings/final_optdigits_embedding_p40.csv`

### Direct full-data neighborhood quality

| k | Trustworthiness | Continuity |
|---:|----------------:|-----------:|
| 5 | 0.996845 | 0.992437 |
| 10 | 0.994644 | 0.989400 |
| 20 | 0.991689 | 0.984876 |
| 50 | 0.986279 | 0.977478 |

- Mean Trustworthiness: `0.992364`
- Mean Continuity: `0.986048`
- Worst full-data neighborhood score: `0.977478`
- Artifact: `outputs/tables/final_neighborhood_quality_p40.csv`

## Independent audits

- P10 metrics recomputed from saved embeddings: PASS
- P11 per-seed quality recomputed: PASS
- P11 kNN stability recomputed: PASS
- P12 full-data Trustworthiness/Continuity recomputed from the saved final embedding: PASS
- P12 final KL rebuilt from full `P` and final embedding: PASS
- `pytest`: 4 passed
- package audit: PASS

## Interpretation limits

The 2D visualization is intended for local-neighborhood analysis. Inter-cluster distances, cluster area, and apparent cluster size should not be interpreted as faithful global geometry of the original 64-dimensional space.
