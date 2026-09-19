# 15 — Reproduction Profiles

The package separates correctness verification from expensive empirical reproduction.

## `quick`
Purpose: verify the package/core implementation rapidly.

Covers:
- clean-data contract;
- core mathematical smoke tests;
- tiny exact t-SNE fit;
- P00–P08 when the notebook quick profile is executed.

Recommended command:

```bash
python scripts/run_case_study.py --profile quick --output-root rerun_outputs
```

## `screening`
Purpose: reproduce P09–P10 without launching the full 5,620-sample optimization.

Covers:
- deterministic 1,000-sample stratified screen;
- perplexities `{5,10,20,30,40,50}`;
- fixed optimizer/seed comparison;
- Trustworthiness and Continuity at `k={5,10,20,50}`;
- Pareto shortlist for P11.

Typical runtime is much shorter than the full profile, but hardware-dependent.

## `full`
Purpose: reproduce the complete empirical study.

Covers:
- P09 screening;
- P10 neighborhood evaluation;
- P11 multi-seed robustness with seeds `{0,42,123}`;
- primary/alternative selection;
- P12 exact full-data run on all 5,620 samples.

Verified reference selection:
- Primary: `40`
- Alternative: `30`

Verified full-run reference:
- `5620 × 64 → 5620 × 2`
- 1000 iterations
- final/restored KL `1.081797712299418`
- reference runtime `1808.58 s` on the recorded environment.

## Why outputs go to `rerun_outputs/`

The bundled `outputs/` directory is treated as immutable reference evidence. Reruns write to `rerun_outputs/` so a portability test cannot accidentally destroy the verified final artifacts.
