# 13 — Legacy Project Audit

The previous project is retained only as historical reference outside this package. The rebuilt package intentionally excludes legacy experiment outputs because the prior processed CSV exported the DataFrame index and an old loader could treat that `Unnamed: 0` column as a feature. The rebuilt dataset is explicitly selected by 64 Pixel columns and saved with `index=False`.

Also removed from the scientific selection logic: hard-coded/inconsistent perplexity labels and using final KL as the main cross-perplexity ranking metric.
