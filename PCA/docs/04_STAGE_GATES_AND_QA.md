# Stage Gates and QA

| Gate | Pass criteria |
|---|---|
| CP1 | 561 features; labels align; finite |
| CP2 | train scaled mean≈0; std≈1; finite |
| CP3 | symmetric covariance; orthogonal eigenvectors; EVR≈1; score covariance near diagonal |
| CP4 | monotonic cumulative EVR; k80/k90/k95/k99 |
| CP5 | 2D/3D plots run; PC variance descending |
| CP6 | MSE finite, non-increasing; all-PC MSE≈0 |
| CP7 | loading eigenvector norm≈1 |
| CP8 | trade-off documented, no optimality overclaim |
| CP9 | final arrays correct shape/finite; PCA90 prefix PCA95 |
| CP10 | all export artifacts exist |

## Stop conditions
Dừng phase nếu:
- test data bị dùng để fit;
- NaN/Inf không giải thích được;
- covariance không symmetric;
- EVR không sum≈1;
- reconstruction error tăng bất thường;
- output contract phase trước bị thay đổi.

Numerical noise cỡ 1e-12/1e-14 có thể chấp nhận tùy metric.
