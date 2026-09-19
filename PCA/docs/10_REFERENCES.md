# References and Research Verification

## Official UCI HAR
Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X.  
Human Activity Recognition Using Smartphones.  
UCI Machine Learning Repository.  
DOI: 10.24432/C54S4K

https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones

Verified facts used:
- 30 volunteers
- 6 activities
- 10,299 records
- 561-feature vectors
- accelerometer + gyroscope
- predefined train/test partition

## NumPy
`numpy.linalg.eigh`

https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html

Reason:
- covariance is symmetric;
- `eigh` is for symmetric/Hermitian matrices;
- eigenvectors are returned by columns;
- eigenvalues are ascending and must be sorted descending for PCA ranking.

## Introductory paper
D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz.  
“A Public Domain Dataset for Human Activity Recognition Using Smartphones.” ESANN, 2013.
