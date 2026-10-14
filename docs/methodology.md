# Methodology

- Use the official training and testing CSVs separately.
- Treat `label` as the binary target: 0=benign, 1=attack.
- Exclude `label`, `attack_cat`, and `id` from model features.
- Fit numeric imputation and categorical encoding within a scikit-learn pipeline on training data only.
- Train a Random Forest classifier with a fixed random seed.
- Report accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC when both classes exist.
- Fit optional Isolation Forest on benign training rows and treat its output as a separate anomaly signal.
