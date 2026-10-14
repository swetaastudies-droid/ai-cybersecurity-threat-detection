# Architecture

1. **Data loading** validates the official training/testing CSVs.
2. **Cleaning** normalizes columns and replaces numeric infinities with missing values.
3. **Feature pipeline** separates the label and fits imputation/encoding only on training data.
4. **Classifier** uses a Random Forest pipeline to predict benign versus attack.
5. **Anomaly detector** optionally fits Isolation Forest on benign training records only.
6. **Evaluation** writes metrics and predictions for the held-out test data.
7. **Dashboard** visualizes saved benchmark results and threshold-based review flags.

The initial application is offline and dataset-based. It is not connected to a live network, SIEM, or SOC.
