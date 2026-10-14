# Model evaluation

Run:

```bash
python train_model.py
python evaluate_model.py
```

Actual measured results are written to `outputs/metrics.json`. Interpret precision, recall, F1-score, confusion matrix and ROC-AUC together. Do not report placeholder or invented results. Keep the held-out test set for final evaluation and avoid repeated tuning against it.
