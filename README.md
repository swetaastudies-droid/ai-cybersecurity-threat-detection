# AI-Powered Cybersecurity Threat Detection System

An educational network intrusion detection prototype using the UNSW-NB15 public benchmark, a Random Forest classifier, optional Isolation Forest anomaly detection, and a Streamlit dashboard.

## Scope and safety

This project analyzes a public labeled dataset. It does **not** monitor live networks or prove real-world compromise. Model scores are not calibrated real-world incident probabilities. Alerts are for analyst review only.

## Dataset

Download the UNSW-NB15 training and testing CSV files from the [official UNSW research page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them in:

- `data/raw/UNSW_NB15_training-set.csv`
- `data/raw/UNSW_NB15_testing-set.csv`

Check the dataset terms before redistributing it. The raw data is excluded from Git.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the pipeline

```bash
python -m src.data_loader
python train_model.py
python evaluate_model.py
python -m streamlit run app.py
```

Optional CLI prediction on a CSV with the required model feature columns:
```bash
python -m src.detection path/to/input.csv --output outputs/new_predictions.csv
```

Run automated tests:
```bash
python -m pytest -v
```

## Outputs

- `models/threat_detection_model.joblib`: fitted preprocessing + classifier.
- `models/anomaly_detection_model.joblib`: optional Isolation Forest.
- `outputs/metrics.json`: measured evaluation metrics.
- `outputs/predictions.csv`: held-out test predictions.

Metrics are generated when you run the evaluation; do not invent or pre-fill them.

## Model notes and limitations

- `label=0` means benign and `label=1` means attack in this binary task.
- `attack_cat` and `id` are excluded from the feature set to reduce leakage/identifier issues.
- Preprocessing is fitted on training data and reused for testing.
- Isolation Forest is a separate anomaly signal and does not produce attack probabilities.
- Benchmark performance does not establish performance on current production traffic.
- Validate data licensing, model behavior, thresholds, false positives and false negatives before any real-world use.


## Virtual simulation

After training and evaluation, run:

```bash
python run_simulation.py
```

This processes up to 2,000 held-out benchmark records with both the classifier and Isolation Forest, writes a combined CSV, generates synthetic event metadata separately, and saves two diagnostic graphs under `outputs/`. The synthetic event metadata is fictional and is not passed into the model because it does not contain the network-flow features expected by the trained pipeline.
