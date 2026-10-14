"""Create leakage-aware features and reusable preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "label"
# attack_cat is closely related to the target and id is an identifier, not a traffic feature.
EXCLUDED_COLUMNS = {"label", "attack_cat", "id"}


def split_features_target(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Return feature frames, labels, preprocessing pipeline, and feature names."""
    if TARGET not in train_df or TARGET not in test_df:
        raise ValueError("Both train and test datasets must contain 'label'.")

    # Use training column order, only retaining columns present in both datasets.
    feature_columns = [
        c for c in train_df.columns
        if c not in EXCLUDED_COLUMNS and c in test_df.columns
    ]
    if not feature_columns:
        raise ValueError("No eligible feature columns were found.")

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # Convert columns to numeric only when every non-missing value converts cleanly
    # in both datasets. Otherwise, treat the column as categorical.
    for col in feature_columns:
        train_num = pd.to_numeric(X_train[col], errors="coerce")
        test_num = pd.to_numeric(X_test[col], errors="coerce")
        train_valid = train_num.notna().sum() == X_train[col].notna().sum()
        test_valid = test_num.notna().sum() == X_test[col].notna().sum()
        if train_valid and test_valid:
            X_train[col] = train_num
            X_test[col] = test_num

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    return X_train, X_test, y_train, y_test, preprocessor, feature_columns
