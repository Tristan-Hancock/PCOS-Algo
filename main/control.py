#!/usr/bin/env python3
"""
train_model.py

Usage:
  python train_model.py --csv data/simulated_pcos.csv --target pcos_risk_score --outdir models

Notes:
- Activate your .venv first.
- Installs required: pandas, numpy, scikit-learn, matplotlib, joblib
"""

import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

# ---------------------------
# Utility functions
# ---------------------------
def compute_bmi(row, weight_col="weight_lbs", ft_col="height_ft", in_col="height_in"):
    # weight in lbs, height in ft + in
    try:
        w = float(row.get(weight_col, np.nan))
        ft = float(row.get(ft_col, 0))
        inch = float(row.get(in_col, 0))
        h_in = ft * 12.0 + inch
        if np.isnan(w) or h_in <= 0:
            return np.nan
        bmi = (w / (h_in ** 2)) * 703.0
        return bmi
    except Exception:
        return np.nan

def add_feature_engineering(df):
    # BMI
    df["bmi"] = df.apply(compute_bmi, axis=1)
    # Example: symptom count (sum of known binary symptom cols if they exist)
    symptom_cols = [c for c in df.columns if c.lower() in (
        "excess_hair", "acne", "skin_darkening", "missed_periods", "long_cycles", "less_than_9_periods")]
    if symptom_cols:
        df["symptom_count"] = df[symptom_cols].sum(axis=1)
    else:
        df["symptom_count"] = 0
    # Ensure fat distribution col exists
    if "hair_pattern" in df.columns:
        # rename to fat_dist if original name differs
        df["fat_dist"] = df["hair_pattern"]
    elif "fat_distribution" in df.columns:
        df["fat_dist"] = df["fat_distribution"]
    else:
        # If not present, create a placeholder category
        df["fat_dist"] = "unknown"
    return df

# ---------------------------
# Main
# ---------------------------
def main(csv_path, target_col, outdir, test_size=0.2, random_state=42):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Initial shape:", df.shape)

    # Basic cleaning: strip column names
    df.columns = [c.strip() for c in df.columns]

    # Add engineered features
    df = add_feature_engineering(df)

    # Decide whether we're doing regression (continuous target) or classification (binary)
    is_classification = False
    if target_col in df.columns:
        # if column is binary 0/1 or only contains 0/1 values -> classification
        unique_vals = pd.Series(df[target_col].dropna().unique())
        if set(unique_vals.tolist()) <= {0, 1}:
            is_classification = True
    else:
        raise ValueError(f"Target column '{target_col}' not found in CSV columns: {df.columns.tolist()}")

    # Define feature columns to use (drop target and obvious non-features)
    drop_cols = [target_col]
    # Drop PII-like columns if present
    for c in ["id", "name", "email"]:
        if c in df.columns:
            drop_cols.append(c)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"Using {len(feature_cols)} features. Classification? {is_classification}")
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # keep 'bmi' numeric even if dtype object
    if "bmi" in X.columns and "bmi" not in numeric_cols:
        numeric_cols.append("bmi")
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop",
        sparse_threshold=0
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=(y if is_classification else None))
    print("Train / test sizes:", X_train.shape, X_test.shape)

    # Choose models
    if is_classification:
        # baseline classifier → RandomForestClassifier; output probability later
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
        print("Training RandomForestClassifier...")
        pipeline.fit(X_train, y_train)
        # predict probabilities for positive class
        probs = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        print(f"ROC AUC: {auc:.4f}   Brier: {brier:.4f}")
        # Save model
        joblib.dump(pipeline, outdir / "model_classifier.joblib")
        # Save test predictions
        preds = pipeline.predict(X_test)
        result_df = X_test.copy()
        result_df["y_true"] = y_test.values
        result_df["y_pred"] = preds
        result_df["y_prob"] = probs
        result_df.to_csv(outdir / "predictions_test_classification.csv", index=False)
    else:
        # regression
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        pipeline = Pipeline(steps=[("pre", preprocessor), ("reg", model)])
        print("Training RandomForestRegressor...")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        print(f"R2: {r2:.4f}   MAE: {mae:.4f}   RMSE: {rmse:.4f}")
        # Save model
        joblib.dump(pipeline, outdir / "model_regressor.joblib")
        # Save test predictions
        result_df = X_test.copy()
        result_df["y_true"] = y_test.values
        result_df["y_pred"] = preds
        result_df.to_csv(outdir / "predictions_test_regression.csv", index=False)

    # Save a note about features (get feature names after preprocessing)
    try:
        pre = pipeline.named_steps["pre"]
        # derive column names after onehot
        ohe_cols = []
        if categorical_cols:
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            categories = ohe.categories_
            for col, cats in zip(categorical_cols, categories):
                ohe_cols.extend([f"{col}__{c}" for c in cats])
        feature_names = numeric_cols + ohe_cols
        with open(outdir / "feature_names.txt", "w") as f:
            f.write("\n".join(feature_names))
    except Exception as e:
        print("Could not extract feature names:", e)

    # Basic feature importance (only for tree models)
    try:
        if is_classification:
            raw_model = pipeline.named_steps["clf"]
        else:
            raw_model = pipeline.named_steps["reg"]
        if hasattr(raw_model, "feature_importances_"):
            importances = raw_model.feature_importances_
            # align with feature_names if available
            if 'feature_names' in locals():
                fnames = feature_names
            else:
                fnames = [f"f{i}" for i in range(len(importances))]
            imp_df = pd.DataFrame({"feature": fnames, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
            imp_df.to_csv(outdir / "feature_importances.csv", index=False)
            print("Saved feature importances to", outdir / "feature_importances.csv")
    except Exception as e:
        print("Feature importance step skipped:", e)

    # Quick residual / calibration plot for regression
    if not is_classification:
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, preds, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("True target")
        plt.ylabel("Predicted")
        plt.title("True vs Predicted (regression)")
        plt.tight_layout()
        plt.savefig(outdir / "true_vs_pred.png", dpi=150)
        print("Saved true_vs_pred plot to", outdir / "true_vs_pred.png")

    print("All done. Models and outputs are in:", outdir)

# ---------------------------
# Argparse CLI
# ---------------------------
# ---------------------------
# Argparse CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../data/simdata.csv",   # 👈 hard-coded default path
                   help="Path to CSV file (default: data/simdata.csv)")
    p.add_argument("--target", default="pcos_risk_score",
                   help="Target column name (default: pcos_risk_score)")
    p.add_argument("--outdir", default="main/models",
                   help="Output directory to save models/artefacts")
    args = p.parse_args()
    main(args.csv, args.target, args.outdir)

