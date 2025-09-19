#!/usr/bin/env python3
"""
visualize.py

Creates a set of diagnostic plots from:
 - main/models/predictions_test_regression.csv  (must contain y_true, y_pred)
 - main/models/feature_importances.csv         (optional)
 - or loads model_regressor.joblib to compute SHAP (optional)

Outputs saved to main/visuals/

Install extras:
pip install matplotlib seaborn pandas numpy joblib shap scikit-learn

Run:
source main/.venv/bin/activate
python main/visualize.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

PRED_CSV = Path("../main/main/models/predictions_test_regression.csv")
FI_CSV = Path("../main/main/models/feature_importances.csv")
MODEL_FILE = Path("../main/main/models/model_regressor.joblib")
OUTDIR = Path("../main/visuals")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_preds():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {PRED_CSV}")
    df = pd.read_csv(PRED_CSV)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("Predictions CSV must contain columns 'y_true' and 'y_pred'")
    return df

def plot_true_vs_pred(df):
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    plt.figure(figsize=(7,6))
    sns.kdeplot(x=y_true, y=y_pred, fill=True, cmap="Blues", thresh=0.05, levels=10)
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.xlabel("True risk score")
    plt.ylabel("Predicted risk score")
    plt.title("True vs Predicted (with density)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "true_vs_pred_density.png", dpi=200)
    plt.close()

def plot_residuals_hist(df):
    residuals = df["y_true"] - df["y_pred"]
    plt.figure(figsize=(7,4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.xlabel("Residual (true - pred)")
    plt.title("Residuals distribution")
    plt.tight_layout()
    plt.savefig(OUTDIR / "residuals_hist.png", dpi=200)
    plt.close()

def plot_residuals_vs_pred(df):
    residuals = df["y_true"] - df["y_pred"]
    plt.figure(figsize=(7,4))
    sns.scatterplot(x=df["y_pred"], y=residuals, s=18, alpha=0.6)
    # add smoothed trend
    try:
        sns.regplot(x=df["y_pred"], y=residuals, lowess=True, scatter=False, line_kws={"color":"red"})
    except Exception:
        pass
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (true - pred)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(OUTDIR / "residuals_vs_pred.png", dpi=200)
    plt.close()

def plot_error_buckets(df):
    abs_err = np.abs(df["y_true"] - df["y_pred"])
    thresholds = [5,10,15,20]
    stats = {f"<= {t}": (abs_err <= t).mean()*100 for t in thresholds}
    # bar chart
    plt.figure(figsize=(6,3))
    sns.barplot(x=list(stats.keys()), y=list(stats.values()))
    plt.ylabel("Percent of samples (%)")
    plt.title("Percent within absolute error thresholds")
    plt.tight_layout()
    plt.savefig(OUTDIR / "error_buckets.png", dpi=200)
    plt.close()

def plot_residuals_by_group(df, group_col="on_birth_control"):
    if group_col not in df.columns:
        print(f"Group column '{group_col}' not in predictions CSV — skipping group plot.")
        return
    df_plot = df.copy()
    df_plot["residual"] = df_plot["y_true"] - df_plot["y_pred"]
    plt.figure(figsize=(6,4))
    sns.boxplot(x=group_col, y="residual", data=df_plot)
    plt.xlabel(group_col)
    plt.ylabel("Residual (true - pred)")
    plt.title(f"Residuals by {group_col}")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"residuals_by_{group_col}.png", dpi=200)
    plt.close()

def plot_feature_importances():
    if not FI_CSV.exists():
        print("Feature importances file not found — skipping feature importance plot.")
        return
    fi = pd.read_csv(FI_CSV).sort_values("importance", ascending=False).head(30)
    plt.figure(figsize=(8,6))
    sns.barplot(x="importance", y="feature", data=fi, palette="viridis")
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(OUTDIR / "feature_importances_top30.png", dpi=200)
    plt.close()

def print_and_save_summary(df):
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = mean_squared_error(y_true, y_pred, squared=False) if hasattr(__import__("sklearn.metrics"), "mean_squared_error") else (mean_squared_error(y_true,y_pred)**0.5)
    summary = {
        "samples": len(y_true),
        "r2": r2,
        "mae": mae,
        "rmse": rmse_val
    }
    txt = "\n".join([f"{k}: {v:.4f}" for k,v in summary.items()])
    (OUTDIR / "summary.txt").write_text(txt)
    print("Evaluation summary:\n", txt)

# optional SHAP plots (requires model file and shap installed)
def shap_plots(df):
    try:
        import shap
    except Exception:
        print("shap not installed; skipping SHAP plots. To enable: pip install shap")
        return
    if not MODEL_FILE.exists():
        print("Model file not found; skipping SHAP plots.")
        return
    model = joblib.load(MODEL_FILE)
    # we need the preprocessed X used in predictions CSV — best-effort: use columns except y_true,y_pred
    X = df.drop(columns=[c for c in ["y_true","y_pred","y_prob"] if c in df.columns])
    # attempt to compute SHAP (may be slow)
    explainer = shap.Explainer(model.named_steps["reg"] if "reg" in model.named_steps else model.named_steps.get("clf"), model.named_steps["pre"].transform(X))
    shap_values = explainer(model.named_steps["pre"].transform(X))
    # summary
    shap.summary_plot(shap_values, feature_names=explainer.feature_names_in, show=False)
    plt.tight_layout()
    plt.savefig(OUTDIR / "shap_summary.png", dpi=200)
    plt.close()
    # waterfall for top sample
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(OUTDIR / "shap_waterfall_example.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    df = load_preds()
    # Ensure numeric y_true/y_pred
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    plot_true_vs_pred(df)
    plot_residuals_hist(df)
    plot_residuals_vs_pred(df)
    plot_error_buckets(df)
    plot_residuals_by_group(df, group_col="on_birth_control")
    plot_feature_importances()
    print("Saved visuals to:", OUTDIR)
    # optional shap (commented out by default)
    # shap_plots(df)
