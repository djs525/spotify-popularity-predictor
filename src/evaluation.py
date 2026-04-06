"""
evaluation.py
-------------
Metrics calculation and all plotting utilities for the paper.
Every figure is saved to the figures/ directory as a high-resolution PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Consistent style across all figures
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"  {model_name:25s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")
    return metrics


def metrics_table(results: list[dict]):
    """Print a pretty summary table from a list of metric dicts."""
    print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 55)
    for r in results:
        print(f"  {r['model']:<23} {r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['R2']:>8.3f}")


# ---------------------------------------------------------------------------
# 2. EDA plots
# ---------------------------------------------------------------------------

def plot_popularity_distribution(y, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Popularity score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Spotify track popularity")
    fig.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/01_popularity_distribution.png")
        print(f"  Saved 01_popularity_distribution.png")
    return fig


def plot_correlation_heatmap(df, feature_cols: list, save: bool = True):
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Pearson correlation of audio features")
    fig.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/02_correlation_heatmap.png")
        print(f"  Saved 02_correlation_heatmap.png")
    return fig


def plot_genre_popularity(genre_means, top_n: int = 20, save: bool = True):
    top = genre_means.sort_values(ascending=False).head(top_n)
    bot = genre_means.sort_values(ascending=True).head(top_n)
    combined = (
        top[::-1].to_frame("mean_popularity")
        .assign(color="#4C72B0")
        ._append(
            bot.to_frame("mean_popularity").assign(color="#DD8452")
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, data, title in zip(
        axes,
        [top[::-1], bot],
        [f"Top {top_n} genres by mean popularity", f"Bottom {top_n} genres"]
    ):
        ax.barh(data.index, data.values, color="#4C72B0" if "Top" in title else "#DD8452")
        ax.set_xlabel("Mean popularity")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/03_genre_popularity.png")
        print(f"  Saved 03_genre_popularity.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Model result plots
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(y_true, y_pred, model_name: str = "LightGBM",
                              save: bool = True):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, color="#4C72B0", rasterized=True)
    lims = [0, 100]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual popularity")
    ax.set_ylabel("Predicted popularity")
    ax.set_title(f"{model_name}: predicted vs actual")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save:
        fname = f"04_pred_vs_actual_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(f"{FIGURES_DIR}/{fname}")
        print(f"  Saved {fname}")
    return fig


def plot_residuals(y_true, y_pred, model_name: str = "LightGBM", save: bool = True):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.15, s=8, color="#4C72B0", rasterized=True)
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted popularity")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(f"{model_name}: residual plot")
    fig.tight_layout()
    if save:
        fname = f"05_residuals_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(f"{FIGURES_DIR}/{fname}")
        print(f"  Saved {fname}")
    return fig


def plot_model_comparison(results: list[dict], save: bool = True):
    """Grouped bar chart comparing RMSE and MAE across all models."""
    models = [r["model"] for r in results]
    rmse_vals = [r["RMSE"] for r in results]
    mae_vals = [r["MAE"] for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, rmse_vals, width, label="RMSE", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, mae_vals, width, label="MAE", color="#DD8452")

    ax.set_ylabel("Error")
    ax.set_title("Model comparison: RMSE and MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/06_model_comparison.png")
        print(f"  Saved 06_model_comparison.png")
    return fig


def plot_learning_curves(history: dict, save: bool = True):
    """Neural network train/val RMSE over epochs."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train_rmse"]) + 1)
    ax.plot(epochs, history["train_rmse"], label="Train RMSE", color="#4C72B0")
    ax.plot(epochs, history["val_rmse"], label="Val RMSE", color="#DD8452", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Neural network learning curves")
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/07_nn_learning_curves.png")
        print(f"  Saved 07_nn_learning_curves.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Feature importance (LightGBM built-in)
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             save: bool = True):
    import lightgbm as lgb
    importance = model.booster_.feature_importance(importance_type="gain")
    fi = (
        dict(zip(feature_names, importance))
    )
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*sorted_fi)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names[::-1], vals[::-1], color="#4C72B0")
    ax.set_xlabel("Gain-based importance")
    ax.set_title(f"LightGBM top {top_n} features (gain)")
    fig.tight_layout()
    if save:
        fig.savefig(f"{FIGURES_DIR}/08_feature_importance_lgbm.png")
        print(f"  Saved 08_feature_importance_lgbm.png")
    return fig