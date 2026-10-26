"""Shared helpers used by all notebooks: data loading, price classes, evaluation and plotting."""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

LAB_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = LAB_DIR.parent / "data" / "AmesHousing.csv"
PROCESSED_DIR = LAB_DIR / "processed_data"
RESULTS_DIR = LAB_DIR / "results"

RANDOM_STATE = 42

TARGET = "PriceClass"

# Class boundaries from the lab requirements; each class includes its lower bound.
PRICE_BINS = [-np.inf, 100_000, 130_000, 180_000, 250_000, 400_000, np.inf]
CLASS_LABELS = {
    1: "< 100k",
    2: "100k - 130k",
    3: "130k - 180k",
    4: "180k - 250k",
    5: "250k - 400k",
    6: ">= 400k",
}


def load_raw() -> pd.DataFrame:
    """Load the raw Ames Housing dataset (the file is tab-separated)."""
    return pd.read_csv(RAW_DATA_PATH, sep="\t")


def add_price_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add the PriceClass target (1-6) and drop SalePrice, which would leak the target."""
    df = df.copy()
    df[TARGET] = pd.cut(df["SalePrice"], bins=PRICE_BINS, labels=list(CLASS_LABELS), right=False).astype(int)
    return df.drop(columns="SalePrice")


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a processed dataset to processed_data/<name>.csv and return its path."""
    path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_processed(name: str) -> pd.DataFrame:
    """Load processed_data/<name>.csv; text columns become pandas categories (a CSV cannot store that type)."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    text_cols = df.select_dtypes(exclude="number").columns
    df[text_cols] = df[text_cols].astype("category")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y; y is shifted from classes 1-6 to 0-5 because XGBoost expects labels starting at 0."""
    return df.drop(columns=TARGET), df[TARGET] - 1


# ---------------------------------------------------------------------------
# Training helpers: steps that must only ever see the training data
# ---------------------------------------------------------------------------

OUTLIER_LIVING_AREA = 4000


def drop_outliers(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Remove houses with Gr Liv Area > 4000 (recommended by the dataset's author). Training data only."""
    keep = X["Gr Liv Area"] <= OUTLIER_LIVING_AREA
    return X[keep], y[keep]


def add_smote_samples(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority classes with SMOTE (SMOTENC when there are categorical columns).

    SMOTE cannot work with NaN, so it runs on an imputed copy. Only the *new* synthetic rows are kept and appended
    to the original training rows, which keep their NaNs for XGBoost.
    """
    cat_cols = X.select_dtypes("category").columns.tolist()
    num_cols = X.columns.difference(cat_cols)
    filled = X.copy()
    filled[num_cols] = filled[num_cols].fillna(filled[num_cols].median())
    for col in cat_cols:
        filled[col] = filled[col].cat.add_categories("Missing").fillna("Missing")

    if cat_cols:
        sampler = SMOTENC(categorical_features=cat_cols, random_state=RANDOM_STATE)
    else:
        sampler = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(filled, y)

    # imblearn returns the original rows first, followed by the synthetic ones
    X_new, y_new = X_res.iloc[len(X):].copy(), y_res.iloc[len(X):]
    for col in cat_cols:
        X_new[col] = pd.Categorical(X_new[col].replace("Missing", np.nan), categories=X[col].cat.categories)
    X_new = X_new[X.columns].astype(X.dtypes.to_dict())
    return pd.concat([X, X_new], ignore_index=True), pd.concat([y, y_new], ignore_index=True)


def fit_model(model, X: pd.DataFrame, y: pd.Series, remove_outliers: bool = False, balance: str | None = None):
    """Fit a model, optionally removing outliers and handling class imbalance on the training data.

    balance: None, "class_weight" (weights inversely proportional to class frequency) or "smote".
    """
    if remove_outliers:
        X, y = drop_outliers(X, y)
    sample_weight = None
    if balance == "class_weight":
        sample_weight = compute_sample_weight("balanced", y)
    elif balance == "smote":
        X, y = add_smote_samples(X, y)
    elif balance is not None:
        raise ValueError(f"unknown balance method: {balance}")
    return model.fit(X, y, sample_weight=sample_weight)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_scores(y_true, y_pred) -> dict:
    """Accuracy and macro-averaged precision, recall and F1 (every class counts equally)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def cross_validate(make_model, X: pd.DataFrame, y: pd.Series, cv, **fit_kwargs) -> pd.DataFrame:
    """k-fold CV; returns one row per fold with the train and validation scores.

    make_model: function returning a new, unfitted model (a fresh one is needed for every fold).
    fit_kwargs: passed to fit_model (remove_outliers, balance) and applied to the training folds only.
    """
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = fit_model(make_model(), X_tr, y_tr, **fit_kwargs)
        train_scores = compute_scores(y_tr, model.predict(X_tr))
        val_scores = compute_scores(y_val, model.predict(X_val))
        rows.append({"fold": fold,
                     **{f"train_{k}": v for k, v in train_scores.items()},
                     **{f"val_{k}": v for k, v in val_scores.items()}})
    return pd.DataFrame(rows)
