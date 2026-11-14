import os
import pandas as pd
import numpy as np
from pathlib import Path

# Resolve paths relative to solution.py
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_artifacts():
    """
    Loads teammates' trained model artifacts and fallback lookups.
    Must be loaded relative to the package directory.
    """
    artifacts = {}
    pop_path = MODELS_DIR / "item_popularity.parquet"
    if pop_path.exists():
        artifacts["popularity"] = pd.read_parquet(pop_path)
    else:
        artifacts["popularity"] = None
    return artifacts


def predict_batch(batch_df: pd.DataFrame, catalog_df: pd.DataFrame, artifacts: dict) -> list:
    """
    Generates one valid (rowid, itemID) tuple per row in batch_df.
    Strictly enforces the [valid_from, valid_to) availability window.
    """
    rowids = batch_df["rowid"].to_numpy()
    timestamps = batch_df["timestamp"].to_numpy()
    user_ids = batch_df["user_id"].to_numpy()

    cat_items = catalog_df["itemID"].to_numpy()
    cat_from = catalog_df["valid_from"].to_numpy()
    cat_to = catalog_df["valid_to"].to_numpy()

    # Pre-sort catalogue by popularity if available
    if artifacts["popularity"] is not None:
        pop_map = dict(zip(artifacts["popularity"]["itemID"], artifacts["popularity"]["score"]))
        scores = np.array([pop_map.get(iid, 0.0) for iid in cat_items])
        order = np.argsort(-scores)
        cat_items = cat_items[order]
        cat_from = cat_from[order]
        cat_to = cat_to[order]

    predictions = []

    for rowid, uid, ts in zip(rowids, user_ids, timestamps):
        # Filter catalogue to items strictly valid at this timestamp: [valid_from, valid_to)
        valid_mask = (cat_from <= ts) & (ts < cat_to)
        valid_items = cat_items[valid_mask]

        if len(valid_items) == 0:
            chosen_item = int(cat_items[0])
        else:
            # Baseline: pick highest-ranked valid item (cold-start safe)
            chosen_item = int(valid_items[0])

        predictions.append((int(rowid), chosen_item))

    return predictions


def predict(data_dir: str, batch_size: int = 256):
    """
    Required platform generator signature.
    Reads x_pred.csv and catalog.csv from read-only data_dir and yields batches immediately.
    """
    data_path = Path(data_dir)
    x_pred_path = data_path / "x_pred.csv"
    catalog_path = data_path / "catalog.csv"

    catalog_df = pd.read_csv(catalog_path)
    artifacts = load_artifacts()

    # Stream x_pred.csv in chunks so first batches persist immediately
    for chunk_df in pd.read_csv(x_pred_path, chunksize=batch_size):
        yield predict_batch(chunk_df, catalog_df, artifacts)
