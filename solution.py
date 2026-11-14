import os
import pandas as pd
import numpy as np
from pathlib import Path

# Resolve paths relative to solution.py
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"


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


def predict(data_dir, batch_size):
    """
    Main entry point called by the evaluation harness.
    Must be a generator yielding batches of (rowid, itemID) tuples.
    """
    data = Path(data_dir)
    rows = pd.read_csv(data / "x_pred.csv", dtype=str)
    catalog = pd.read_csv(data / "catalog.csv", dtype=str)
    
    # Ensure proper numeric typing for timestamps and availability windows
    rows["timestamp"] = pd.to_numeric(rows["timestamp"], errors="coerce")
    catalog["valid_from"] = pd.to_numeric(catalog["valid_from"], errors="coerce")
    catalog["valid_to"] = pd.to_numeric(catalog["valid_to"], errors="coerce")

    cat_items = catalog["itemID"].to_numpy()
    cat_from = catalog["valid_from"].to_numpy()
    cat_to = catalog["valid_to"].to_numpy()

    default_item = cat_items[0] if len(cat_items) > 0 else "itm_default"

    artifacts = load_artifacts()

    # Process rows in chunks according to batch_size and yield iteratively
    for first in range(0, len(rows), batch_size):
        part = rows.iloc[first:first + batch_size]
        batch = []
        for rowid, timestamp in zip(part["rowid"], part["timestamp"]):
            # Strictly enforce the [valid_from, valid_to) availability window[cite: 2, 6]
            valid_mask = (cat_from <= timestamp) & (timestamp < cat_to)
            valid_items = cat_items[valid_mask]

            if len(valid_items) > 0:
                chosen_item = valid_items[0]
            else:
                chosen_item = default_item

            batch.append((str(rowid), str(chosen_item)))
        
        yield batch
