
"""Data checks for the Phase 2 validation dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def check_feedback(feedback):
    required = {
        "event_id", "user_id", "timestamp", "itemID", "clicked"
    }
    missing = required - set(feedback.columns)
    if missing:
        raise ValueError(f"Missing feedback columns: {sorted(missing)}")

    if feedback.empty:
        raise ValueError("Feedback is empty")

    if feedback[list(required)].isna().any().any():
        raise ValueError("Feedback contains missing required values")

    if not feedback["clicked"].isin([0, 1]).all():
        raise ValueError("Feedback contains invalid click labels")

    users = feedback.groupby("event_id")["user_id"].nunique()
    times = feedback.groupby("event_id")["timestamp"].nunique()

    if (users > 1).any() or (times > 1).any():
        raise ValueError("Inconsistent user or timestamp within events")

    if feedback.duplicated(["event_id", "itemID"]).any():
        raise ValueError("Duplicate items within feedback events")

    return {
        "rows": len(feedback),
        "events": feedback["event_id"].nunique(),
        "users": feedback["user_id"].nunique(),
        "items": feedback["itemID"].nunique(),
    }


def check_catalog(catalog):
    required = {"itemID", "valid_from", "valid_to"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Missing catalog columns: {sorted(missing)}")

    if catalog.empty:
        raise ValueError("Catalog is empty")

    if catalog[list(required)].isna().any().any():
        raise ValueError("Catalog contains missing required values")

    if catalog["itemID"].duplicated().any():
        raise ValueError("Duplicate catalog item IDs")

    if (catalog["valid_from"] > catalog["valid_to"]).any():
        raise ValueError("Invalid catalog availability windows")

    return {"catalog_items": len(catalog)}


def event_summary(feedback):
    events = feedback.groupby("event_id").agg(
        clicks=("clicked", "sum"),
        items=("itemID", "size"),
    )

    total = len(events)
    single_click = int((events["clicks"] == 1).sum())
    no_click = int((events["clicks"] == 0).sum())

    return {
        "total_events": total,
        "single_click_events": single_click,
        "no_click_events": no_click,
        "multi_click_events": int((events["clicks"] > 1).sum()),
        "ranking_coverage": single_click / total if total else 0.0,
        "mean_displayed_items": float(events["items"].mean()),
    }


def catalog_coverage(feedback, catalog):
    """Measure overlap, not historical recommendation validity."""
    catalog_items = set(catalog["itemID"])
    observed_items = set(feedback["itemID"])

    return {
        "historical_items": len(observed_items),
        "historical_items_in_round_catalog": len(
            observed_items & catalog_items
        ),
        "historical_items_outside_round_catalog": len(
            observed_items - catalog_items
        ),
    }


def find_round_file(data_dir, filename):
    root = Path(data_dir).resolve()
    candidates = [
        root / filename,
        root.parent / filename,
        root.parent / "phase-2-round-1-input" / filename,
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


def request_summary(data_dir):
    path = find_round_file(data_dir, "x_pred.csv")
    if path is None:
        return {"status": "x_pred.csv not found"}

    requests = pd.read_csv(path, dtype={"rowid": str, "user_id": str})
    required = {"rowid", "user_id", "timestamp"}
    missing = required - set(requests.columns)

    if missing:
        return {"error": f"Missing request columns: {sorted(missing)}"}

    timestamps = pd.to_numeric(
        requests["timestamp"], errors="coerce"
    )

    return {
        "requests": len(requests),
        "unique_rowids": requests["rowid"].nunique(),
        "duplicate_rowids": int(
            requests["rowid"].duplicated().sum()
        ),
        "missing_values": int(
            requests[list(required)].isna().sum().sum()
        ),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "first_timestamp": (
            int(timestamps.min())
            if timestamps.notna().any() else None
        ),
        "last_timestamp": (
            int(timestamps.max())
            if timestamps.notna().any() else None
        ),
    }


def config_summary(data_dir):
    path = find_round_file(data_dir, "round_config.json")
    if path is None:
        return {"status": "round_config.json not found"}

    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        return {"status": "Configuration is not a JSON object"}

    return {
        "status": "loaded",
        "top_level_keys": sorted(config.keys()),
    }


def print_diagnostics(feedback, catalog, data_dir):
    sections = {
        "Feedback": check_feedback(feedback),
        "Catalog": check_catalog(catalog),
        "Events": event_summary(feedback),
        "Round catalog coverage": catalog_coverage(
            feedback, catalog
        ),
        "Prediction requests": request_summary(data_dir),
        "Round configuration": config_summary(data_dir),
    }

    for title, values in sections.items():
        print(f"\n{title}")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    return sections


def check_predictions(
    predictions, requests, catalog, end_inclusive=False
):
    """Check a completed prediction table against Round 1 requests."""
    required = {"rowid", "itemID"}
    if not required.issubset(predictions.columns):
        raise ValueError("Predictions need rowid and itemID columns")

    if not {"rowid", "timestamp"}.issubset(requests.columns):
        raise ValueError("Requests need rowid and timestamp columns")

    if predictions[["rowid", "itemID"]].isna().any().any():
        raise ValueError("Predictions contain missing IDs")

    if predictions["rowid"].duplicated().any():
        raise ValueError("Duplicate prediction rowids")

    if requests["rowid"].duplicated().any():
        raise ValueError("Duplicate request rowids")

    expected = set(requests["rowid"].astype(str))
    actual = set(predictions["rowid"].astype(str))

    if expected != actual:
        raise ValueError(
            f"Missing rowids: {len(expected - actual)}, "
            f"unexpected rowids: {len(actual - expected)}"
        )

    if catalog["itemID"].duplicated().any():
        raise ValueError("Duplicate catalog item IDs")

    merged = predictions.merge(
        requests[["rowid", "timestamp"]],
        on="rowid",
        how="left",
        validate="one_to_one",
    ).merge(
        catalog[["itemID", "valid_from", "valid_to"]],
        on="itemID",
        how="left",
        validate="many_to_one",
    )

    timestamps = pd.to_numeric(
        merged["timestamp"], errors="coerce"
    )
    starts = pd.to_numeric(
        merged["valid_from"], errors="coerce"
    )
    ends = pd.to_numeric(
        merged["valid_to"], errors="coerce"
    )

    if end_inclusive:
        valid_end = timestamps <= ends
    else:
        valid_end = timestamps < ends

    available = (timestamps >= starts) & valid_end
    available = available.fillna(False)

    return {
        "total_predictions": len(predictions),
        "valid_predictions": int(available.sum()),
        "invalid_predictions": int((~available).sum()),
        "availability_rate": float(available.mean()),
    }
