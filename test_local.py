import shutil
import time
from pathlib import Path
import pandas as pd
from solution import predict

MOCK_DIR = Path("mock_round_data")

def setup_mock_data():
    if MOCK_DIR.exists():
        shutil.rmtree(MOCK_DIR)
    MOCK_DIR.mkdir(parents=True)

    catalog = pd.DataFrame({
        "itemID": [101, 102, 103, 104],
        "valid_from": [1000, 1500, 1000, 2000],
        "valid_to":   [1600, 2500, 3000, 3000],
    })
    catalog.to_csv(MOCK_DIR / "catalog.csv", index=False)

    x_pred = pd.DataFrame({
        "rowid": list(range(1, 101)),
        "user_id": [f"user_{i}" for i in range(1, 101)],
        "timestamp": [1200 if i % 2 == 0 else 1800 for i in range(1, 101)],
    })
    x_pred.to_csv(MOCK_DIR / "x_pred.csv", index=False)

def run_verification():
    setup_mock_data()
    print("Running local verification...")
    
    total = 0
    for batch in predict(str(MOCK_DIR), batch_size=10):
        total += len(batch)
        print(f"Yielded batch of size {len(batch)}")

    print(f"Success! Generated {total} predictions.")
    shutil.rmtree(MOCK_DIR)

if __name__ == "__main__":
    run_verification()
