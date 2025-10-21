from pathlib import Path
import os

BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parents[1]))
DATA_FILE = BASE_DIR / "data/dataset/synthetic_demand.csv"
MODEL_DIR = BASE_DIR / "data/models"

# MLFlow
MLFLOW_TRACKING_URI = f"file:{BASE_DIR}/data/mlruns"
MLFLOW_EXPERIMENT = "synthetic_demand_forecast"

# Drift Threshold
THRESHOLD_MAE = 10
