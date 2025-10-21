import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_model(model, model_dir: Path, version: str):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_v{version}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")
    return model_path


def load_latest_model(model_dir: Path):
    models = sorted(model_dir.glob("model_v*.pkl"))
    if not models:
        raise FileNotFoundError("No trained models found.")
    latest = models[-1]
    logger.info(f"Loaded model: {latest.name}")
    return joblib.load(latest), latest.name
