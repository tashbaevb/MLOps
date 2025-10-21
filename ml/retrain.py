import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from utils import config, io_utils, mlflow_utils
from ml.pipeline import build_pipeline
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


# To-Do: Check for the last week, and retrain only if, but not now

def main():
    mlflow_utils.setup_mlflow()

    data = pd.read_csv(config.DATA_FILE)
    recent = data.iloc[-5:]
    X_recent, y_recent = recent[["weeks"]], recent["demand"]

    try:
        model, model_name = io_utils.load_latest_model(config.MODEL_DIR)
    except FileNotFoundError:
        logger.warning("No model found, running initial training.")
        from ml.train import main as train_main
        train_main()
        return

    y_pred = model.predict(X_recent)
    mae = mean_absolute_error(y_recent, y_pred)
    logger.info(f"Current MAE={mae:.2f}")

    mlflow_utils.log_run(model, mae, {"model_checked": model_name}, "synthetic_demand_forecast")

    if mae > config.THRESHOLD_MAE:
        logger.info("Drift detected — retraining model...")
        train = data.iloc[-15:]
        X, y = train[["weeks"]], train["demand"]

        new_model = build_pipeline()
        new_model.fit(X, y)

        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        io_utils.save_model(new_model, config.MODEL_DIR, version)
        mlflow_utils.log_run(new_model, mae, {"retrained_from": model_name}, "synthetic_demand_forecast")

        logger.info(f"New model retrained (v{version}).")
    else:
        logger.info("Model performance acceptable. No retraining required.")


if __name__ == "__main__":
    main()
