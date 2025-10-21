import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from utils import config, io_utils, mlflow_utils
from ml.pipeline import build_pipeline
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    mlflow_utils.setup_mlflow()

    data = pd.read_csv(config.DATA_FILE)
    train, test = data.iloc[:-10], data.iloc[-10:]

    X_train, y_train = train[["weeks"]], train["demand"]
    X_test, y_test = test[["weeks"]], test["demand"]

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = io_utils.save_model(model, config.MODEL_DIR, version)

    mlflow_utils.log_run(model, mae, {"version": version}, "synthetic_demand_forecast")
    logger.info(f"Model trained (MAE={mae:.2f}) and logged to MLflow.")


if __name__ == "__main__":
    main()
