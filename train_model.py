import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import mlflow.sklearn
import joblib
import os
from datetime import datetime

DATA_FILE = "data/dataset/synthetic_demand.csv"
MODEL_DIR = "data/models"
METRICS_DIR = "data/metrics"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    mlflow.set_tracking_uri("file:./data/mlruns")
    mlflow.set_experiment("synthetic_demand_forecast")

    data = pd.read_csv(DATA_FILE)
    train = data.iloc[:-10]
    test = data.iloc[-10:]

    X_train, y_train = train[["weeks"]], train["demand"]
    X_test, y_test = test[["weeks"]], test["demand"]

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(MODEL_DIR, f"model_v{model_version}.pkl")
        joblib.dump(model, model_file)

        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mae", mae)
        mlflow.log_artifact(DATA_FILE)
        mlflow.log_artifact(model_file)
        mlflow.sklearn.log_model(model, "model")

        metrics_file = os.path.join(METRICS_DIR, f"metrics_v{model_version}.txt")
        with open(metrics_file, "w") as f:
            f.write(f"MAE: {mae}\n")
            f.write(f"Model version: {model_version}\n")

    print(f"Model {model_version} trained. MAE={mae:.2f}")


if __name__ == "__main__":
    main()
