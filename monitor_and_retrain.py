import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib, os
from datetime import datetime
import mlflow

DATA_FILE = "data/dataset/synthetic_demand.csv"
MODEL_DIR = "data/models"
METRICS_DIR = "data/metrics"
THRESHOLD_MAE = 10


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    mlflow.set_tracking_uri("file:./data/mlruns")
    mlflow.set_experiment("synthetic_demand_forecast")

    data = pd.read_csv(DATA_FILE)
    recent = data.iloc[-5:]
    X_recent, y_recent = recent[["weeks"]], recent[["demand"]]

    model_files = sorted(os.listdir(MODEL_DIR))
    latest_model_file = os.path.join(MODEL_DIR, model_files[-1])
    model = joblib.load(latest_model_file)

    y_pred = model.predict(X_recent)
    mae = mean_absolute_error(y_recent, y_pred)

    print(f"Latest model: {latest_model_file}, MAE={mae:.2f}")

    with mlflow.start_run():
        mlflow.log_metric("mae_recent", mae)
        mlflow.log_param("latest_model", latest_model_file)

        if mae > THRESHOLD_MAE:
            print("Drift detected → retraining model...")
            train = data.iloc[-15:]
            X, y = train[["weeks"]], train[["demand"]]
            new_model = LinearRegression()
            new_model.fit(X, y)

            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(MODEL_DIR, f"model_v{model_version}.pkl")
            joblib.dump(new_model, model_file)

            metrics_file = os.path.join(METRICS_DIR, f"metrics_v{model_version}.txt")
            with open(metrics_file, "w") as f:
                f.write(f"MAE: {mae}\nModel version: {model_version}\n")

            mlflow.log_artifact(model_file)
            mlflow.log_artifact(metrics_file)
            mlflow.sklearn.log_model(new_model, "model_retrained")

            print(f"New model retrained and saved: {model_file}")
        else:
            print("Model is still good, no retrain needed")


if __name__ == "__main__":
    main()
