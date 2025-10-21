import mlflow
from utils import config


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def log_run(model, mae: float, params: dict, model_name: str):
    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, name="model")
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name,
        )
