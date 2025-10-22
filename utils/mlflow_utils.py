import mlflow
from utils import config
from mlflow.models.signature import infer_signature
import pandas as pd


def setup_mlflow():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def log_run(model, mae: float, params: dict, model_name: str):
    input_example = pd.DataFrame({"weeks": [10.0]})
    signature = infer_signature(input_example, model.predict(input_example))

    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(model, name="model", input_example=input_example, signature=signature)
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                model_name,
            )
