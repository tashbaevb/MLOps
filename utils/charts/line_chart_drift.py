import matplotlib.pyplot as plt
import pandas as pd
from utils import io_utils, config


def line_chart():
    data = pd.read_csv(config.DATA_FILE)
    X = data[['weeks']]
    y = data['demand']

    old_model, _ = io_utils.load_latest_model(config.NO_MLOPS_MODEL_DIR)
    new_model, _ = io_utils.load_latest_model(config.MODEL_DIR)

    plt.plot(data['weeks'], y, label="Real Demand", color="black")
    plt.plot(data['weeks'], old_model.predict(X), label="No MLOps (stale)", linestyle="--", color="red")
    plt.plot(data['weeks'], new_model.predict(X), label="With MLOps (updated)", linestyle="--", color="green")
    plt.xlabel("Weeks")
    plt.ylabel("Demand")
    plt.legend()
    plt.title("Model Drift: With vs Without MLOps")
    plt.show()


def bart_chart():
    mae_no_mlops = 12.4
    mae_with_mlops = 3.1

    plt.bar(["No MLOps", "With MLOps"], [mae_no_mlops, mae_with_mlops], color=["red", "green"])
    plt.title("Model Error (MAE) Comparison")
    plt.ylabel("MAE")
    plt.show()
