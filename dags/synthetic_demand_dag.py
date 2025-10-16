from airflow.sdk import dag, task
from datetime import datetime, timedelta
import sys, os
from train_model import main as train_main
from monitor_and_retrain import main as monitor_main

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

default_args = {
    "owner": "mpols_user",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}


@dag(
    dag_id="synthetic_demand_pipeline",
    default_args=default_args,
    description="Train and monitor synthetic demand model",
    schedule="*/5 * * * *",
    start_date=datetime(2025, 10, 8),
    catchup=False,
    tags=["mlops", "synthetic_demand"]
)
def synthetic_demand_pipeline():
    @task()
    def train_task():
        train_main()

    @task()
    def monitor_task():
        monitor_main()

    train = train_task()
    monitor = monitor_task()

    train >> monitor


dag = synthetic_demand_pipeline()
