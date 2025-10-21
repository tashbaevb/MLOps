from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from ml.retrain import main as retrain_model

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG("synthetic_demand_dag", default_args=default_args, schedule="@daily", catchup=False) as dag:
    retrain_task = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
