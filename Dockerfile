FROM python:3.11-slim

ENV AIRFLOW_HOME=/opt/mlops/airflow
WORKDIR $AIRFLOW_HOME

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY . $AIRFLOW_HOME

RUN mkdir -p $AIRFLOW_HOME/dags \
    $AIRFLOW_HOME/data/models \
    $AIRFLOW_HOME/data/metrics \
    $AIRFLOW_HOME/data/mlruns

ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_FOLDER=$AIRFLOW_HOME/dags
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:///$AIRFLOW_HOME/airflow.db
ENV MLFLOW_TRACKING_URI=file://$AIRFLOW_HOME/data/mlruns

WORKDIR /opt/mlops
