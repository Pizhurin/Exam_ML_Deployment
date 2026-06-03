"""
Загрузка и очистка данных → Feature Engineering → Training → Validation → Deploy
Расписание: каждый день в 00:00 UTC
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

# Дефолтные аргументы DAG
DEFAULT_ARGS = {
    "owner": "pin",
    "depends_on_past": False,
    "email": ["pizhurin@yahoo.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# Функции для выполнения задач
def task_extract_and_validate(**context):
    """
    Забирает свежие данные из S3, выполняет schema validation.
    При ошибке — поднимает исключение, Airflow делает retry.
    """
    import boto3
    import pandas as pd
    import io
    from airflow.providers.amazon.aws.hooks.aws import AwsHook

    s3_bucket = Variable.get("S3_BUCKET", default_var="ml-data-bucket")
    s3_key = Variable.get("S3_DATA_KEY", default_var="telco/latest/churn_data.csv")
    endpoint = Variable.get("S3_ENDPOINT", default_var="https://storage.yandexcloud.net")

    # Инициализируем хук с ID вашего нового соединения
    hook = AwsHook(aws_conn_id='yandex_s3')
    
    # Получаем клиент s3 с уже настроенными внутри ключами
    s3 = hook.get_client_type('s3', region_name=None)
    # Принудительно устанавливаем endpoint для Yandex Object Storage
    s3.meta.endpoint_url = endpoint

    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    required_columns = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    if len(df) < 1000:
        raise ValueError(f"Данных слишком мало: {len(df)} строк (минимум 1000)")

    # Сохраняем во временный файл для следующего шага
    local_path = "/tmp/churn_data_raw.csv"
    df.to_csv(local_path, index=False)

    # Передаём путь через XCom
    context["ti"].xcom_push(key="raw_data_path", value=local_path)
    print(f"Загружено {len(df)} записей из s3://{s3_bucket}/{s3_key}")


def task_feature_engineering(**context):
    """Feature Engineering и сохранение признаков в Feature Store."""
    from pipeline.churn_pipeline import load_and_clean, engineer_features
    import joblib

    raw_path = context["ti"].xcom_pull(key="raw_data_path", task_ids="extract_and_validate")
    df_raw = load_and_clean(raw_path)
    df_features, encoders = engineer_features(df_raw)

    # Сохраняем processed датасет
    processed_path = "/tmp/churn_features.csv"
    df_features.to_csv(processed_path, index=False)

    # Сохраняем энкодеры в Feature Store
    encoders_path = "/tmp/feature_encoders.pkl"
    joblib.dump(encoders, encoders_path)

    context["ti"].xcom_push(key="features_path", value=processed_path)
    context["ti"].xcom_push(key="encoders_path", value=encoders_path)
    print(f"Feature Engineering завершён. Признаков: {df_features.shape[1] - 1}")


def task_train_model(**context):
    """Обучение модели с логированием в MLflow."""
    import pandas as pd
    from pipeline.churn_pipeline import train_model

    features_path = context["ti"].xcom_pull(key="features_path", task_ids="feature_engineering")
    df = pd.read_csv(features_path)

    run_id = train_model(df)
    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    print(f"Обучение завершено. MLflow run_id: {run_id}")


def task_validate_model(**context):
    """Валидация модели. Возвращает 'deploy' или 'skip_deploy'."""
    from pipeline.churn_pipeline import validate_and_promote

    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train_model")
    promoted = validate_and_promote(run_id)

    context["ti"].xcom_push(key="model_promoted", value=promoted)

    if promoted:
        print("Модель прошла валидацию. Ветка: deploy")
        return "deploy_to_kubernetes"
    else:
        print("Модель не прошла валидацию. Ветка: skip")
        return "skip_deploy"


def task_deploy(**context):
    """Деплой новой модели в Kubernetes."""
    import mlflow

    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train_model")
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    model_version = versions[0].version if versions else "latest"

    from pipeline.churn_pipeline import trigger_deployment  
    trigger_deployment(model_version=str(model_version))
    print(f"Деплой завершён. Версия модели: {model_version}")


def task_run_monitoring(**context):
    """Запуск Evidently AI для drift-детекции после деплоя."""
    print("Запуск мониторинга дрейфа данных через Evidently AI...")
    # В реальной системе: вызов monitoring/drift_monitor.py
    print("Мониторинг запущен.")


# Определение DAG
with DAG(
    dag_id="churn_prediction_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description="Полный ML-пайплайн прогнозирования оттока: ETL→FE→Train→Validate→Deploy",
    schedule_interval="@daily",   # Каждый день в 00:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,               # Только один активный запуск одновременно
    tags=["ml", "churn", "production"],
) as dag:

    # Старт
    start = EmptyOperator(task_id="start")

    # 1: Загрузка и валидация данных
    extract_and_validate = PythonOperator(
        task_id="extract_and_validate",
        python_callable=task_extract_and_validate,
    )

    # 2: Feature Engineering
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=task_feature_engineering,
    )

    # 3: Обучение
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        execution_timeout=timedelta(hours=1),  # SLO: обучение < 60 мин
    )

    # 4: Валидация с ветвлением
    validate_model = BranchPythonOperator(
        task_id="validate_model",
        python_callable=task_validate_model,
    )

    # Ветка для деплоя
    deploy = PythonOperator(
        task_id="deploy",
        python_callable=task_deploy,
    )

    # Ветка с пропуском деплоя
    skip_deploy = EmptyOperator(task_id="skip_deploy")

    # Мониторинг
    run_monitoring = PythonOperator(
        task_id="run_monitoring",
        python_callable=task_run_monitoring,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # запускается, если хоть одна ветка успешна
    )

    # Окончание
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Зависимости
    (
        start
        >> extract_and_validate
        >> feature_engineering
        >> train_model_task
        >> validate_model
        >> [deploy, skip_deploy]
        >> run_monitoring
        >> end
    )
