import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any

import subprocess
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
import lightgbm as lgb
import shap
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Конфигурация
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn-lgbm"
MIN_ROC_AUC = 0.78 
CHAMPION_DELTA = 0.01 
DATA_PATH = os.getenv("DATA_PATH", "data/telco_churn.csv")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Загружает датасет и выполняет очистку.
    Возвращает очищенный DataFrame.
    """
    logger.info(f"Загрузка данных из {path}")
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop(columns=["customerID"])

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    logger.info(f"Загружено {len(df)} записей. Churn rate: {df['Churn'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Создаёт признаки из сырых данных.
    Возвращает преобразованный DataFrame и словарь энкодеров для Feature Store.
    """
    logger.info("Feature engineering...")
    df = df.copy()

    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = (df[col] == "Yes").astype(int)

    encoders = {}
    for col in ["gender", "InternetService", "Contract", "PaymentMethod"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Среднемесячные расходы (TotalCharges / tenure)
    df["avg_monthly_charges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Разница фактических и средних расходов
    df["charges_delta"] = df["MonthlyCharges"] - df["avg_monthly_charges"]

    # Количество подключённых сервисов
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].sum(axis=1)

    # Клиент с контрактом на 1-2 года более лоялен
    df["is_long_contract"] = (df["Contract"] >= 1).astype(int)

    logger.info(f"Итоговых признаков: {df.shape[1] - 1}")
    return df, encoders


def train_model(df: pd.DataFrame) -> str:
    """
    Обучает LightGBM-модель, логирует эксперимент в MLflow.
    Возвращает run_id.
    """
    target = "Churn"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),  # балансировка классов
        "random_state": 42,
        "n_jobs": -1,
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"lgbm_{datetime.utcnow().strftime('%Y%m%d_%H%M')}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Логируем параметры
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("churn_rate", float(y.mean()))

        # Кросс-валидация на train
        cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(cv_kfold.split(X_train, y_train)):
            fold_model = lgb.LGBMClassifier(**params)
            fold_model.fit(
                X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
            )
            fold_score = roc_auc_score(y_train.iloc[val_idx], fold_model.predict_proba(X_train.iloc[val_idx])[:, 1])
            cv_scores.append(fold_score)
            logger.info(f"  Fold {fold_idx+1}/5 AUC: {fold_score:.4f}")

        mlflow.log_metric("cv_auc_mean", np.mean(cv_scores))
        mlflow.log_metric("cv_auc_std", np.std(cv_scores))

        # Обучение
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
        )

        # Метрики на тесте
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.4).astype(int)

        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Precision в топ-20
        top20_threshold = np.percentile(y_proba, 80)
        top20_mask = y_proba >= top20_threshold
        precision_top20 = y_test[top20_mask].mean() if top20_mask.sum() > 0 else 0.0

        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_precision_top20pct", precision_top20)

        logger.info(f"Test ROC-AUC: {roc_auc:.4f} | Precision@top20%: {precision_top20:.4f}")

        # SHAP для объяснимости
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(X_test[:100])
        mlflow.log_text(str(feature_cols), "feature_names.txt")

        # Логируем модель в MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(5)
        )

        # Сохраняем explainer
        joblib.dump(shap_explainer, "/tmp/shap_explainer.pkl")
        mlflow.log_artifact("/tmp/shap_explainer.pkl", artifact_path="explainer")

    return run_id


def validate_and_promote(run_id: str) -> bool:
    """
    Проверяет метрики модели против SLO.
    Если метрики выше порога — продвигает как Prod.
    Если есть текущий champion — сравнивает с ним и продвигает только при улучшении на delta.
    Возвращает True, если модель продвинута в Production, иначе False."""

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    roc_auc = metrics.get("test_roc_auc", 0)
    precision_top20 = metrics.get("test_precision_top20pct", 0)

    logger.info(f"Валидация: ROC-AUC={roc_auc:.4f}, Precision@top20={precision_top20:.4f}")

    # Минимальный SLO
    if roc_auc < MIN_ROC_AUC:
        logger.warning(f"ROC-AUC {roc_auc:.4f} < SLO {MIN_ROC_AUC}. Модель не продвигается.")
        return False

    # Сравниваем с текущей прод. моделью
    try:
        champion_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if champion_versions:
            champion_run_id = champion_versions[0].run_id
            champion_metrics = client.get_run(champion_run_id).data.metrics
            champion_auc = champion_metrics.get("test_roc_auc", 0)

            if roc_auc < champion_auc + CHAMPION_DELTA:
                logger.info(
                    f"Challenger AUC {roc_auc:.4f} не превышает Champion AUC {champion_auc:.4f} "
                    f"на delta={CHAMPION_DELTA}. Деплой отменён."
                )
                return False

            for v in champion_versions:
                client.transition_model_version_stage(MODEL_NAME, v.version, "Archived")
            logger.info(f"Предыдущий champion v{champion_versions[0].version} архивирован")

    except Exception as e:
        logger.info(f"Production-модели не найдено или ошибка: {e}. Первый деплой.")

    # Версию по run_id в прод
    all_versions = client.search_model_versions(f"run_id='{run_id}'")
    if not all_versions:
        logger.error("Версия модели не найдена в реестре!")
        return False

    new_version = all_versions[0].version
    client.transition_model_version_stage(MODEL_NAME, new_version, "Production")
    logger.info(f"Модель v{new_version} (run={run_id}) продвинута в Production. ROC-AUC={roc_auc:.4f}")
    return True


def trigger_deployment(model_version: str):
    """
    Перезапускает контейнер churn-api.
    При рестарте FastAPI заново загружает модель из MLflow Registry.
    """
    
    logger.info(f"Деплой новой модели v{model_version}: перезапуск контейнера churn-api...")

    # Путь к docker-compose.yml — ищем от корня проекта
    compose_file = os.getenv("COMPOSE_FILE", "/opt/ml-churn/docker-compose.yml")
    project_dir = os.path.dirname(compose_file)

    cmd = [
        "docker", "compose",
        "-f", compose_file,
        "restart", "churn-api"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=project_dir,
        )
        if result.returncode == 0:
            logger.info("Контейнер churn-api перезапущен. Новая модель загружена.")
        else:
            logger.warning(
                f"Docker compose restart вернул код {result.returncode}.\n"
                f"stderr: {result.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        logger.error("Таймаут при перезапуске churn-api (> 60 сек)")


def run_pipeline():
    logger.info("=" * 60)
    logger.info("Запуск полного ML-пайплайна")
    logger.info("=" * 60)

    # 1. Загрузка и очистка данных
    df_raw = load_and_clean(DATA_PATH)

    # 2. Feature Engineering
    df_features, encoders = engineer_features(df_raw)

    # 3. Обучение
    run_id = train_model(df_features)

    # 4. Валидация и продвижение
    promoted = validate_and_promote(run_id)

    # 5. Деплой
    if promoted:
        trigger_deployment(model_version=run_id)
        logger.info("Пайплайн завершён успешно. Новая модель в Production.")
    else:
        logger.warning("Пайплайн завершён. Модель не прошла валидацию — Production не изменён.")

    return promoted


if __name__ == "__main__":
    run_pipeline()