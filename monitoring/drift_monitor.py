from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
import requests
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка значений SLO (из SLI_SLO.md)
PSI_THRESHOLD_WARNING = 0.10
PSI_THRESHOLD_CRITICAL = 0.20
DRIFT_FEATURES_PCT_THRESHOLD = 0.30  # 30% признаков с дрейфом → алерт
PREDICTION_KL_THRESHOLD = 0.10
RETRAINING_TRIGGER_URL = os.getenv(
    "AIRFLOW_TRIGGER_URL", 
    "http://airflow:8080/api/v1/dags/churn_prediction_ml_pipeline/dagRuns"
    )


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    PSI < 0.10 — норма, 0.10–0.20 — умеренный сдвиг, > 0.20 — критический.
    """
    # Baseline
    breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # убираем дублирующиеся края

    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]

    eps = 1e-8
    baseline_pct = (baseline_counts + eps) / len(baseline)
    current_pct = (current_counts + eps) / len(current)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def detect_feature_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Определяет data drift для каждого признака.
    Числовые признаки - тест Колмогорова–Смирнова.
    Категориальные - тест хи-квадрат.
    """
    feature_cols = [c for c in baseline_df.columns if c != "Churn"]
    drift_results = {}
    drifted_count = 0

    for col in feature_cols:
        base_vals = baseline_df[col].dropna()
        curr_vals = current_df[col].dropna()

        if base_vals.dtype in [np.float64, np.float32, np.int64, np.int32]:
            # двухвыборочный, α=0.05
            stat, p_value = stats.ks_2samp(base_vals, curr_vals)
            test = "KS"
        else:
            all_cats = pd.concat([base_vals, curr_vals]).unique()
            base_counts = base_vals.value_counts().reindex(all_cats, fill_value=0)
            curr_counts = curr_vals.value_counts().reindex(all_cats, fill_value=0)
            stat, p_value = stats.chisquare(curr_counts, base_counts * len(curr_vals) / len(base_vals))
            test = "Chi2"

        is_drifted = p_value < 0.05
        if is_drifted:
            drifted_count += 1

        drift_results[col] = {
            "test": test,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "is_drifted": is_drifted,
            "psi": round(compute_psi(base_vals.values, curr_vals.values), 4)
                   if base_vals.dtype in [np.float64, np.float32, np.int64, np.int32] else None,
        }

    drift_rate = drifted_count / len(feature_cols) if feature_cols else 0
    return {
        "features": drift_results,
        "drifted_features_count": drifted_count,
        "total_features": len(feature_cols),
        "drift_rate": round(drift_rate, 4),
        "drift_alarm": drift_rate >= DRIFT_FEATURES_PCT_THRESHOLD,
    }


def monitor_prediction_drift(baseline_scores: np.ndarray, current_scores: np.ndarray) -> Dict[str, Any]:
    """
    Мониторинг дрейфа распределения предсказаний (KL-дивергенция).
    """
    eps = 1e-8
    bins = np.linspace(0, 1, 20)

    base_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)
    curr_hist, _ = np.histogram(current_scores, bins=bins, density=True)

    base_hist = base_hist + eps
    curr_hist = curr_hist + eps

    base_hist /= base_hist.sum()
    curr_hist /= curr_hist.sum()

    kl_divergence = float(stats.entropy(curr_hist, base_hist))

    return {
        "kl_divergence": round(kl_divergence, 6),
        "prediction_drift_alarm": kl_divergence >= PREDICTION_KL_THRESHOLD,
        "current_at_risk_pct": round(float((current_scores >= 0.4).mean()), 4),
        "baseline_at_risk_pct": round(float((baseline_scores >= 0.4).mean()), 4),
    }


def trigger_retraining(reason: str):
    """Отправляет POST-запрос в Airflow для запуска переобучения."""
    logger.warning(f"🚨 Триггер переобучения: {reason}")
    payload = {
        "dag_run_id": f"drift_trigger_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "conf": {"trigger_reason": reason, "triggered_by": "drift_monitor"},
    }
    try:
        response = requests.post(
            RETRAINING_TRIGGER_URL,
            json=payload,
            auth=("airflow", os.getenv("AIRFLOW_PASSWORD", "airflow")),
            timeout=30,
        )
        if response.status_code in (200, 201):
            logger.info("✅ DAG переобучения запущен успешно")
        else:
            logger.error(f"❌ Airflow API вернул {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"❌ Не удалось запустить Airflow DAG: {e}")


def run_monitoring(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Основная функция мониторинга. Объединяет все проверки.
    """
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "baseline_rows": len(baseline_data),
        "current_rows": len(current_data),
    }

    # 1. Дрейф признаков
    logger.info("Проверка дрейфа признаков...")
    drift_report = detect_feature_drift(baseline_data, current_data)
    report["feature_drift"] = drift_report

    # 2. Дрейф предсказаний
    logger.info("Проверка дрейфа предсказаний...")
    prediction_drift = monitor_prediction_drift(baseline_scores, current_scores)
    report["prediction_drift"] = prediction_drift

    # 3. PSI признаков
    key_features = ["tenure", "MonthlyCharges", "Contract"]
    psi_results = {}
    for feat in key_features:
        if feat in baseline_data.columns:
            psi = compute_psi(baseline_data[feat].values, current_data[feat].values)
            psi_results[feat] = {
                "psi": round(psi, 4),
                "status": "OK" if psi < PSI_THRESHOLD_WARNING
                          else ("WARNING" if psi < PSI_THRESHOLD_CRITICAL else "CRITICAL"),
            }
            if psi >= PSI_THRESHOLD_CRITICAL:
                logger.warning(f"⚠️ PSI критический для {feat}: {psi:.4f}")
    report["psi_key_features"] = psi_results

    # 4. Решение о переобучении
    retrain_reasons = []

    if drift_report["drift_alarm"]:
        retrain_reasons.append(
            f"Дрейф признаков: {drift_report['drift_rate']:.1%} > 30%"
        )

    if prediction_drift["prediction_drift_alarm"]:
        retrain_reasons.append(
            f"KL-дивергенция предсказаний: {prediction_drift['kl_divergence']:.4f} > {PREDICTION_KL_THRESHOLD}"
        )

    critical_psi_features = [
        f for f, v in psi_results.items() if v["status"] == "CRITICAL"
    ]
    if critical_psi_features:
        retrain_reasons.append(
            f"Критический PSI у признаков: {', '.join(critical_psi_features)}"
        )

    report["retrain_triggered"] = bool(retrain_reasons)
    report["retrain_reasons"] = retrain_reasons

    # 5. Тригер переобучения
    if retrain_reasons:
        trigger_retraining(reason=" | ".join(retrain_reasons))
    else:
        logger.info("✅ Дрейфа не обнаружено. Переобучение не требуется.")

    # Сохранить отчёт
    report_path = f"/tmp/drift_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Отчёт сохранён: {report_path}")

    return report


if __name__ == "__main__":
    np.random.seed(42)
    n = 5000
    feature_names = [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract",
        "InternetService", "num_services", "charges_delta", "avg_monthly_charges"
    ]

    baseline_df = pd.DataFrame({
        "tenure": np.random.randint(0, 72, n),
        "MonthlyCharges": np.random.normal(65, 30, n).clip(20, 120),
        "TotalCharges": np.random.normal(2500, 2000, n).clip(0, 8000),
        "Contract": np.random.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20]),
        "InternetService": np.random.choice([0, 1, 2], n),
        "num_services": np.random.randint(0, 8, n),
        "charges_delta": np.random.normal(0, 10, n),
        "avg_monthly_charges": np.random.normal(60, 25, n).clip(15, 100),
    })

    # Имитируем умеренный дрейф
    current_df = baseline_df.copy()
    current_df["MonthlyCharges"] = np.random.normal(72, 30, n).clip(20, 120)  # сдвиг +7
    current_df["tenure"] = np.random.randint(0, 60, n)  # стало меньше долгосрочных

    baseline_scores = np.random.beta(2, 5, n)
    current_scores = np.random.beta(2.5, 4, n)  # небольшой дрейф предсказаний

    report = run_monitoring(baseline_df, current_df, baseline_scores, current_scores)
    print(json.dumps(report, ensure_ascii=False, indent=2))
