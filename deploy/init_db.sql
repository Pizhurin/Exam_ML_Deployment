-- Создаём отдельные БД для MLflow и Feature Store

CREATE DATABASE mlflow;
CREATE DATABASE features;

GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE features TO airflow;

-- 1. Feature Store: таблица для хранения признаков
\connect features;
CREATE TABLE IF NOT EXISTS customer_features (
    customer_id     VARCHAR(50) PRIMARY KEY,
    tenure          INTEGER,
    monthly_charges FLOAT,
    total_charges   FLOAT,
    contract        INTEGER,
    internet_service INTEGER,
    num_services    INTEGER,
    avg_monthly_charges FLOAT,
    charges_delta   FLOAT,
    is_long_contract INTEGER,
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- 2. Таблица результатов скоринга
CREATE TABLE IF NOT EXISTS churn_scores (
    id              SERIAL PRIMARY KEY,
    customer_id     VARCHAR(50) NOT NULL,
    churn_probability FLOAT NOT NULL,
    is_at_risk      BOOLEAN NOT NULL,
    risk_level      VARCHAR(10),
    top_reasons     JSONB,
    model_version   VARCHAR(50),
    scored_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_churn_scores_customer ON churn_scores(customer_id);
CREATE INDEX IF NOT EXISTS idx_churn_scores_scored_at ON churn_scores(scored_at);
