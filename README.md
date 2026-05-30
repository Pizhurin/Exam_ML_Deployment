## Exam_ML_Deployment

> **MLOps Level 2** · Прогнозирование оттока  
> **Инфраструктура:** Beget VPS, Ubuntu 24.04, OpenTofu  

---

## Архитектура на VPS

```
Beget VPS (2 CPU / 4 GB RAM / 30 GB)
└── Docker Compose
    ├── postgres        — БД для Airflow + MLflow + Feature Store
    ├── redis           — брокер Celery для Airflow
    ├── mlflow          — реестр моделей (:5000)
    ├── airflow-webserver  (:8080)
    ├── airflow-scheduler
    ├── airflow-worker
    ├── churn-api       — FastAPI serving (:8000)
    ├── prometheus      — сбор метрик
    └── grafana         — дашборды (:3000)
```

---

## Быстрый старт

### 1. Поднять VPS через OpenTofu

```bash
cd infrastructure/

# Если ещё не установлен OpenTofu:
# https://opentofu.org/docs/intro/install/

tofu init
tofu plan
tofu apply

# Запомни IP из вывода:
# server_ip = "XX.XX.XX.XX"
```

### 2. Развернуть стек одной командой

```bash
chmod +x deploy/bootstrap.sh
bash deploy/bootstrap.sh <SERVER_IP>
```

Скрипт сам:
- установит Docker на сервере
- настроит firewall (ufw)
- скопирует проект
- сгенерирует случайные пароли
- запустит `docker compose up -d`

Готово. После ~2 минут доступны:

| Сервис | URL |
|--------|-----|
| **Churn API** | `http://<IP>:8000` |
| **MLflow UI** | `http://<IP>:5000` |
| **Airflow UI** | `http://<IP>:8080` |
| **Grafana** | `http://<IP>:3000` |

---

### 3. Загрузить датасет и обучить первую модель

```bash
# Подключиться к серверу
ssh root@<SERVER_IP>
cd /opt/ml-churn

# Скачать датасет
mkdir -p data
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/telco_churn.csv

# Запустить пайплайн вручную (первое обучение)
docker compose exec airflow-worker \
  python /opt/airflow/dags/churn_pipeline.py
```

---

## Управление

```bash
# Статус всех сервисов
docker compose ps

# Логи конкретного сервиса
docker compose logs -f churn-api
docker compose logs -f airflow-scheduler

# Перезапуск одного сервиса
docker compose restart churn-api

# Полная остановка
docker compose down

# Остановка с удалением данных (осторожно!)
docker compose down -v
```

---

## Тестирование API

```bash
# Health check
curl http://<IP>:8000/health

# Предсказание оттока
curl -X POST http://<IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
      "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
      "tenure": 2, "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
      "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
      "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
      "PaymentMethod": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
      "avg_monthly_charges": 85.5, "charges_delta": 0.0, "num_services": 2,
      "is_long_contract": 0
    }],
    "return_explanations": true
  }'
```

---

## CI/CD (GitHub Actions)

Добавь в Settings → Secrets репозитория:

| Secret | Значение |
|--------|---------|
| `BEGET_SERVER_IP` | IP сервера из `tofu output` |
| `SSH_PRIVATE_KEY` | Содержимое `~/.ssh/id_ed25519` |


После этого каждый `git push` в `main` → автоматический деплой обновлений на VPS.

---

## Структура проекта

```
.
├── infrastructure/
│   ├── main.tf             # OpenTofu: создание VPS на Beget
│   └── outputs.tf          # IP-адрес и SSH-команда
│
├── deploy/
│   ├── bootstrap.sh        # Скрипт первоначальной настройки сервера
│   ├── init_db.sql         # Создание БД mlflow, features, таблиц
│   ├── Dockerfile          # Mlflow build
│   └── prometheus.yml      # Конфиг сбора метрик
│
├── pipeline/
│   ├── churn_pipeline.py   # ETL → FE → обучение → валидация → деплой
│   └── airflow_dag.py      # Airflow DAG (расписание каждое воскресенье)
│
├── api/
│   ├── main.py             # FastAPI: /predict, /health, /metrics
│   └── Dockerfile          # Multi-stage build
│
├── monitoring/
│   └── drift_monitor.py    # PSI + drift detection → триггер переобучения
│
├── docs/
│   ├── ML_MANIFEST.md      # Манифест ML-системы
│   ├── SLI_SLO.md          # SLI/SLO на 3 уровнях
│   ├── ADR-001-*.md        # Архитектурное решение (MDD)
│   └── mdd_analysis.py     # Статистический анализ латентности
│
├── docker-compose.yml      # Весь стек одним файлом
├── .env.example            # Шаблон переменных окружения
└── requirements.txt
```

