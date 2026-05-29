#   1. Устанавливаем Docker + Docker Compose Plugin
#   2. Настраиваем firewall (ufw)
#   3. Копируем проект на сервер
#   4. Генерируем .env
#   5. Запускаем docker compose

set -euo pipefail

SERVER_IP="${1:?Укажи IP сервера: bash bootstrap.sh <IP>}"
REMOTE_USER="root"
PROJECT_DIR="/opt/ml-churn"
LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"

echo "--------------------------------------------"
echo "    ML-стек на Beget VPS: $SERVER_IP"
echo "--------------------------------------------"

# 1: Проверяем SSH-доступ
echo "→ Проверка SSH-доступа..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${REMOTE_USER}@${SERVER_IP}" "echo 'SSH OK'"

# 2: Установка Docker на сервере
echo "→ Установка Docker..."
ssh "${REMOTE_USER}@${SERVER_IP}" bash << 'REMOTE'
set -e
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg lsb-release ufw

# Docker GPG и репозиторий
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker
systemctl start docker
echo "Docker $(docker --version) установлен"
REMOTE

# 3: Настройка firewall
echo "→ Настройка ufw..."
ssh "${REMOTE_USER}@${SERVER_IP}" bash << 'REMOTE'
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    comment 'SSH'
ufw allow 8000/tcp  comment 'Churn API'
ufw allow 5000/tcp  comment 'MLflow UI'
ufw allow 8080/tcp  comment 'Airflow UI'
ufw allow 3000/tcp  comment 'Grafana'
ufw --force enable
echo "Firewall настроен"
REMOTE

# 4: Копируем проект на сервер
echo "→ Копирование проекта на сервер..."
ssh "${REMOTE_USER}@${SERVER_IP}" "mkdir -p ${PROJECT_DIR}"
rsync -az --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  "${LOCAL_PROJECT}/" \
  "${REMOTE_USER}@${SERVER_IP}:${PROJECT_DIR}/"

# 5: Генерация .env на сервере
echo "→ Генерация .env..."
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || \
             python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())")

POSTGRES_PASS="pg_$(openssl rand -hex 12)"
AIRFLOW_PASS="af_$(openssl rand -hex 8)"
GRAFANA_PASS="gf_$(openssl rand -hex 8)"

ssh "${REMOTE_USER}@${SERVER_IP}" bash << REMOTE
cat > ${PROJECT_DIR}/.env << EOF
POSTGRES_PASSWORD=${POSTGRES_PASS}
AIRFLOW_FERNET_KEY=${FERNET_KEY}
AIRFLOW_ADMIN_PASSWORD=${AIRFLOW_PASS}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_PASS}
SERVER_IP=${SERVER_IP}
EOF
chmod 600 ${PROJECT_DIR}/.env
echo ".env создан"
REMOTE

# 6: Запуск стека
echo "→ Запуск docker compose..."
ssh "${REMOTE_USER}@${SERVER_IP}" bash << REMOTE
cd ${PROJECT_DIR}
docker compose pull --quiet
docker compose up -d --build
echo "Ожидаем инициализации Airflow (90 сек)..."
sleep 90
docker compose ps
REMOTE

# 6: Вывод итогов
echo ""
echo "--------------------------------------------"
echo "    ДЕПЛОЙ ЗАВЕРШЁН!"
echo "--------------------------------------------"
echo "    Churn API  : http://${SERVER_IP}:8000"
echo "    MLflow UI  : http://${SERVER_IP}:5000"
echo "    Airflow UI : http://${SERVER_IP}:8080"
echo "    Grafana    : http://${SERVER_IP}:3000"
echo "--------------------------------------------"
echo "    Пароли сохранены в ${PROJECT_DIR}/.env"
echo "--------------------------------------------"

# Получаем пароли для вывода
ssh "${REMOTE_USER}@${SERVER_IP}" "grep -E 'PASSWORD|FERNET' ${PROJECT_DIR}/.env"
