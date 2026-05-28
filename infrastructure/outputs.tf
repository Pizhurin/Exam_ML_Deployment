# infrastructure/outputs.tf

output "server_ip" {
  description = "Публичный IPv4 адрес ML-сервера"
  value       = beget_compute_instance.ml_vps.ip_address
}

output "ssh_command" {
  description = "Подключение к серверу"
  value       = "ssh root@${beget_compute_instance.ml_vps.ip_address}"
}

output "mlflow_url" {
  description = "MLflow UI (после деплоя)"
  value       = "http://${beget_compute_instance.ml_vps.ip_address}:5000"
}

output "airflow_url" {
  description = "Airflow UI (после деплоя)"
  value       = "http://${beget_compute_instance.ml_vps.ip_address}:8080"
}

output "api_url" {
  description = "Churn Prediction API (после деплоя)"
  value       = "http://${beget_compute_instance.ml_vps.ip_address}:8000"
}

output "grafana_url" {
  description = "Grafana Dashboard (после деплоя)"
  value       = "http://${beget_compute_instance.ml_vps.ip_address}:3000"
}
