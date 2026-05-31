# main.tf

terraform {
  required_providers {
    beget = {
      source = "tf.beget.com/beget/beget"
      version = ">= 0.0.33"
    }
  }
}

provider "beget" {}

data "beget_software" "ubuntu" {
  slug = "ubuntu-24-04"
}

resource "beget_ssh_key" "my_key" {
  name       = "ml-ssh-key"
  public_key = file("~/.ssh/id_ed25519.pub")
}

resource "beget_compute_instance" "ml_vps" {
  name        = "exam-ml-deployment"
  description = "ML-сервер: Airflow + MLflow + FastAPI + Postgres + Grafana"
  region      = "ru1"

  configuration = {
    cpu       = 2
    ram_mb    = 4096   # 4 GB — минимум для Airflow + MLflow одновременно
    disk_mb   = 30720  # 30 GB — данные, модели, логи, Docker-образы
    cpu_class = "normal_cpu"
  }

  image = {
    software = {
      id = data.beget_software.ubuntu.id
    }
  }

  access = {
    ssh_keys = [beget_ssh_key.my_key.id]
  }
}
