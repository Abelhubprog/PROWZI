# Google Cloud Platform Cost-Optimized Infrastructure
# Zero-budget deployment using free tier and cheapest services

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.84"
    }
  }
}

# Variables for cost optimization
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region (us-central1 for lowest cost)"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (development/staging/production)"
  type        = string
  default     = "development"
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "sql.googleapis.com",
    "redis.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudfunctions.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# GKE Autopilot Cluster (Cost-Optimized)
resource "google_container_cluster" "prowzi_cluster" {
  name     = "prowzi-cluster"
  location = var.region
  project  = var.project_id
  
  # Autopilot mode for cost optimization and minimal management
  enable_autopilot = true
  
  # Network configuration
  network    = google_compute_network.prowzi_vpc.self_link
  subnetwork = google_compute_subnetwork.prowzi_subnet.self_link
  
  # IP allocation for cost optimization
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "10.1.0.0/16"
    services_ipv4_cidr_block = "10.2.0.0/16"
  }
  
  # Security hardening
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Network policy for security
  network_policy {
    enabled = true
  }
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Maintenance window during low-cost hours
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
  
  depends_on = [
    google_project_service.required_apis
  ]
}

# VPC Network (Free tier)
resource "google_compute_network" "prowzi_vpc" {
  name                    = "prowzi-vpc"
  project                 = var.project_id
  auto_create_subnetworks = false
  mtu                     = 1460
}

# Subnet (Free tier)
resource "google_compute_subnetwork" "prowzi_subnet" {
  name          = "prowzi-subnet"
  project       = var.project_id
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.prowzi_vpc.id
  
  # Enable private Google access for cost savings
  private_ip_google_access = true
  
  # Secondary ranges for GKE
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Cloud SQL PostgreSQL (Smallest instance)
resource "google_sql_database_instance" "prowzi_db" {
  name             = "prowzi-db-${var.environment}"
  project          = var.project_id
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    # Smallest tier for cost optimization
    tier = "db-f1-micro"
    
    # Disk configuration for cost savings
    disk_type    = "PD_HDD"  # Cheapest disk type
    disk_size    = 10        # Minimum size
    disk_autoresize       = true
    disk_autoresize_limit = 20  # Limit growth for cost control
    
    # Backup configuration (minimal retention)
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = false  # Disable for cost savings
      location                       = var.region
      
      backup_retention_settings {
        retained_backups = 7  # Minimum for safety
        retention_unit   = "COUNT"
      }
    }
    
    # IP configuration
    ip_configuration {
      ipv4_enabled    = false  # Disable public IP for security and cost
      private_network = google_compute_network.prowzi_vpc.id
      require_ssl     = true
    }
    
    # Maintenance window
    maintenance_window {
      hour = 3
      day  = 7  # Sunday
      update_track = "stable"
    }
  }
  
  deletion_protection = false  # Allow deletion for development
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection
  ]
}

# Database
resource "google_sql_database" "prowzi_database" {
  name     = "prowzi"
  instance = google_sql_database_instance.prowzi_db.name
  project  = var.project_id
}

# Database user
resource "google_sql_user" "prowzi_user" {
  name     = "prowzi_user"
  instance = google_sql_database_instance.prowzi_db.name
  password = random_password.db_password.result
  project  = var.project_id
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Private service networking for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "prowzi-private-ip"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.prowzi_vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.prowzi_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Cloud Memorystore Redis (Smallest instance)
resource "google_redis_instance" "prowzi_redis" {
  name           = "prowzi-redis"
  project        = var.project_id
  region         = var.region
  memory_size_gb = 1  # Smallest size
  tier           = "BASIC"  # Cheapest tier
  
  # Security
  auth_enabled = true
  transit_encryption_mode = "SERVER_AUTH"
  
  # Network
  authorized_network = google_compute_network.prowzi_vpc.id
  connect_mode      = "PRIVATE_SERVICE_ACCESS"
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection
  ]
}

# Cloud Storage bucket for backups (Standard storage)
resource "google_storage_bucket" "prowzi_backups" {
  name     = "${var.project_id}-prowzi-backups"
  project  = var.project_id
  location = var.region
  
  # Cost optimization
  storage_class = "STANDARD"
  
  # Lifecycle management for cost control
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
  
  # Security
  uniform_bucket_level_access = true
  
  # Versioning disabled for cost savings
  versioning {
    enabled = false
  }
}

# Pub/Sub topic for async messaging
resource "google_pubsub_topic" "prowzi_events" {
  name    = "prowzi-events"
  project = var.project_id
  
  # Message retention for cost control
  message_retention_duration = "604800s"  # 7 days
}

# Secret Manager for secure storage
resource "google_secret_manager_secret" "jwt_private_key" {
  secret_id = "prowzi-jwt-private-key"
  project   = var.project_id
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret" "database_url" {
  secret_id = "prowzi-database-url"
  project   = var.project_id
  
  replication {
    automatic = true
  }
}

# Store database URL in Secret Manager
resource "google_secret_manager_secret_version" "database_url" {
  secret = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://${google_sql_user.prowzi_user.name}:${google_sql_user.prowzi_user.password}@${google_sql_database_instance.prowzi_db.private_ip_address}:5432/${google_sql_database.prowzi_database.name}"
}

# Service Account for Workload Identity
resource "google_service_account" "prowzi_workload_sa" {
  account_id   = "prowzi-workload-sa"
  project      = var.project_id
  display_name = "Prowzi Workload Identity Service Account"
  description  = "Service account for Prowzi workloads with minimal permissions"
}

# IAM binding for Secret Manager
resource "google_secret_manager_secret_iam_binding" "jwt_key_access" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.jwt_private_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  
  members = [
    "serviceAccount:${google_service_account.prowzi_workload_sa.email}"
  ]
}

resource "google_secret_manager_secret_iam_binding" "db_url_access" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.database_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  
  members = [
    "serviceAccount:${google_service_account.prowzi_workload_sa.email}"
  ]
}

# Cloud Functions for lightweight processing (smallest configuration)
resource "google_cloudfunctions_function" "prowzi_evaluator" {
  name    = "prowzi-evaluator"
  project = var.project_id
  region  = var.region
  
  runtime = "go119"  # Lightweight runtime
  
  # Smallest configuration
  available_memory_mb = 128
  timeout             = 60
  max_instances       = 10
  
  # Event trigger for cost efficiency
  event_trigger {
    event_type = "google.pubsub.topic.publish"
    resource   = google_pubsub_topic.prowzi_events.name
  }
  
  # Source code archive (create separately)
  source_archive_bucket = google_storage_bucket.prowzi_backups.name
  source_archive_object = "evaluator-source.zip"
  
  # Entry point
  entry_point = "ProcessEvent"
  
  # Environment variables
  environment_variables = {
    DATABASE_URL = "projects/${var.project_id}/secrets/prowzi-database-url/versions/latest"
  }
}

# Firewall rules for security
resource "google_compute_firewall" "allow_internal" {
  name    = "prowzi-allow-internal"
  project = var.project_id
  network = google_compute_network.prowzi_vpc.name
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080", "5432", "6379"]
  }
  
  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["prowzi"]
}

resource "google_compute_firewall" "deny_all" {
  name    = "prowzi-deny-all"
  project = var.project_id
  network = google_compute_network.prowzi_vpc.name
  
  deny {
    protocol = "all"
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["prowzi"]
  priority      = 1000
}

# Cloud Monitoring alerts for cost control
resource "google_monitoring_alert_policy" "high_cost_alert" {
  display_name = "Prowzi High Cost Alert"
  project      = var.project_id
  combiner     = "OR"
  
  conditions {
    display_name = "High CPU usage"
    
    condition_threshold {
      filter          = "resource.type=\"gce_instance\""
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8
      duration        = "300s"
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = []  # Add notification channels as needed
  
  alert_strategy {
    auto_close = "604800s"  # 7 days
  }
}

# Outputs
output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.prowzi_cluster.endpoint
  sensitive   = true
}

output "database_private_ip" {
  description = "Database private IP address"
  value       = google_sql_database_instance.prowzi_db.private_ip_address
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.prowzi_redis.host
}

output "storage_bucket" {
  description = "Storage bucket name"
  value       = google_storage_bucket.prowzi_backups.name
}

output "workload_identity_sa" {
  description = "Workload Identity service account email"
  value       = google_service_account.prowzi_workload_sa.email
}

# Cost estimation (comments for reference)
# Monthly cost estimate (us-central1):
# - GKE Autopilot: ~$20-50/month (depending on usage)
# - Cloud SQL db-f1-micro: ~$7/month
# - Redis 1GB Basic: ~$25/month
# - Cloud Storage: ~$1-5/month (depending on usage)
# - Secret Manager: ~$1/month
# - Cloud Functions: ~$0-5/month (depending on invocations)
# - Networking: ~$1-5/month
# Total estimated cost: ~$55-93/month