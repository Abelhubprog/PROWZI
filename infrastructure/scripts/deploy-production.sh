#!/bin/bash

# Prowzi Production Deployment Script
# Deploys the complete Prowzi platform to production environment

set -euo pipefail

# Configuration
NAMESPACE="prowzi-platform"
REGISTRY="ghcr.io/prowzi"
VERSION="${DEPLOY_VERSION:-latest}"
ENVIRONMENT="${DEPLOY_ENV:-production}"
REGION="${AWS_REGION:-us-west-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check required tools
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed"; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "helm is required but not installed"; exit 1; }
    command -v aws >/dev/null 2>&1 || { log_error "AWS CLI is required but not installed"; exit 1; }
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check kubectl context
    CURRENT_CONTEXT=$(kubectl config current-context)
    if [[ "$CURRENT_CONTEXT" != *"$ENVIRONMENT"* ]]; then
        log_warning "Current kubectl context: $CURRENT_CONTEXT"
        read -p "Continue with this context? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Pre-flight checks passed"
}

# Database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Apply schema migrations
    kubectl apply -f migrations/20250602_row_level_security.sql
    kubectl apply -f migrations/add_evi_weights.sql
    
    # Verify migration status
    kubectl exec -n $NAMESPACE deployment/postgres -- psql -U prowzi -d prowzi -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 5;"
    
    log_success "Database migrations completed"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy secrets
    kubectl apply -f infrastructure/k8s/base/secrets.yaml -n $NAMESPACE
    
    # Deploy PostgreSQL
    helm upgrade --install postgres bitnami/postgresql \
        --namespace $NAMESPACE \
        --values infrastructure/charts/prowzi/values/postgres.yaml \
        --version 12.12.10
    
    # Deploy Redis
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --values infrastructure/charts/prowzi/values/redis.yaml \
        --version 18.1.5
    
    # Deploy Pulsar
    helm upgrade --install pulsar apache/pulsar \
        --namespace $NAMESPACE \
        --values infrastructure/charts/prowzi/values/pulsar.yaml \
        --version 3.0.0
    
    log_success "Infrastructure components deployed"
}

# Deploy platform services
deploy_platform() {
    log_info "Deploying platform services..."
    
    # Deploy Prowzi platform using Helm
    helm upgrade --install prowzi-platform infrastructure/charts/prowzi \
        --namespace $NAMESPACE \
        --set image.tag=$VERSION \
        --set environment=$ENVIRONMENT \
        --set region=$REGION \
        --values infrastructure/charts/prowzi/values/$ENVIRONMENT.yaml \
        --wait \
        --timeout=600s
    
    log_success "Platform services deployed"
}

# Deploy agent runtime
deploy_agents() {
    log_info "Deploying agent runtime..."
    
    # Deploy agent components
    kubectl apply -f infrastructure/k8s/deployments/ -n $NAMESPACE
    
    # Wait for agents to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/agent-orchestrator -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/agent-evaluator -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/agent-guardian -n $NAMESPACE
    
    log_success "Agent runtime deployed"
}

# Deploy sensors
deploy_sensors() {
    log_info "Deploying sensors..."
    
    # Deploy sensor components
    kubectl apply -f infrastructure/k8s/deployments/sensors/ -n $NAMESPACE
    
    # Wait for sensors to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/sensor-solana -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/sensor-arxiv -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/sensor-github -n $NAMESPACE
    
    log_success "Sensors deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values infrastructure/monitoring/prometheus/values.yaml \
        --version 51.2.0
    
    # Deploy Grafana dashboards
    kubectl apply -f infrastructure/monitoring/grafana/dashboards/ -n monitoring
    
    log_success "Monitoring stack deployed"
}

# Run health checks
health_checks() {
    log_info "Running health checks..."
    
    # Check service health
    kubectl get pods -n $NAMESPACE
    
    # Check gateway health
    GATEWAY_URL=$(kubectl get service prowzi-gateway -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if curl -f "http://$GATEWAY_URL/health" >/dev/null 2>&1; then
        log_success "Gateway health check passed"
    else
        log_error "Gateway health check failed"
        exit 1
    fi
    
    # Check database connectivity
    kubectl exec -n $NAMESPACE deployment/postgres -- pg_isready -U prowzi
    
    # Check Redis connectivity
    kubectl exec -n $NAMESPACE deployment/redis -- redis-cli ping
    
    log_success "All health checks passed"
}

# Update DNS and certificates
update_dns() {
    log_info "Updating DNS and certificates..."
    
    # Update Route53 records
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z1234567890ABC \
        --change-batch file://infrastructure/dns/production-records.json
    
    # Deploy cert-manager if not exists
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
    
    # Apply SSL certificates
    kubectl apply -f infrastructure/k8s/base/certificates.yaml -n $NAMESPACE
    
    log_success "DNS and certificates updated"
}

# Deployment summary
deployment_summary() {
    log_info "Deployment Summary:"
    echo "=================================="
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Region: $REGION"
    echo "Namespace: $NAMESPACE"
    echo "=================================="
    
    # Show deployment status
    kubectl get deployments -n $NAMESPACE
    kubectl get services -n $NAMESPACE
    
    log_success "Deployment completed successfully!"
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    # Rollback Helm releases
    helm rollback prowzi-platform 0 -n $NAMESPACE
    
    # Rollback database migrations if needed
    # kubectl exec -n $NAMESPACE deployment/postgres -- psql -U prowzi -d prowzi -c "SELECT rollback_migration();"
    
    log_success "Rollback completed"
}

# Main deployment flow
main() {
    log_info "Starting Prowzi production deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    # Set trap for cleanup on error
    trap 'log_error "Deployment failed"; rollback; exit 1' ERR
    
    preflight_checks
    run_migrations
    deploy_infrastructure
    
    # Wait for infrastructure to be ready
    sleep 30
    
    deploy_platform
    deploy_agents
    deploy_sensors
    deploy_monitoring
    update_dns
    
    # Wait for all services to stabilize
    sleep 60
    
    health_checks
    deployment_summary
}

# Handle command line arguments
case "${1:-}" in
    "rollback")
        rollback
        ;;
    "health-check")
        health_checks
        ;;
    "infrastructure-only")
        preflight_checks
        deploy_infrastructure
        ;;
    *)
        main
        ;;
esac