#!/bin/bash

# Prowzi Authentication Service Setup Script
# This script sets up the authentication service for development/production

set -e

echo "🚀 Setting up Prowzi Authentication Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        print_error "Rust/Cargo is not installed. Please install from https://rustup.rs/"
        exit 1
    fi
    
    if ! command -v psql &> /dev/null; then
        print_warning "PostgreSQL client not found. Install postgresql-client for database operations."
    fi
    
    if ! command -v redis-cli &> /dev/null; then
        print_warning "Redis client not found. Install redis-tools for Redis operations."
    fi
    
    print_status "Dependencies check completed."
}

# Generate RSA key pairs for JWT tokens
generate_keys() {
    print_status "Generating RSA key pairs for JWT tokens..."
    
    KEYS_DIR="./keys"
    mkdir -p "$KEYS_DIR"
    
    # Generate JWT keys
    if [[ ! -f "$KEYS_DIR/jwt_private.pem" ]]; then
        openssl genrsa -out "$KEYS_DIR/jwt_private.pem" 2048
        openssl rsa -in "$KEYS_DIR/jwt_private.pem" -pubout -out "$KEYS_DIR/jwt_public.pem"
        print_status "JWT keys generated"
    else
        print_warning "JWT keys already exist, skipping generation"
    fi
    
    # Generate refresh token keys
    if [[ ! -f "$KEYS_DIR/refresh_private.pem" ]]; then
        openssl genrsa -out "$KEYS_DIR/refresh_private.pem" 2048
        openssl rsa -in "$KEYS_DIR/refresh_private.pem" -pubout -out "$KEYS_DIR/refresh_public.pem"
        print_status "Refresh token keys generated"
    else
        print_warning "Refresh token keys already exist, skipping generation"
    fi
    
    # Set proper permissions
    chmod 600 "$KEYS_DIR"/*.pem
    print_status "Key permissions set"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        
        # Read the generated keys and populate .env file
        JWT_PRIVATE_KEY=$(cat keys/jwt_private.pem | sed ':a;N;$!ba;s/\n/\\n/g')
        JWT_PUBLIC_KEY=$(cat keys/jwt_public.pem | sed ':a;N;$!ba;s/\n/\\n/g')
        REFRESH_PRIVATE_KEY=$(cat keys/refresh_private.pem | sed ':a;N;$!ba;s/\n/\\n/g')
        REFRESH_PUBLIC_KEY=$(cat keys/refresh_public.pem | sed ':a;N;$!ba;s/\n/\\n/g')
        
        # Replace placeholders in .env file
        sed -i "s|YOUR_JWT_PRIVATE_KEY_HERE|${JWT_PRIVATE_KEY}|g" .env
        sed -i "s|YOUR_JWT_PUBLIC_KEY_HERE|${JWT_PUBLIC_KEY}|g" .env
        sed -i "s|YOUR_REFRESH_PRIVATE_KEY_HERE|${REFRESH_PRIVATE_KEY}|g" .env
        sed -i "s|YOUR_REFRESH_PUBLIC_KEY_HERE|${REFRESH_PUBLIC_KEY}|g" .env
        
        print_status "Environment file created. Please review and update .env with your specific configuration."
        print_warning "Make sure to update DATABASE_URL and REDIS_URL in .env file!"
    else
        print_warning ".env file already exists, skipping creation"
    fi
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    if [[ -z "${DATABASE_URL}" ]]; then
        print_warning "DATABASE_URL not set. Please set it in .env file and run migrations manually:"
        print_warning "  sqlx migrate run --database-url \$DATABASE_URL"
        return
    fi
    
    # Check if sqlx-cli is installed
    if ! command -v sqlx &> /dev/null; then
        print_status "Installing sqlx-cli..."
        cargo install sqlx-cli --features postgres
    fi
    
    # Run migrations
    cd ../../.. # Go to project root
    sqlx migrate run --database-url "$DATABASE_URL"
    cd platform/auth
    
    print_status "Database migrations completed"
}

# Build the service
build_service() {
    print_status "Building authentication service..."
    
    cargo build --release
    
    if [[ $? -eq 0 ]]; then
        print_status "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Create systemd service file (for Linux production deployment)
create_systemd_service() {
    if [[ "$1" == "production" ]]; then
        print_status "Creating systemd service file..."
        
        cat > prowzi-auth.service << EOF
[Unit]
Description=Prowzi Authentication Service
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=prowzi
WorkingDirectory=/opt/prowzi/auth
ExecStart=/opt/prowzi/auth/target/release/prowzi-auth
Restart=always
RestartSec=5
Environment=RUST_LOG=prowzi_auth=info,tower_http=info
EnvironmentFile=/opt/prowzi/auth/.env

[Install]
WantedBy=multi-user.target
EOF
        
        print_status "Systemd service file created: prowzi-auth.service"
        print_warning "Copy this file to /etc/systemd/system/ and run 'systemctl enable prowzi-auth' as root"
    fi
}

# Main setup function
main() {
    print_status "Starting Prowzi Authentication Service Setup"
    
    # Parse command line arguments
    ENVIRONMENT=${1:-development}
    
    case $ENVIRONMENT in
        development|staging|production)
            print_status "Setting up for $ENVIRONMENT environment"
            ;;
        *)
            print_error "Invalid environment. Use: development, staging, or production"
            exit 1
            ;;
    esac
    
    # Run setup steps
    check_dependencies
    generate_keys
    create_env_file
    build_service
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        create_systemd_service production
        print_warning "Additional production setup required:"
        print_warning "1. Set up reverse proxy (nginx/traefik)"
        print_warning "2. Configure TLS certificates"
        print_warning "3. Set up monitoring and logging"
        print_warning "4. Configure firewall rules"
    fi
    
    print_status "Setup completed! 🎉"
    print_status ""
    print_status "Next steps:"
    print_status "1. Review and update .env file with your configuration"
    print_status "2. Ensure PostgreSQL and Redis are running"
    print_status "3. Run database migrations if not done automatically"
    print_status "4. Start the service: cargo run (development) or systemctl start prowzi-auth (production)"
    print_status ""
    print_status "Service will be available at: http://localhost:3001"
}

# Run main function with all arguments
main "$@"
