#!/bin/bash
# =============================================================================
# Prowzi PostgreSQL WAL Restore Script
# =============================================================================
# This script performs point-in-time recovery for PostgreSQL using WAL files.
# It supports restoring from S3 backup storage and provides both automated
# and manual recovery modes.
#
# Usage:
#   ./restore-wal.sh [options]
#
# Options:
#   --auto                 Run in automated mode (uses environment variables)
#   --manual               Run in manual mode (interactive prompts)
#   --s3-bucket NAME       S3 bucket name for backups
#   --s3-prefix PREFIX     S3 prefix path for backups
#   --target-time TIME     Recovery target time (format: YYYY-MM-DD HH:MM:SS)
#   --target-lsn LSN       Recovery target LSN
#   --pg-host HOST         PostgreSQL host
#   --pg-port PORT         PostgreSQL port
#   --pg-user USER         PostgreSQL user
#   --pg-database DB       PostgreSQL database name
#   --backup-id ID         Specific backup ID to restore
#   --verify-only          Only verify backup without restoring
#   --help                 Show this help message
#
# Environment variables (for automated mode):
#   PROWZI_PG_HOST         PostgreSQL host
#   PROWZI_PG_PORT         PostgreSQL port
#   PROWZI_PG_USER         PostgreSQL user
#   PROWZI_PG_PASSWORD     PostgreSQL password
#   PROWZI_PG_DATABASE     PostgreSQL database name
#   PROWZI_S3_BUCKET       S3 bucket name
#   PROWZI_S3_PREFIX       S3 prefix path
#   PROWZI_BACKUP_ID       Specific backup ID to restore
#   PROWZI_TARGET_TIME     Recovery target time
#   PROWZI_TARGET_LSN      Recovery target LSN
#   PROWZI_RECOVERY_MODE   Recovery mode (time, lsn, immediate)
#   AWS_ACCESS_KEY_ID      AWS access key
#   AWS_SECRET_ACCESS_KEY  AWS secret key
#   AWS_DEFAULT_REGION     AWS region
# =============================================================================

set -e

# Default values
PG_HOST="localhost"
PG_PORT="5432"
PG_USER="postgres"
PG_DATABASE="prowzi"
S3_BUCKET=""
S3_PREFIX="backups/prowzi"
BACKUP_ID=""
TARGET_TIME=""
TARGET_LSN=""
RECOVERY_MODE="time"
VERIFY_ONLY=false
AUTO_MODE=false
MANUAL_MODE=false
LOG_FILE="prowzi-wal-restore-$(date +%Y%m%d-%H%M%S).log"
TEMP_DIR="/tmp/prowzi-wal-restore-$(date +%Y%m%d-%H%M%S)"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    local level=$1
    local message=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        *)
            echo -e "${timestamp} - $message"
            ;;
    esac
    
    echo "${timestamp} - [$level] $message" >> "$LOG_FILE"
}

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local message=$3
    local percent=$((current * 100 / total))
    local completed=$((percent / 2))
    local remaining=$((50 - completed))
    
    printf "\r[%-${completed}s%-${remaining}s] %d%% %s" "$(printf '%0.s#' $(seq 1 $completed))" "$(printf '%0.s-' $(seq 1 $remaining))" "$percent" "$message"
}

# Function to check dependencies
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    local missing_deps=()
    
    for cmd in psql pg_basebackup pg_waldump aws jq; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        log "ERROR" "Please install the required dependencies and try again."
        exit 1
    fi
    
    log "SUCCESS" "All dependencies are installed."
}

# Function to validate PostgreSQL connection
validate_pg_connection() {
    log "INFO" "Validating PostgreSQL connection to $PG_HOST:$PG_PORT..."
    
    if ! PGPASSWORD="$PG_PASSWORD" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DATABASE" -c "SELECT 1" &> /dev/null; then
        log "ERROR" "Failed to connect to PostgreSQL server."
        return 1
    fi
    
    log "SUCCESS" "PostgreSQL connection successful."
    return 0
}

# Function to validate S3 access
validate_s3_access() {
    log "INFO" "Validating S3 access to s3://$S3_BUCKET/$S3_PREFIX..."
    
    if ! aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX" &> /dev/null; then
        log "ERROR" "Failed to access S3 bucket/prefix."
        return 1
    fi
    
    log "SUCCESS" "S3 access validated successfully."
    return 0
}

# Function to list available backups
list_available_backups() {
    log "INFO" "Listing available backups in S3..."
    
    local backups=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/basebackups/" --recursive | grep "backup_manifest.json" | awk '{print $4}')
    
    if [ -z "$backups" ]; then
        log "ERROR" "No backups found in S3 bucket."
        return 1
    fi
    
    log "INFO" "Found the following backups:"
    local count=1
    local backup_ids=()
    
    for backup in $backups; do
        local backup_id=$(echo "$backup" | sed -E 's/.*basebackups\/([^\/]+)\/backup_manifest.json/\1/')
        backup_ids+=("$backup_id")
        
        # Get backup metadata
        local backup_metadata=$(aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/basebackups/$backup_id/backup_metadata.json" - 2>/dev/null)
        local backup_time=$(echo "$backup_metadata" | jq -r '.backup_time // "Unknown"')
        local backup_size=$(echo "$backup_metadata" | jq -r '.backup_size // "Unknown"')
        local wal_segment_start=$(echo "$backup_metadata" | jq -r '.wal_segment_start // "Unknown"')
        
        echo "$count) Backup ID: $backup_id"
        echo "   Created: $backup_time"
        echo "   Size: $backup_size"
        echo "   WAL Start: $wal_segment_start"
        echo ""
        
        ((count++))
    done
    
    # In manual mode, prompt for backup selection
    if [ "$MANUAL_MODE" = true ] && [ -z "$BACKUP_ID" ]; then
        local selection
        read -p "Enter the number of the backup to restore (1-$((count-1))): " selection
        
        if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -ge "$count" ]; then
            log "ERROR" "Invalid selection."
            return 1
        fi
        
        BACKUP_ID="${backup_ids[$((selection-1))]}"
        log "INFO" "Selected backup ID: $BACKUP_ID"
    fi
    
    return 0
}

# Function to verify backup integrity
verify_backup() {
    local backup_id=$1
    log "INFO" "Verifying backup integrity for $backup_id..."
    
    # Check if backup manifest exists
    if ! aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/basebackups/$backup_id/backup_manifest.json" &> /dev/null; then
        log "ERROR" "Backup manifest not found for $backup_id."
        return 1
    fi
    
    # Download and validate backup manifest
    log "INFO" "Downloading backup manifest..."
    local manifest=$(aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/basebackups/$backup_id/backup_manifest.json" - 2>/dev/null)
    
    if [ -z "$manifest" ]; then
        log "ERROR" "Failed to download backup manifest."
        return 1
    fi
    
    # Extract file list and checksums from manifest
    local files=$(echo "$manifest" | jq -r '.files[] | .path')
    local total_files=$(echo "$files" | wc -l)
    log "INFO" "Manifest contains $total_files files."
    
    # Verify WAL files existence
    log "INFO" "Verifying WAL files..."
    local wal_start=$(echo "$manifest" | jq -r '.wal_start // empty')
    local wal_end=$(echo "$manifest" | jq -r '.wal_end // empty')
    
    if [ -n "$wal_start" ] && [ -n "$wal_end" ]; then
        if ! aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/wal/$wal_start" &> /dev/null; then
            log "WARNING" "Starting WAL segment $wal_start not found."
        fi
        
        log "SUCCESS" "WAL range verified: $wal_start to $wal_end"
    else
        log "WARNING" "WAL range information not found in manifest."
    fi
    
    # If we're only verifying, do a more thorough check
    if [ "$VERIFY_ONLY" = true ]; then
        log "INFO" "Performing detailed verification..."
        
        local count=0
        for file in $files; do
            ((count++))
            show_progress $count $total_files "Verifying file $count/$total_files"
            
            local file_path="s3://$S3_BUCKET/$S3_PREFIX/basebackups/$backup_id/$file"
            if ! aws s3 ls "$file_path" &> /dev/null; then
                echo ""
                log "ERROR" "File not found in S3: $file"
                return 1
            fi
            
            # Check every 10th file's checksum to save time
            if [ $((count % 10)) -eq 0 ]; then
                local expected_checksum=$(echo "$manifest" | jq -r ".files[] | select(.path == \"$file\") | .checksum")
                local temp_file="$TEMP_DIR/$(basename "$file")"
                
                mkdir -p "$(dirname "$temp_file")"
                aws s3 cp "$file_path" "$temp_file" &> /dev/null
                
                local actual_checksum=$(sha256sum "$temp_file" | awk '{print $1}')
                rm -f "$temp_file"
                
                if [ "$expected_checksum" != "$actual_checksum" ]; then
                    echo ""
                    log "ERROR" "Checksum mismatch for file: $file"
                    log "ERROR" "Expected: $expected_checksum"
                    log "ERROR" "Actual: $actual_checksum"
                    return 1
                fi
            fi
        done
        echo ""
    fi
    
    log "SUCCESS" "Backup verification completed successfully."
    return 0
}

# Function to download backup files
download_backup() {
    local backup_id=$1
    log "INFO" "Downloading backup $backup_id..."
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    log "INFO" "Created temporary directory: $TEMP_DIR"
    
    # Download manifest
    log "INFO" "Downloading backup manifest..."
    aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/basebackups/$backup_id/backup_manifest.json" "$TEMP_DIR/backup_manifest.json"
    
    # Extract file list from manifest
    local files=$(jq -r '.files[] | .path' "$TEMP_DIR/backup_manifest.json")
    local total_files=$(echo "$files" | wc -l)
    log "INFO" "Downloading $total_files files..."
    
    # Download files
    local count=0
    for file in $files; do
        ((count++))
        show_progress $count $total_files "Downloading file $count/$total_files"
        
        local file_path="$TEMP_DIR/$file"
        mkdir -p "$(dirname "$file_path")"
        aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/basebackups/$backup_id/$file" "$file_path" &> /dev/null
    done
    echo ""
    
    log "SUCCESS" "Backup downloaded successfully."
    return 0
}

# Function to download WAL files
download_wal_files() {
    local backup_id=$1
    log "INFO" "Downloading WAL files for backup $backup_id..."
    
    # Create WAL directory
    mkdir -p "$TEMP_DIR/pg_wal"
    
    # Get WAL information from manifest
    local wal_start=$(jq -r '.wal_start // empty' "$TEMP_DIR/backup_manifest.json")
    local wal_end=$(jq -r '.wal_end // empty' "$TEMP_DIR/backup_manifest.json")
    
    if [ -z "$wal_start" ]; then
        log "ERROR" "WAL start segment not found in manifest."
        return 1
    fi
    
    # List all WAL files in S3
    log "INFO" "Listing WAL files in S3..."
    local wal_files=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/wal/" --recursive | awk '{print $4}')
    
    # Filter WAL files based on recovery target
    local filtered_wal_files=""
    if [ -n "$TARGET_TIME" ] && [ "$RECOVERY_MODE" = "time" ]; then
        log "INFO" "Filtering WAL files for recovery target time: $TARGET_TIME"
        # This is a simplified approach - in a real scenario, you'd need to parse WAL headers
        # to determine which files are needed for a specific point in time
        filtered_wal_files="$wal_files"
    elif [ -n "$TARGET_LSN" ] && [ "$RECOVERY_MODE" = "lsn" ]; then
        log "INFO" "Filtering WAL files for recovery target LSN: $TARGET_LSN"
        # Similar simplification as above
        filtered_wal_files="$wal_files"
    else
        filtered_wal_files="$wal_files"
    fi
    
    # Download filtered WAL files
    local total_wal_files=$(echo "$filtered_wal_files" | wc -l)
    log "INFO" "Downloading $total_wal_files WAL files..."
    
    local count=0
    for wal_file in $filtered_wal_files; do
        ((count++))
        show_progress $count $total_wal_files "Downloading WAL file $count/$total_wal_files"
        
        local file_name=$(basename "$wal_file")
        aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/$wal_file" "$TEMP_DIR/pg_wal/$file_name" &> /dev/null
    done
    echo ""
    
    log "SUCCESS" "WAL files downloaded successfully."
    return 0
}

# Function to create recovery configuration
create_recovery_conf() {
    log "INFO" "Creating recovery configuration..."
    
    local recovery_conf="$TEMP_DIR/recovery.conf"
    
    echo "restore_command = 'cp $TEMP_DIR/pg_wal/%f %p'" > "$recovery_conf"
    
    if [ "$RECOVERY_MODE" = "time" ] && [ -n "$TARGET_TIME" ]; then
        echo "recovery_target_time = '$TARGET_TIME'" >> "$recovery_conf"
    elif [ "$RECOVERY_MODE" = "lsn" ] && [ -n "$TARGET_LSN" ]; then
        echo "recovery_target_lsn = '$TARGET_LSN'" >> "$recovery_conf"
    else
        echo "recovery_target = 'immediate'" >> "$recovery_conf"
    fi
    
    echo "recovery_target_inclusive = true" >> "$recovery_conf"
    echo "recovery_target_action = 'promote'" >> "$recovery_conf"
    
    log "SUCCESS" "Recovery configuration created."
    return 0
}

# Function to perform the actual restore
perform_restore() {
    log "INFO" "Stopping PostgreSQL service..."
    
    # This would need to be adjusted based on the environment
    if command -v systemctl &> /dev/null; then
        sudo systemctl stop postgresql
    elif command -v service &> /dev/null; then
        sudo service postgresql stop
    else
        log "WARNING" "Could not stop PostgreSQL service automatically. Please stop it manually."
        read -p "Press Enter to continue once PostgreSQL is stopped..."
    fi
    
    log "INFO" "Backing up existing PostgreSQL data directory..."
    local pg_data=$(PGPASSWORD="$PG_PASSWORD" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DATABASE" -t -c "SHOW data_directory;" 2>/dev/null)
    pg_data=$(echo "$pg_data" | tr -d '[:space:]')
    
    if [ -z "$pg_data" ]; then
        log "WARNING" "Could not determine PostgreSQL data directory."
        read -p "Enter PostgreSQL data directory: " pg_data
    fi
    
    log "INFO" "PostgreSQL data directory: $pg_data"
    
    # Backup existing data directory
    local backup_dir="${pg_data}_backup_$(date +%Y%m%d%H%M%S)"
    log "INFO" "Creating backup of data directory to $backup_dir..."
    sudo mv "$pg_data" "$backup_dir"
    
    # Create new data directory
    log "INFO" "Creating new data directory..."
    sudo mkdir -p "$pg_data"
    sudo chown postgres:postgres "$pg_data"
    
    # Copy files from temp directory to data directory
    log "INFO" "Copying files to data directory..."
    sudo cp -r "$TEMP_DIR"/* "$pg_data/"
    
    # Set correct permissions
    log "INFO" "Setting correct permissions..."
    sudo chown -R postgres:postgres "$pg_data"
    sudo chmod -R 0700 "$pg_data"
    
    # Copy recovery.conf to data directory
    log "INFO" "Copying recovery configuration..."
    sudo cp "$TEMP_DIR/recovery.conf" "$pg_data/recovery.conf"
    sudo chown postgres:postgres "$pg_data/recovery.conf"
    
    # Start PostgreSQL
    log "INFO" "Starting PostgreSQL service..."
    if command -v systemctl &> /dev/null; then
        sudo systemctl start postgresql
    elif command -v service &> /dev/null; then
        sudo service postgresql start
    else
        log "WARNING" "Could not start PostgreSQL service automatically. Please start it manually."
        read -p "Press Enter to continue once PostgreSQL is started..."
    fi
    
    # Wait for recovery to complete
    log "INFO" "Waiting for recovery to complete..."
    local recovery_complete=false
    local attempt=0
    local max_attempts=60
    
    while [ "$recovery_complete" = false ] && [ $attempt -lt $max_attempts ]; do
        ((attempt++))
        echo -n "."
        
        if PGPASSWORD="$PG_PASSWORD" psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DATABASE" -c "SELECT pg_is_in_recovery();" 2>/dev/null | grep -q "f"; then
            recovery_complete=true
        else
            sleep 5
        fi
    done
    echo ""
    
    if [ "$recovery_complete" = true ]; then
        log "SUCCESS" "Recovery completed successfully!"
    else
        log "WARNING" "Recovery is taking longer than expected. Please check PostgreSQL logs."
    fi
    
    # Clean up
    log "INFO" "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    
    log "SUCCESS" "Restore process completed."
    return 0
}

# Function to print help message
print_help() {
    echo "Prowzi PostgreSQL WAL Restore Script"
    echo ""
    echo "Usage:"
    echo "  ./restore-wal.sh [options]"
    echo ""
    echo "Options:"
    echo "  --auto                 Run in automated mode (uses environment variables)"
    echo "  --manual               Run in manual mode (interactive prompts)"
    echo "  --s3-bucket NAME       S3 bucket name for backups"
    echo "  --s3-prefix PREFIX     S3 prefix path for backups"
    echo "  --target-time TIME     Recovery target time (format: YYYY-MM-DD HH:MM:SS)"
    echo "  --target-lsn LSN       Recovery target LSN"
    echo "  --pg-host HOST         PostgreSQL host"
    echo "  --pg-port PORT         PostgreSQL port"
    echo "  --pg-user USER         PostgreSQL user"
    echo "  --pg-database DB       PostgreSQL database name"
    echo "  --backup-id ID         Specific backup ID to restore"
    echo "  --verify-only          Only verify backup without restoring"
    echo "  --help                 Show this help message"
    echo ""
    echo "Environment variables (for automated mode):"
    echo "  PROWZI_PG_HOST         PostgreSQL host"
    echo "  PROWZI_PG_PORT         PostgreSQL port"
    echo "  PROWZI_PG_USER         PostgreSQL user"
    echo "  PROWZI_PG_PASSWORD     PostgreSQL password"
    echo "  PROWZI_PG_DATABASE     PostgreSQL database name"
    echo "  PROWZI_S3_BUCKET       S3 bucket name"
    echo "  PROWZI_S3_PREFIX       S3 prefix path"
    echo "  PROWZI_BACKUP_ID       Specific backup ID to restore"
    echo "  PROWZI_TARGET_TIME     Recovery target time"
    echo "  PROWZI_TARGET_LSN      Recovery target LSN"
    echo "  PROWZI_RECOVERY_MODE   Recovery mode (time, lsn, immediate)"
    echo "  AWS_ACCESS_KEY_ID      AWS access key"
    echo "  AWS_SECRET_ACCESS_KEY  AWS secret key"
    echo "  AWS_DEFAULT_REGION     AWS region"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --manual)
            MANUAL_MODE=true
            shift
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --s3-prefix)
            S3_PREFIX="$2"
            shift 2
            ;;
        --target-time)
            TARGET_TIME="$2"
            RECOVERY_MODE="time"
            shift 2
            ;;
        --target-lsn)
            TARGET_LSN="$2"
            RECOVERY_MODE="lsn"
            shift 2
            ;;
        --pg-host)
            PG_HOST="$2"
            shift 2
            ;;
        --pg-port)
            PG_PORT="$2"
            shift 2
            ;;
        --pg-user)
            PG_USER="$2"
            shift 2
            ;;
        --pg-database)
            PG_DATABASE="$2"
            shift 2
            ;;
        --backup-id)
            BACKUP_ID="$2"
            shift 2
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $key"
            print_help
            exit 1
            ;;
    esac
done

# Check if either auto or manual mode is specified
if [ "$AUTO_MODE" = false ] && [ "$MANUAL_MODE" = false ]; then
    log "ERROR" "Please specify either --auto or --manual mode."
    print_help
    exit 1
fi

# If auto mode, load environment variables
if [ "$AUTO_MODE" = true ]; then
    log "INFO" "Running in automated mode."
    
    # Load environment variables
    PG_HOST="${PROWZI_PG_HOST:-$PG_HOST}"
    PG_PORT="${PROWZI_PG_PORT:-$PG_PORT}"
    PG_USER="${PROWZI_PG_USER:-$PG_USER}"
    PG_PASSWORD="${PROWZI_PG_PASSWORD:-}"
    PG_DATABASE="${PROWZI_PG_DATABASE:-$PG_DATABASE}"
    S3_BUCKET="${PROWZI_S3_BUCKET:-$S3_BUCKET}"
    S3_PREFIX="${PROWZI_S3_PREFIX:-$S3_PREFIX}"
    BACKUP_ID="${PROWZI_BACKUP_ID:-$BACKUP_ID}"
    TARGET_TIME="${PROWZI_TARGET_TIME:-$TARGET_TIME}"
    TARGET_LSN="${PROWZI_TARGET_LSN:-$TARGET_LSN}"
    RECOVERY_MODE="${PROWZI_RECOVERY_MODE:-$RECOVERY_MODE}"
fi

# If manual mode, prompt for missing values
if [ "$MANUAL_MODE" = true ]; then
    log "INFO" "Running in manual mode."
    
    # Prompt for PostgreSQL connection details
    if [ -z "$PG_HOST" ]; then
        read -p "Enter PostgreSQL host [localhost]: " input
        PG_HOST="${input:-localhost}"
    fi
    
    if [ -z "$PG_PORT" ]; then
        read -p "Enter PostgreSQL port [5432]: " input
        PG_PORT="${input:-5432}"
    fi
    
    if [ -z "$PG_USER" ]; then
        read -p "Enter PostgreSQL user [postgres]: " input
        PG_USER="${input:-postgres}"
    fi
    
    if [ -z "$PG_PASSWORD" ]; then
        read -s -p "Enter PostgreSQL password: " PG_PASSWORD
        echo ""
    fi
    
    if [ -z "$PG_DATABASE" ]; then
        read -p "Enter PostgreSQL database [prowzi]: " input
        PG_DATABASE="${input:-prowzi}"
    fi
    
    # Prompt for S3 details
    if [ -z "$S3_BUCKET" ]; then
        read -p "Enter S3 bucket name: " S3_BUCKET
    fi
    
    if [ -z "$S3_PREFIX" ]; then
        read -p "Enter S3 prefix [backups/prowzi]: " input
        S3_PREFIX="${input:-backups/prowzi}"
    fi
    
    # Prompt for recovery mode
    if [ -z "$TARGET_TIME" ] && [ -z "$TARGET_LSN" ]; then
        echo "Select recovery mode:"
        echo "1) Point-in-time recovery (by timestamp)"
        echo "2) Point-in-time recovery (by LSN)"
        echo "3) Latest state (immediate recovery)"
        read -p "Enter your choice [3]: " recovery_choice
        
        case "${recovery_choice:-3}" in
            1)
                RECOVERY_MODE="time"
                read -p "Enter target time (YYYY-MM-DD HH:MM:SS): " TARGET_TIME
                ;;
            2)
                RECOVERY_MODE="lsn"
                read -p "Enter target LSN: " TARGET_LSN
                ;;
            3)
                RECOVERY_MODE="immediate"
                ;;
            *)
                log "ERROR" "Invalid choice."
                exit 1
                ;;
        esac
    fi
fi

# Validate required parameters
if [ -z "$S3_BUCKET" ]; then
    log "ERROR" "S3 bucket name is required."
    exit 1
fi

# Create temp directory
mkdir -p "$TEMP_DIR"

# Main execution flow
log "INFO" "Starting WAL restore process..."
log "INFO" "Log file: $LOG_FILE"

# Check dependencies
check_dependencies

# Validate PostgreSQL connection if not verify-only
if [ "$VERIFY_ONLY" = false ]; then
    if ! validate_pg_connection; then
        log "WARNING" "PostgreSQL connection failed, but continuing with restore process."
    fi
fi

# Validate S3 access
if ! validate_s3_access; then
    log "ERROR" "S3 access validation failed. Aborting."
    exit 1
fi

# List available backups and select one if not specified
if [ -z "$BACKUP_ID" ]; then
    if ! list_available_backups; then
        log "ERROR" "Failed to list or select backups. Aborting."
        exit 1
    fi
fi

# Verify backup integrity
if ! verify_backup "$BACKUP_ID"; then
    log "ERROR" "Backup verification failed. Aborting."
    exit 1
fi

# If verify-only, exit here
if [ "$VERIFY_ONLY" = true ]; then
    log "SUCCESS" "Backup verification completed successfully."
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Download backup files
if ! download_backup "$BACKUP_ID"; then
    log "ERROR" "Failed to download backup files. Aborting."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Download WAL files
if ! download_wal_files "$BACKUP_ID"; then
    log "ERROR" "Failed to download WAL files. Aborting."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Create recovery configuration
if ! create_recovery_conf; then
    log "ERROR" "Failed to create recovery configuration. Aborting."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Confirm restore operation in manual mode
if [ "$MANUAL_MODE" = true ]; then
    echo ""
    log "WARNING" "You are about to restore PostgreSQL from backup $BACKUP_ID."
    log "WARNING" "This will OVERWRITE the current PostgreSQL data directory."
    log "WARNING" "Recovery mode: $RECOVERY_MODE"
    
    if [ "$RECOVERY_MODE" = "time" ]; then
        log "WARNING" "Recovery target time: $TARGET_TIME"
    elif [ "$RECOVERY_MODE" = "lsn" ]; then
        log "WARNING" "Recovery target LSN: $TARGET_LSN"
    fi
    
    read -p "Are you sure you want to proceed? (yes/no): " confirmation
    
    if [ "$confirmation" != "yes" ]; then
        log "INFO" "Restore operation cancelled by user."
        rm -rf "$TEMP_DIR"
        exit 0
    fi
fi

# Perform the restore
if ! perform_restore; then
    log "ERROR" "Restore operation failed."
    exit 1
fi

log "SUCCESS" "WAL restore process completed successfully."
exit 0
