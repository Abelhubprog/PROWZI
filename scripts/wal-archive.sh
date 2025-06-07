bash
#!/bin/bash
set -euo pipefail

# This script is called by postgres archive_command
# Usage: wal-archive.sh %p %f

WAL_PATH=$1
WAL_FILE=$2
S3_BUCKET="prowzi-wal-archive-${ENVIRONMENT}"
S3_PREFIX="cluster-${CLUSTER_ID}/wal"

# Compress WAL file
gzip -c "${WAL_PATH}" > "/tmp/${WAL_FILE}.gz"

# Upload to S3 with server-side encryption
aws s3 cp "/tmp/${WAL_FILE}.gz" \
  "s3://${S3_BUCKET}/${S3_PREFIX}/${WAL_FILE}.gz" \
  --sse AES256 \
  --storage-class STANDARD_IA

# Verify upload
aws s3api head-object \
  --bucket "${S3_BUCKET}" \
  --key "${S3_PREFIX}/${WAL_FILE}.gz" > /dev/null

# Clean up
rm -f "/tmp/${WAL_FILE}.gz"

# Log success
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Archived ${WAL_FILE} to S3" >> /var/log/wal-archive.log
DR Test Workflow (.github/workflows/dr-drill.yml):

yaml
name: Disaster Recovery Drill

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:

jobs:
  dr-drill:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
    - uses: actions/checkout@v4

    - name: Setup AWS CLI
      uses: aws-actions/configure-aws-credentials@v4
      with:
        role-to-assume: ${{ secrets.DR_TEST_ROLE_ARN }}
        aws-region: us-west-2  # DR region

    - name: Create test RDS instance
      id: create-rds
      run: |
        INSTANCE_ID="prowzi-dr-test-$(date +%s)"

        aws rds create-db-instance \
          --db-instance-identifier "$INSTANCE_ID" \
          --db-instance-class db.t3.medium \
          --engine postgres \
          --engine-version 15.4 \
          --master-username postgres \
          --master-user-password "${{ secrets.DR_TEST_DB_PASSWORD }}" \
          --allocated-storage 100 \
          --no-publicly-accessible \
          --backup-retention-period 0

        echo "instance_id=$INSTANCE_ID" >> $GITHUB_OUTPUT

    - name: Wait for RDS
      run: |
        aws rds wait db-instance-available \
          --db-instance-identifier "${{ steps.create-rds.outputs.instance_id }}"

    - name: Get RDS endpoint
      id: get-endpoint
      run: |
        ENDPOINT=$(aws rds describe-db-instances \
          --db-instance-identifier "${{ steps.create-rds.outputs.instance_id }}" \
          --query 'DBInstances[0].Endpoint.Address' \
          --output text)

        echo "endpoint=$ENDPOINT" >> $GITHUB_OUTPUT

    - name: Restore from WAL
      run: |
        # Get latest base backup
        LATEST_BACKUP=$(aws s3 ls s3://prowzi-wal-archive-prod/cluster-main/base/ \
          | sort | tail -n 1 | awk '{print $4}')

        # Download and restore
        aws s3 cp "s3://prowzi-wal-archive-prod/cluster-main/base/$LATEST_BACKUP" - \
          | gunzip \
          | PGPASSWORD="${{ secrets.DR_TEST_DB_PASSWORD }}" \
            pg_restore -h "${{ steps.get-endpoint.outputs.endpoint }}" \
            -U postgres -d postgres --no-owner --no-privileges

        # Apply WAL files
        ./scripts/restore-wal.sh \
          "${{ steps.get-endpoint.outputs.endpoint }}" \
          "prowzi-wal-archive-prod" \
          "cluster-main"

        RESTORE_TIME=$SECONDS
        echo "Restore completed in ${RESTORE_TIME} seconds"

        # Fail if restore took too long
        if [ $RESTORE_TIME -gt 900 ]; then
          echo "ERROR: Restore took longer than 15 minutes"
          exit 1
        fi

    - name: Verify data integrity
      run: |
        PGPASSWORD="${{ secrets.DR_TEST_DB_PASSWORD }}" psql \
          -h "${{ steps.get-endpoint.outputs.endpoint }}" \
          -U postgres -d prowzi <<EOF
        -- Check row counts
        SELECT 'events' as table_name, COUNT(*) as row_count FROM prowzi.events
        UNION ALL
        SELECT 'briefs', COUNT(*) FROM prowzi.briefs
        UNION ALL
        SELECT 'missions', COUNT(*) FROM prowzi.missions;

        -- Verify latest data
        SELECT MAX(created_at) as latest_event FROM prowzi.events;

        -- Test query performance
        EXPLAIN ANALYZE
        SELECT * FROM prowzi.events 
        WHERE created_at > NOW() - INTERVAL '1 day'
        LIMIT 100;
        EOF

    - name: Cleanup
      if: always()
      run: |
        aws rds delete-db-instance \
          --db-instance-identifier "${{ steps.create-rds.outputs.instance_id }}" \
          --skip-final-snapshot \
          --delete-automated-backups || true

    - name: Send notification
      if: failure()
      run: |
        curl -X POST "${{ secrets.SLACK_WEBHOOK }}" \
          -H 'Content-Type: application/json' \
          -d '{
            "text": "🚨 DR Drill Failed! Restore or verification issues detected.",
            "color": "danger"
          }'
