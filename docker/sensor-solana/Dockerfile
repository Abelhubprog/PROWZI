# Multi-stage Docker build for Solana sensor service
# Monitors Solana blockchain for new token launches, DeFi protocols, and transactions

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY platform/sensors/solana/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r solana && useradd -r -g solana -s /bin/false solana

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/solana/.local

# Copy source code
COPY platform/sensors/solana/src ./src

# Set ownership
RUN chown -R solana:solana /app /home/solana

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8089/health || exit 1

# Switch to non-root user
USER solana

# Update PATH to include user packages
ENV PATH=/home/solana/.local/bin:$PATH

# Expose port
EXPOSE 8089

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.main"]