# Multi-stage Docker build for ArXiv sensor service
# Monitors ArXiv for new AI/ML research papers and publications

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY platform/sensors/arxiv/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r arxiv && useradd -r -g arxiv -s /bin/false arxiv

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/arxiv/.local

# Copy source code
COPY platform/sensors/arxiv/src ./src

# Set ownership
RUN chown -R arxiv:arxiv /app /home/arxiv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8087/health || exit 1

# Switch to non-root user
USER arxiv

# Update PATH to include user packages
ENV PATH=/home/arxiv/.local/bin:$PATH

# Expose port
EXPOSE 8087

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.main"]