# Multi-stage Docker build for GitHub sensor service
# Monitors GitHub repositories for AI/ML projects, commits, and releases

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY platform/sensors/github/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r github && useradd -r -g github -s /bin/false github

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/github/.local

# Copy source code
COPY platform/sensors/github/src ./src

# Set ownership
RUN chown -R github:github /app /home/github

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8088/health || exit 1

# Switch to non-root user
USER github

# Update PATH to include user packages
ENV PATH=/home/github/.local/bin:$PATH

# Expose port
EXPOSE 8088

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.main"]