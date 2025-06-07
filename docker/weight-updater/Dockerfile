# Multi-stage Docker build for Weight Updater service
# Updates ML model weights and manages model versioning

# Build stage
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY agent-runtime/weight-updater/Cargo.toml agent-runtime/weight-updater/Cargo.lock* ./

# Create dummy source to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src target/release/deps/weight-updater*

# Copy source code
COPY agent-runtime/weight-updater/src ./src

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Create non-root user
RUN groupadd -r weightupdater && useradd -r -g weightupdater -s /bin/false weightupdater

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/weight-updater ./weight-updater

# Set ownership
RUN chown -R weightupdater:weightupdater /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Switch to non-root user
USER weightupdater

# Expose port for health checks and metrics
EXPOSE 8090

# Set entrypoint
ENTRYPOINT ["./weight-updater"]