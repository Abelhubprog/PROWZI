FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Cargo files
COPY agent-runtime/core/Cargo.toml agent-runtime/core/
COPY agent-runtime/orchestrator/Cargo.toml agent-runtime/orchestrator/
COPY Cargo.lock .

# Copy source code
COPY agent-runtime/core/src/ agent-runtime/core/src/
COPY agent-runtime/orchestrator/src/ agent-runtime/orchestrator/src/

# Build the application
WORKDIR /app/agent-runtime/orchestrator
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/prowzi-orchestrator .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ./prowzi-orchestrator --health-check || exit 1

# Expose port
EXPOSE 8081

# Run the application
CMD ["./prowzi-orchestrator"]