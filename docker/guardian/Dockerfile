# Multi-stage Docker build for Guardian service
# Guardian monitors system health and enforces operational limits

# Build stage
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY agent-runtime/guardian/Cargo.toml agent-runtime/guardian/Cargo.lock* ./

# Create dummy source to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src target/release/deps/guardian*

# Copy source code
COPY agent-runtime/guardian/src ./src

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Create non-root user
RUN groupadd -r guardian && useradd -r -g guardian -s /bin/false guardian

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/guardian ./guardian

# Set ownership
RUN chown -R guardian:guardian /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8084/health || exit 1

# Switch to non-root user
USER guardian

# Expose port for health checks and metrics
EXPOSE 8084

# Set entrypoint
ENTRYPOINT ["./guardian"]