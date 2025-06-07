# Multi-stage Docker build for Notifier service
# Notifier handles alerts, notifications, and communication channels

# Build stage
FROM golang:1.21-alpine as builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates

# Copy dependency manifests
COPY platform/notifier/go.mod platform/notifier/go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY platform/notifier/ ./

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o notifier .

# Runtime stage
FROM alpine:latest

# Create non-root user
RUN addgroup -g 1001 -S notifier && \
    adduser -S notifier -u 1001 -G notifier

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates

WORKDIR /app

# Copy built application
COPY --from=builder /app/notifier ./notifier

# Set ownership
RUN chown -R notifier:notifier /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8086/health || exit 1

# Switch to non-root user
USER notifier

# Expose port
EXPOSE 8086

# Set entrypoint
ENTRYPOINT ["./notifier"]