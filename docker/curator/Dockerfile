FROM node:20-slim as base

# Set environment variables
ENV NODE_ENV=production

# Install dumb-init for proper signal handling
RUN apt-get update && apt-get install -y \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package files
COPY platform/curator/package*.json ./
COPY platform/shared/package*.json ./shared/

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY platform/curator/ .
COPY platform/shared/ ./shared/

# Build the application
RUN npm run build

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:8003/health', (res) => process.exit(res.statusCode === 200 ? 0 : 1))"

# Expose port
EXPOSE 8003

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]