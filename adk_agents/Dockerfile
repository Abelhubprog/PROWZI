FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 adk-agent && chown -R adk-agent:adk-agent /app
USER adk-agent

# Environment variables
ENV PYTHONPATH=/app
ENV ADK_UI=disabled
ENV RUST_LOG=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import asyncio; import nats; asyncio.run(nats.connect('${NATS_URL}', connect_timeout=2))" || exit 1

# Run the agent
CMD ["python", "analysis_agent.py"]