FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY .env .
COPY start_server.sh .

# Make start script executable
RUN chmod +x start_server.sh

# Create data and settings directories (will be overridden by volumes)
RUN mkdir -p data settings

# Expose the port
EXPOSE 8050

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "src.app:app.server", "--workers", "1", "--threads", "8", "--timeout", "120"]