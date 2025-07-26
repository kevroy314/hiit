FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_dash.txt .
RUN pip install --no-cache-dir -r requirements_dash.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data settings

# Expose the port
EXPOSE 8050

# Command to run the application
CMD ["python", "app_dash.py"]