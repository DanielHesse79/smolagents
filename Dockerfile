FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
# Install smolagents with required extras: qdrant, litellm, toolkit
# Also install gradio, docker, and reportlab for PDF generation
RUN pip install --no-cache-dir -e ".[qdrant,litellm,toolkit]" && \
    pip install --no-cache-dir gradio>=5.50.0 docker reportlab

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Create data directory
RUN mkdir -p /app/data /app/models

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Run application
CMD ["python", "examples/gradio_ui.py"]

