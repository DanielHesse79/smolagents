FROM python:3.12-slim

# Install system dependencies
# ffmpeg is needed for audio processing in Open Deep Research (pydub)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
# Install intelcore with required extras: qdrant, litellm, toolkit
# Also install gradio, docker, reportlab, scrapy, and Open Deep Research dependencies
RUN pip install --no-cache-dir -e ".[qdrant,litellm,toolkit]" && \
    pip install --no-cache-dir gradio>=5.50.0 docker reportlab scrapy && \
    pip install --no-cache-dir \
        beautifulsoup4>=4.12.3 \
        markdownify>=0.13.1 \
        pathvalidate>=3.2.1 \
        google-search-results>=2.4.2 \
        puremagic>=1.28 \
        tqdm>=4.66.4 \
        mammoth>=1.8.0 \
        pdfminer.six>=20240706 \
        python-pptx>=1.0.2 \
        python-docx>=1.1.0 \
        pandas>=2.2.3 \
        SpeechRecognition \
        pydub \
        youtube-transcript-api>=0.6.2

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

