# Docker Setup Guide

This guide explains how to run "Daniel's Army of Agents" using Docker Desktop with docker-compose.

## Prerequisites

- **Docker Desktop** installed and running
  - Windows: [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
  - macOS: [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
  - Linux: Docker Engine and Docker Compose
- **GPU Support (Optional)**: For GPU acceleration with Ollama, ensure Docker Desktop has GPU support enabled (WSL2 backend on Windows)

## Quick Start

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd smolagents
   ```

2. **Build and start all services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Open your browser and navigate to: `http://localhost:7860`
   - The Gradio UI should be accessible

4. **Stop the services**:
   ```bash
   docker-compose down
   ```

## Architecture

The Docker setup consists of three services:

- **app**: The Gradio UI application (port 7860)
- **qdrant**: Vector database for persistent memory (ports 6333, 6334)
- **ollama**: LLM service for running local models (port 11434)

All services communicate through a Docker network (`smolagents-network`).

## Services

### Application Container

The main application container runs the Gradio UI with all required dependencies.

**Ports:**
- `7860`: Gradio web interface

**Volumes:**
- `app_data`: SQLite database and application data
- `app_models`: Model preferences and configurations

### Qdrant Container

Qdrant vector database for storing embeddings and agent memory.

**Ports:**
- `6333`: HTTP API
- `6334`: gRPC API

**Volumes:**
- `qdrant_data`: Persistent vector database storage

### Ollama Container

Ollama service for running local LLM models.

**Ports:**
- `11434`: Ollama API

**Volumes:**
- `ollama_data`: Persistent model storage (`/root/.ollama`)

## Environment Variables

The application uses environment variables for configuration. These are set in `docker-compose.yml`:

- `OLLAMA_BASE_URL`: Ollama service URL (default: `http://ollama:11434`)
- `QDRANT_URL`: Qdrant service hostname (default: `qdrant`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)
- `SQLITE_DB_PATH`: SQLite database path (default: `/app/data/publications.db`)
- `OLLAMA_REQUIRED_MODELS`: Optional comma-separated list of specific Ollama models to check for (if not set, system auto-selects best models from available)
- `GRADIO_SERVER_PORT`: Gradio server port (default: `7860`)
- `PYTHONUNBUFFERED`: Python output buffering (default: `1`)
- `PYTHONIOENCODING`: Python I/O encoding (default: `utf-8`)

## Data Persistence

All data is stored in Docker named volumes, which persist across container restarts:

- **qdrant_data**: Qdrant vector database
- **ollama_data**: Ollama models and cache
- **app_data**: SQLite database and application files
- **app_models**: Model preferences and configurations

To view volumes:
```bash
docker volume ls
```

To inspect a volume:
```bash
docker volume inspect smolagents_qdrant_data
```

## GPU Support

**Tip: GPU acceleration requires NVIDIA GPU with CUDA support**

To enable GPU support for Ollama (recommended for better performance):

### Prerequisites

1. **NVIDIA GPU with CUDA support** - Verify with:
   ```bash
   nvidia-smi
   ```

2. **NVIDIA Container Toolkit** - Required for Docker to access GPU:
   - **Windows**: Docker Desktop automatically includes this when GPU support is enabled
   - **Linux**: Install `nvidia-container-toolkit`:
     ```bash
     # Ubuntu/Debian
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```
   - **macOS**: Not supported (use CPU mode)

3. **Enable GPU in Docker Desktop** (Windows):
   - Settings → Resources → WSL Integration → Enable GPU support
   - Restart Docker Desktop

### Configuration

1. **Uncomment GPU configuration in `docker-compose.yml`**:
   ```yaml
   ollama:
     # ... other config ...
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

2. **Restart the services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Verify GPU is accessible in container**:
   ```bash
   docker-compose exec ollama nvidia-smi
   ```

### Troubleshooting

- **"No devices found"**: Ensure NVIDIA Container Toolkit is installed and Docker is restarted
- **"GPU not detected"**: Check that `nvidia-smi` works on the host system
- **Windows WSL2**: Ensure WSL2 backend is enabled in Docker Desktop settings

## Common Commands

### Start services in background:
```bash
docker-compose up -d
```

### View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f ollama
docker-compose logs -f qdrant
```

### Stop services:
```bash
docker-compose down
```

### Stop and remove volumes (⚠️ deletes all data):
```bash
docker-compose down -v
```

### Rebuild after code changes:
```bash
docker-compose up --build
```

### Execute commands in container:
```bash
# Access app container shell
docker-compose exec app bash

# Access Ollama container shell
docker-compose exec ollama sh
```

### Pull latest Ollama models:
```bash
# Access Ollama container
docker-compose exec ollama ollama pull deepseek-r1:8b
docker-compose exec ollama ollama pull mistral:latest
```

## Troubleshooting

### Port Already in Use

If port 7860 is already in use:
1. Stop the conflicting service
2. Or change the port in `docker-compose.yml`:
   ```yaml
   app:
     ports:
       - "7861:7860"  # Use 7861 on host
   ```

### Ollama Models Not Found

After first startup, you need to pull models:
```bash
docker-compose exec ollama ollama pull deepseek-r1:8b
docker-compose exec ollama ollama pull mistral:latest
```

### Qdrant Connection Issues

If the app can't connect to Qdrant:
1. Check Qdrant is running: `docker-compose ps`
2. Check logs: `docker-compose logs qdrant`
3. Verify network: `docker network inspect smolagents_smolagents-network`

### Application Can't Connect to Services

Ensure environment variables are set correctly. The app uses service names (`ollama`, `qdrant`) for internal Docker network communication.

### Data Not Persisting

Verify volumes are created:
```bash
docker volume ls | grep smolagents
```

If volumes are missing, recreate them:
```bash
docker-compose down -v
docker-compose up -d
```

## Development

For development with live code reloading, you can mount the source code:

```yaml
# Add to docker-compose.yml app service
volumes:
  - ./src:/app/src
  - ./examples:/app/examples
```

Then restart:
```bash
docker-compose up --build
```

## Production Considerations

For production deployments, consider:

1. **Resource Limits**: Add resource constraints in `docker-compose.yml`
2. **Health Checks**: Add health check configurations
3. **Restart Policies**: Already set to `unless-stopped`
4. **Secrets Management**: Use Docker secrets or environment files
5. **Reverse Proxy**: Add nginx/traefik for SSL and domain routing
6. **Monitoring**: Add logging and monitoring solutions

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Gradio Documentation](https://gradio.app/docs/)

