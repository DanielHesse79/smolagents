# Setup script for smolagents
# This script helps install and configure all dependencies

Write-Host "=== smolagents Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check for Python
Write-Host "Checking for Python installation..." -ForegroundColor Yellow
$pythonFound = $false
$pythonCmd = $null

# Try different Python commands
$pythonCommands = @("python", "python3", "py")
foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $version -match "Python") {
            $pythonFound = $true
            $pythonCmd = $cmd
            Write-Host "✓ Found Python: $version" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonFound) {
    Write-Host "✗ Python not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.10 or newer:" -ForegroundColor Yellow
    Write-Host "  1. Download from https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  2. Make sure to check 'Add Python to PATH' during installation" -ForegroundColor White
    Write-Host "  3. Restart your terminal and run this script again" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$versionOutput = & $pythonCmd --version 2>&1
$versionMatch = $versionOutput -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "✗ Python 3.10+ required. Found: $versionOutput" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Python version OK: $versionOutput" -ForegroundColor Green
} else {
    Write-Host "⚠ Could not parse Python version" -ForegroundColor Yellow
}

Write-Host ""

# Check for pip
Write-Host "Checking for pip..." -ForegroundColor Yellow
try {
    $pipVersion = & $pythonCmd -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ pip is available" -ForegroundColor Green
    } else {
        Write-Host "✗ pip not found" -ForegroundColor Red
        Write-Host "Installing pip..." -ForegroundColor Yellow
        & $pythonCmd -m ensurepip --upgrade
    }
} catch {
    Write-Host "✗ pip not found. Please install pip manually." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Install dependencies
Write-Host "Installing smolagents and dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray

try {
    # Upgrade pip first
    Write-Host "Upgrading pip..." -ForegroundColor Gray
    & $pythonCmd -m pip install --upgrade pip --quiet
    
    # Install smolagents with all optional dependencies
    Write-Host "Installing smolagents[all]..." -ForegroundColor Gray
    & $pythonCmd -m pip install -e ".[all]" --quiet
    
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Error installing dependencies: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try installing manually:" -ForegroundColor Yellow
    Write-Host "  $pythonCmd -m pip install -e `".[all]`"" -ForegroundColor White
    exit 1
}

Write-Host ""

# Check Ollama
Write-Host "Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaVersion = & ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Ollama is installed: $ollamaVersion" -ForegroundColor Green
        
        # Check for required models
        Write-Host "Checking for required Ollama models..." -ForegroundColor Gray
        $models = & ollama list 2>&1 | Select-String -Pattern "deepseek|mistral"
        if ($models) {
            Write-Host "✓ Found some models" -ForegroundColor Green
        } else {
            Write-Host "⚠ Required models not found. You may need to run:" -ForegroundColor Yellow
            Write-Host "  ollama pull deepseek-r1:8b" -ForegroundColor White
            Write-Host "  ollama pull mistral:latest" -ForegroundColor White
        }
    } else {
        Write-Host "⚠ Ollama not found (optional)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Ollama not found (optional)" -ForegroundColor Yellow
    Write-Host "  Install from: https://ollama.com" -ForegroundColor Gray
}

Write-Host ""

# Check Docker
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = & docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker is installed: $dockerVersion" -ForegroundColor Green
        Write-Host "  To start Qdrant: docker run -p 6333:6333 qdrant/qdrant" -ForegroundColor Gray
    } else {
        Write-Host "⚠ Docker not found (optional)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Docker not found (optional)" -ForegroundColor Yellow
    Write-Host "  Install from: https://www.docker.com/products/docker-desktop" -ForegroundColor Gray
}

Write-Host ""

# Create .env file template
Write-Host "Creating .env file template..." -ForegroundColor Yellow
$envTemplate = @"
# Environment variables for smolagents
# Copy this file to .env and fill in your API keys

# Hugging Face API token (optional, for private models or higher rate limits)
# HF_TOKEN=your_huggingface_token_here
# HF_API_KEY=your_huggingface_token_here

# Mistral API key (optional, if using Mistral API instead of Ollama)
# MISTRAL_API_KEY=your_mistral_api_key_here

# OpenAI API key (optional, if using OpenAI models)
# OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API key (optional, if using Anthropic models)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Search API keys (optional, for web search tools)
# SERPAPI_API_KEY=your_serpapi_key_here
# SERPER_API_KEY=your_serper_api_key_here

# Together AI API key (optional, for Together AI provider)
# TOGETHER_API_KEY=your_together_api_key_here
"@

if (-not (Test-Path ".env")) {
    $envTemplate | Out-File -FilePath ".env.example" -Encoding UTF8
    Write-Host "✓ Created .env.example file" -ForegroundColor Green
    Write-Host "  Copy it to .env and fill in your API keys if needed" -ForegroundColor Gray
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. If using Ollama, pull required models:" -ForegroundColor White
Write-Host "     ollama pull deepseek-r1:8b" -ForegroundColor Gray
Write-Host "     ollama pull mistral:latest" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. (Optional) Start Qdrant with Docker:" -ForegroundColor White
Write-Host "     docker run -p 6333:6333 qdrant/qdrant" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Run the Gradio UI:" -ForegroundColor White
Write-Host "     $pythonCmd examples/gradio_ui.py" -ForegroundColor Gray
Write-Host ""

