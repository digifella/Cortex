@echo off
REM Cortex Suite - Hybrid Model Architecture Launcher for Windows
REM Supports both Docker Model Runner and Ollama backends

setlocal enabledelayedexpansion

echo 🚀 Cortex Suite - Hybrid Model Architecture Launcher
echo ==================================================

REM Configuration
set "DOCKER_COMPOSE_FILE=docker-compose-hybrid.yml"
if "%MODEL_STRATEGY%"=="" set "MODEL_STRATEGY=hybrid_docker_preferred"
if "%DEPLOYMENT_ENV%"=="" set "DEPLOYMENT_ENV=production"

REM Check if we're in the right directory
if not exist "%DOCKER_COMPOSE_FILE%" (
    echo ❌ Docker compose file not found: %DOCKER_COMPOSE_FILE%
    echo Make sure you're running this script from the docker directory
    pause
    exit /b 1
)

echo ℹ️  Checking prerequisites...

REM Check Docker
docker --version >nul 2>&1
if not errorlevel 1 (
    echo ✅ Docker is available
) else (
    echo ❌ Docker is not installed!
    echo Please install Docker Desktop and try again.
    echo https://docs.docker.com/get-docker/
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if not errorlevel 1 (
    echo ✅ Docker is running
) else (
    echo ❌ Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check Docker Compose
docker compose version >nul 2>&1
if not errorlevel 1 (
    echo ✅ Docker Compose is available
) else (
    echo ❌ Docker Compose is not available!
    echo Please ensure you have Docker Compose v2 installed.
    pause
    exit /b 1
)

REM Check Docker Model Runner support
docker model --help >nul 2>&1
if not errorlevel 1 (
    echo ✅ Docker Model Runner detected - enterprise features available
    set "DOCKER_MODELS_AVAILABLE=true"
) else (
    echo ⚠️  Docker Model Runner not available - using Ollama fallback
    set "DOCKER_MODELS_AVAILABLE=false"
)

echo ✅ Prerequisites check complete
echo.

REM Check disk space
echo ℹ️  Checking available disk space...
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set bytes=%%a
set /a gb_free=!bytes:~0,-9!/1
if !gb_free! LSS 15 (
    echo ⚠️  Only !gb_free!GB available. Recommend at least 15GB for full installation.
    set /p "continue=Continue anyway? (y/N): "
    if /i not "!continue!"=="y" exit /b 1
) else (
    echo ✅ !gb_free!GB available - sufficient for installation
)

echo.
echo 🎯 Choose Deployment Profile:
echo.
echo 1. Hybrid (Recommended) - Docker Model Runner + Ollama fallback
echo 2. Enterprise - Docker Model Runner only (requires Docker Model Runner)
echo 3. Standard - Ollama only (traditional approach)
echo 4. Development - Minimal setup for testing
echo.

:profile_selection
set /p "profile_choice=Select profile (1-4): "

if "%profile_choice%"=="1" (
    set "PROFILE=hybrid"
    set "MODEL_STRATEGY=hybrid_docker_preferred"
    goto profile_selected
)
if "%profile_choice%"=="2" (
    if "%DOCKER_MODELS_AVAILABLE%"=="true" (
        set "PROFILE=enterprise"
        set "MODEL_STRATEGY=docker_only"
        goto profile_selected
    ) else (
        echo ❌ Docker Model Runner not available. Please choose another option.
        goto profile_selection
    )
)
if "%profile_choice%"=="3" (
    set "PROFILE=ollama"
    set "MODEL_STRATEGY=ollama_only"
    goto profile_selected
)
if "%profile_choice%"=="4" (
    set "PROFILE=ollama"
    set "MODEL_STRATEGY=ollama_only"
    set "DEPLOYMENT_ENV=development"
    goto profile_selected
)

echo Invalid choice. Please enter 1-4.
goto profile_selection

:profile_selected
echo ✅ Selected profile: %PROFILE% (strategy: %MODEL_STRATEGY%)

echo.
echo ℹ️  Setting up environment configuration...

REM Create .env file if it doesn't exist
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo ✅ Created .env from template
    ) else (
        echo ❌ .env.example not found!
        pause
        exit /b 1
    )
) else (
    echo ℹ️  .env file already exists
)

REM Update environment variables
powershell -Command "(Get-Content .env) -replace 'MODEL_DISTRIBUTION_STRATEGY=.*', 'MODEL_DISTRIBUTION_STRATEGY=%MODEL_STRATEGY%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'DEPLOYMENT_ENV=.*', 'DEPLOYMENT_ENV=%DEPLOYMENT_ENV%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'ENABLE_DOCKER_MODELS=.*', 'ENABLE_DOCKER_MODELS=%DOCKER_MODELS_AVAILABLE%' | Set-Content .env"

echo ✅ Environment configuration updated

echo.
echo ℹ️  Starting Cortex Suite services...

REM Stop any existing services
docker compose -f %DOCKER_COMPOSE_FILE% down >nul 2>&1

REM Build Docker Compose command with profiles
set "COMPOSE_CMD=docker compose -f %DOCKER_COMPOSE_FILE%"

if "%PROFILE%"=="hybrid" (
    set "COMPOSE_CMD=%COMPOSE_CMD% --profile hybrid"
)
if "%PROFILE%"=="enterprise" (
    set "COMPOSE_CMD=%COMPOSE_CMD% --profile docker-models --profile enterprise"
)
if "%PROFILE%"=="ollama" (
    set "COMPOSE_CMD=%COMPOSE_CMD% --profile ollama"
)

REM Start services
echo ℹ️  Building and starting services (this may take a few minutes)...
%COMPOSE_CMD% up -d --build
if errorlevel 1 (
    echo ❌ Failed to start services!
    echo Check the logs with: %COMPOSE_CMD% logs
    pause
    exit /b 1
)

echo ✅ Services started successfully

echo.
echo ℹ️  Waiting for services to be ready...

REM Wait for API
echo ℹ️  Waiting for API server...
set /a attempts=0
:wait_api
set /a attempts+=1
curl -s http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ API server is ready
    goto api_ready
)
if %attempts% GEQ 30 (
    echo ⚠️  API server seems slow to start
    goto api_ready
)
timeout /t 2 >nul
goto wait_api

:api_ready
REM Wait for UI
echo ℹ️  Waiting for Streamlit UI...
set /a attempts=0
:wait_ui
set /a attempts+=1
curl -s http://localhost:8501/_stcore/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ Streamlit UI is ready
    goto ui_ready
)
if %attempts% GEQ 20 (
    echo ⚠️  UI seems slow to start
    goto ui_ready
)
timeout /t 2 >nul
goto wait_ui

:ui_ready
echo.
echo 🤖 AI Model Setup:
echo.
echo The system needs AI models to function. This process will:
echo - Download required models (~11-15GB)
echo - Take 10-30 minutes depending on internet speed
echo - Continue in the background while you use the interface
echo.

set /p "setup_models=Start model setup now? (Y/n): "
if /i "%setup_models%"=="n" (
    echo ℹ️  Skipping model setup - you can configure models later via the Setup Wizard
    goto show_completion
)

REM Start model initialization
echo ℹ️  Setting up AI models...
if "%PROFILE%"=="hybrid" (
    %COMPOSE_CMD% --profile init up model-init-hybrid --no-deps
) else if "%PROFILE%"=="enterprise" (
    %COMPOSE_CMD% --profile init up model-init-hybrid --no-deps
) else (
    %COMPOSE_CMD% --profile init up model-init --no-deps
)

echo ✅ Model setup initiated - check the Setup Wizard for progress

:show_completion
echo.
echo 🎉 Cortex Suite is now running!
echo.
echo 📱 Access Points:
echo    Main Application: http://localhost:8501
echo    API Documentation: http://localhost:8000/docs
echo    Setup Wizard: http://localhost:8501/0_Setup_Wizard
echo.
echo 💡 Quick Start:
echo    1. Visit the Setup Wizard to complete configuration
echo    2. Upload documents via Knowledge Ingest
echo    3. Try AI Research or Proposal Generation
echo.
echo 🔧 Management Commands:
echo    Stop:      %COMPOSE_CMD% down
echo    Restart:   %COMPOSE_CMD% restart
echo    Logs:      %COMPOSE_CMD% logs -f
echo    Status:    %COMPOSE_CMD% ps
echo.
echo 📊 System Info:
echo    Profile:   %PROFILE%
echo    Strategy:  %MODEL_STRATEGY%
echo    Environment: %DEPLOYMENT_ENV%
echo    Docker Models: %DOCKER_MODELS_AVAILABLE%
echo.

REM Check if first time setup
if not exist "%USERPROFILE%\.cortex\setup_progress.json" (
    echo ℹ️  First time setup detected - visit the Setup Wizard to complete configuration
)

echo.
echo ✅ Setup complete! 🚀

REM Ask to open browser
set /p "open_browser=Open browser to Cortex Suite? (Y/n): "
if /i not "%open_browser%"=="n" (
    start http://localhost:8501
)

echo.
echo Press any key to exit...
pause >nul