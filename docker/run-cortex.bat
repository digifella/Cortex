@echo off
REM Cortex Suite - Windows One-Click Launcher
setlocal enabledelayedexpansion

echo ** Cortex Suite - Easy Launcher
echo ================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo OK: Docker is running

REM Check if container already exists
docker ps -a --format "{{.Names}}" | findstr /x "cortex-suite" >nul
if %errorlevel% equ 0 (
    echo PACK: Cortex Suite container found
    
    REM Check if it's running
    docker ps --format "{{.Names}}" | findstr /x "cortex-suite" >nul
    if %errorlevel% equ 0 (
        echo OK: Cortex Suite is already running!
        echo.
        echo WEB: Access your Cortex Suite at:
        echo    Main App: http://localhost:8501
        echo    API Docs: http://localhost:8000/docs
        pause
        exit /b 0
    ) else (
        echo RESTART: Starting existing Cortex Suite...
        docker start cortex-suite
        echo WAIT: Waiting for services to start (30 seconds)...
        timeout /t 30 /nobreak >nul
        echo OK: Cortex Suite is now running!
        echo.
        echo WEB: Access your Cortex Suite at:
        echo    Main App: http://localhost:8501
        echo    API Docs: http://localhost:8000/docs
        pause
        exit /b 0
    )
)

REM First time setup
echo NEW: First time setup - this will take 5-10 minutes
echo WAIT: Please be patient while we:
echo    - Build the Cortex Suite image
echo    - Download AI models (approx4GB)
echo    - Set up the database
echo.

REM Check if .env exists, create from example if not
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo INFO: Created .env configuration file
    ) else (
        echo ERROR: .env.example not found! Make sure you're in the docker directory.
        pause
        exit /b 1
    )
)

REM Build the image
echo BUILD: Building Cortex Suite (this may take a while)...
echo     This includes downloading Python packages and system dependencies...
docker build -t cortex-suite -f Dockerfile ..

if %errorlevel% neq 0 (
    echo ERROR: Build failed! This could be due to:
    echo    - Network connectivity issues
    echo    - Insufficient disk space (need approx10GB)
    echo    - Docker permission issues
    echo.
    echo Try running: docker system prune -f
    echo Then try again.
    pause
    exit /b 1
)

REM Run the container
echo ** Starting Cortex Suite...
docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 -v cortex_data:/home/cortex/data -v cortex_logs:/home/cortex/app/logs --env-file .env --restart unless-stopped cortex-suite

if %errorlevel% neq 0 (
    echo ERROR: Failed to start Cortex Suite!
    echo This might be because the ports are already in use.
    echo Try running: docker stop cortex-suite ^&^& docker rm cortex-suite
    pause
    exit /b 1
)

echo WAIT: Waiting for services to fully start (60 seconds)...
echo    This includes downloading and setting up AI models...

REM Wait and show progress
for /l %%i in (1,1,12) do (
    timeout /t 5 /nobreak >nul
    set /a elapsed=%%i*5
    echo    ... !elapsed! seconds elapsed
)

echo CHECK: Checking if services are ready...

REM Simple health check (basic version for Windows)
curl -s http://localhost:8501/_stcore/health >nul 2>&1
if %errorlevel% equ 0 (
    echo OK: Streamlit UI is ready!
) else (
    echo WARNING:  UI might still be starting up...
)

curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo OK: API is ready!
) else (
    echo WARNING:  API might still be starting up...
)

echo.
echo SUCCESS: Cortex Suite is now running!
echo.
echo WEB: Access your Cortex Suite at:
echo    Main Application: http://localhost:8501
echo    API Documentation: http://localhost:8000/docs
echo.
echo INFO: Useful commands:
echo    Stop:    docker stop cortex-suite
echo    Start:   docker start cortex-suite
echo    Logs:    docker logs cortex-suite -f
echo    Remove:  docker stop cortex-suite ^&^& docker rm cortex-suite
echo.
echo TIP: Your data is safely stored in Docker volumes and will persist
echo    between stops and starts!
echo.
pause
