@echo off
setlocal enabledelayedexpansion

echo.
echo ===============================================
echo    CORTEX SUITE v5.4.0 - Docker Installer
echo    Multi-Platform Support: Intel x86_64, Apple Silicon, ARM64
echo    GPU acceleration and improved reliability
echo    Date: %date% %time%
echo ===============================================
echo.
echo ** Starting Cortex Suite Launcher
echo =================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 goto docker_not_running
echo OK: Docker is running

REM Check for existing container
echo DEBUG: Checking for existing container...
docker container ls -a >container_check.txt 2>&1
if errorlevel 1 goto docker_error_cleanup

findstr /c:"cortex-suite" container_check.txt >nul
if not errorlevel 1 goto container_found
del container_check.txt
echo DEBUG: No cortex-suite container found, proceeding to first-time setup
goto first_time_setup

:container_found
del container_check.txt
echo PACK: Cortex Suite container found

REM Check if container is running
docker container ls >running_check.txt 2>&1
if errorlevel 1 goto docker_error_cleanup2

findstr /c:"cortex-suite" running_check.txt >nul
if not errorlevel 1 goto already_running
del running_check.txt
echo DEBUG: Container exists but not running, starting it
goto start_existing

:already_running
del running_check.txt
echo OK: Cortex Suite is already running!
echo.
echo WEB: Access your Cortex Suite at:
echo    Main App: http://localhost:8501
echo    API Docs: http://localhost:8000/docs
pause
exit /b 0

:start_existing
echo RESTART: Starting existing Cortex Suite...
docker start cortex-suite
if errorlevel 1 goto container_start_failed
echo WAIT: Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul
echo OK: Cortex Suite is now running!
echo.
echo WEB: Access your Cortex Suite at:
echo    Main App: http://localhost:8501
echo    API Docs: http://localhost:8000/docs
pause
exit /b 0

:docker_not_running
echo ERROR: Docker is not running!
echo Please start Docker Desktop and try again.
pause
exit /b 1

:docker_error_cleanup
echo ERROR: Docker command failed
del container_check.txt 2>nul
goto first_time_setup

:docker_error_cleanup2
echo ERROR: Docker command failed
del running_check.txt 2>nul
goto first_time_setup

:first_time_setup
echo DEBUG: Starting first-time setup process
echo.
echo ===============================================
echo    FIRST TIME SETUP
echo ===============================================
echo This will take 5-10 minutes. We will:
echo    - Configure your storage paths (VALIDATED BEFORE BUILD)
echo    - Build the Cortex Suite image
echo    - Download AI models (approx 4GB)
echo    - Set up the database
echo.

REM Check/create .env file
if not exist .env goto check_env_example
goto configure_paths_before_build

:check_env_example
if exist .env.example goto create_env
echo ERROR: .env.example not found! Make sure you're in the correct directory.
pause
exit /b 1

:create_env
copy .env.example .env >nul
echo INFO: Created .env configuration file

:configure_paths_before_build
echo.
echo ===============================================
echo    STORAGE CONFIGURATION (Before Build)
echo ===============================================
echo IMPORTANT: Configure your paths NOW before the long build process.
echo This prevents wasted time if paths are incorrect.
echo.

REM Read existing host mappings from .env (if any)
set "ENV_AI_DB_PATH="
set "ENV_SOURCE_PATH="
for /f "tokens=1* delims==" %%A in ('findstr /B "WINDOWS_AI_DATABASE_PATH=" .env 2^>nul') do set "ENV_AI_DB_PATH=%%B"
for /f "tokens=1* delims==" %%A in ('findstr /B "WINDOWS_KNOWLEDGE_SOURCE_PATH=" .env 2^>nul') do set "ENV_SOURCE_PATH=%%B"

REM Show existing configuration if present
if defined ENV_AI_DB_PATH (
    echo INFO: Found existing storage configuration:
    if defined ENV_AI_DB_PATH echo   AI Database: !ENV_AI_DB_PATH!
    if defined ENV_SOURCE_PATH echo   Knowledge Source: !ENV_SOURCE_PATH!
    echo.
)

:prompt_ai_db_path
echo ===================================================================
echo STORAGE CONFIGURATION: AI Database Path
echo ===================================================================
echo This is where Cortex will store the knowledge graph and vector DB.
echo.
if defined ENV_AI_DB_PATH (
    echo Current: !ENV_AI_DB_PATH!
    echo.
)
set "USE_HOST_AI_DB=N"
set /p USE_HOST_AI_DB=Use host folder for AI database storage? (y/N):
if /I "%USE_HOST_AI_DB%" NEQ "Y" goto skip_ai_db_config

:enter_ai_db_path
set "HOST_AI_DB="
if defined ENV_AI_DB_PATH (
    set /p "HOST_AI_DB=Enter host path [!ENV_AI_DB_PATH!]: "
) else (
    set /p "HOST_AI_DB=Enter host path (e.g., D:\ai_databases): "
)

REM Use existing path if user just pressed ENTER
if not defined HOST_AI_DB (
    if defined ENV_AI_DB_PATH (
        set "HOST_AI_DB=!ENV_AI_DB_PATH!"
        echo INFO: Using existing path: !HOST_AI_DB!
    ) else (
        goto skip_ai_db_config
    )
)

REM Remove quotes if present
set "HOST_AI_DB=%HOST_AI_DB:"=%"

REM Extract drive letter and validate it exists
set "AI_DB_DRIVE=!HOST_AI_DB:~0,2!"
if not exist "!AI_DB_DRIVE!\" (
    echo.
    echo *** ERROR: Drive !AI_DB_DRIVE! does not exist! ***
    echo.
    echo The drive letter you specified cannot be found.
    echo Please check your drive letter and try again.
    echo.
    set "RETRY_AI_DB=N"
    set /p "RETRY_AI_DB=Would you like to enter a different path? (Y/n): "
    if /I "!RETRY_AI_DB!" NEQ "N" goto enter_ai_db_path
    echo Skipping AI database host mapping...
    goto skip_ai_db_config
)

REM Try to create directory if it doesn't exist
if not exist "!HOST_AI_DB!" (
    echo INFO: Directory does not exist, attempting to create...
    echo Path: !HOST_AI_DB!
    mkdir "!HOST_AI_DB!" 2>nul
    if errorlevel 1 (
        echo.
        echo *** ERROR: Failed to create directory ***
        echo Path: !HOST_AI_DB!
        echo.
        echo Possible causes:
        echo   - Invalid path format
        echo   - Permission denied
        echo   - Parent directory doesn't exist
        echo.
        set "RETRY_AI_DB=N"
        set /p "RETRY_AI_DB=Would you like to enter a different path? (Y/n): "
        if /I "!RETRY_AI_DB!" NEQ "N" goto enter_ai_db_path
        echo Skipping AI database host mapping...
        goto skip_ai_db_config
    )
    echo OK: Directory created successfully!
)

REM Path is valid - save to .env
echo OK: AI database path validated: !HOST_AI_DB!
REM Remove old entries and add new ones
findstr /V /B "WINDOWS_AI_DATABASE_PATH= AI_DATABASE_PATH=" .env > .env.tmp 2>nul
move /Y .env.tmp .env >nul 2>&1
>>.env echo WINDOWS_AI_DATABASE_PATH=!HOST_AI_DB!
>>.env echo AI_DATABASE_PATH=/data/ai_databases
set "ENV_AI_DB_PATH=!HOST_AI_DB!"

:skip_ai_db_config
echo.

:prompt_source_path
echo ===================================================================
echo STORAGE CONFIGURATION: Knowledge Source Path
echo ===================================================================
echo This is where your source documents are stored for ingestion.
echo (PDF, Word, text files, etc.)
echo.
if defined ENV_SOURCE_PATH (
    echo Current: !ENV_SOURCE_PATH!
    echo.
)
set "USE_HOST_SRC=N"
set /p USE_HOST_SRC=Use host folder for Knowledge Source documents? (y/N):
if /I "%USE_HOST_SRC%" NEQ "Y" goto skip_source_config

:enter_source_path
set "HOST_SRC="
if defined ENV_SOURCE_PATH (
    set /p "HOST_SRC=Enter host path [!ENV_SOURCE_PATH!]: "
) else (
    set /p "HOST_SRC=Enter host path (e.g., D:\Documents\KB_Source): "
)

REM Use existing path if user just pressed ENTER
if not defined HOST_SRC (
    if defined ENV_SOURCE_PATH (
        set "HOST_SRC=!ENV_SOURCE_PATH!"
        echo INFO: Using existing path: !HOST_SRC!
    ) else (
        goto skip_source_config
    )
)

REM Remove quotes if present
set "HOST_SRC=%HOST_SRC:"=%"

REM Extract drive letter and validate it exists
set "SRC_DRIVE=!HOST_SRC:~0,2!"
if not exist "!SRC_DRIVE!\" (
    echo.
    echo *** ERROR: Drive !SRC_DRIVE! does not exist! ***
    echo.
    echo The drive letter you specified cannot be found.
    echo Please check your drive letter and try again.
    echo.
    set "RETRY_SRC=N"
    set /p "RETRY_SRC=Would you like to enter a different path? (Y/n): "
    if /I "!RETRY_SRC!" NEQ "N" goto enter_source_path
    echo Skipping Knowledge Source host mapping...
    goto skip_source_config
)

REM Verify or create source directory
if not exist "!HOST_SRC!" (
    echo WARNING: Source directory does not exist
    echo Path: !HOST_SRC!
    echo.
    set "CREATE_SRC=N"
    set /p "CREATE_SRC=Create this directory? (y/N): "
    if /I "!CREATE_SRC!" NEQ "Y" (
        set "RETRY_SRC=N"
        set /p "RETRY_SRC=Would you like to enter a different path? (Y/n): "
        if /I "!RETRY_SRC!" NEQ "N" goto enter_source_path
        echo Skipping Knowledge Source host mapping...
        goto skip_source_config
    )
    mkdir "!HOST_SRC!" 2>nul
    if errorlevel 1 (
        echo.
        echo *** ERROR: Failed to create directory ***
        echo.
        set "RETRY_SRC=N"
        set /p "RETRY_SRC=Would you like to enter a different path? (Y/n): "
        if /I "!RETRY_SRC!" NEQ "N" goto enter_source_path
        echo Skipping Knowledge Source host mapping...
        goto skip_source_config
    )
    echo OK: Directory created successfully!
)

REM Path is valid - save to .env
echo OK: Knowledge Source path validated: !HOST_SRC!
REM Remove old entries and add new ones
findstr /V /B "WINDOWS_KNOWLEDGE_SOURCE_PATH= KNOWLEDGE_SOURCE_PATH=" .env > .env.tmp 2>nul
move /Y .env.tmp .env >nul 2>&1
>>.env echo WINDOWS_KNOWLEDGE_SOURCE_PATH=!HOST_SRC!
>>.env echo KNOWLEDGE_SOURCE_PATH=/data/knowledge_base
set "ENV_SOURCE_PATH=!HOST_SRC!"

:skip_source_config
echo.

REM Show final configuration summary
echo ===================================================================
echo CONFIGURATION SUMMARY
echo ===================================================================
if defined ENV_AI_DB_PATH (
    echo AI Database:      !ENV_AI_DB_PATH!
) else (
    echo AI Database:      [Using Docker volume - data inside container]
)
if defined ENV_SOURCE_PATH (
    echo Knowledge Source: !ENV_SOURCE_PATH!
) else (
    echo Knowledge Source: [Using Docker volume - data inside container]
)
echo.

REM Final confirmation before starting the long build
echo ===================================================================
echo READY TO BUILD
echo ===================================================================
echo The Docker build process will now begin.
echo This typically takes 5-10 minutes depending on your internet speed.
echo.
set "CONFIRM_BUILD=Y"
set /p "CONFIRM_BUILD=Proceed with build? (Y/n): "
if /I "!CONFIRM_BUILD!"=="N" (
    echo.
    echo Build cancelled. Run this script again when ready.
    pause
    exit /b 0
)

:build_image
echo.
echo BUILD: Building Cortex Suite (this may take a while)...
echo     This includes downloading Python packages and system dependencies...

REM Detect GPU and build appropriate image
echo DEBUG: Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo OK: NVIDIA GPU detected - Building GPU-accelerated image
    echo BUILD: Using Dockerfile.gpu for CUDA support
    docker build -t cortex-suite -f Dockerfile.gpu .
) else (
    echo INFO: No NVIDIA GPU detected - Building CPU-only image
    echo BUILD: Using standard Dockerfile
    docker build -t cortex-suite -f Dockerfile .
)

if errorlevel 1 goto build_failed

echo ** Starting Cortex Suite...

REM Detect available drives for mounting
echo DETECT: Checking for user directories to mount...
if exist "C:\" (echo   MOUNT: C:\ drive will be available as /mnt/c) else (echo   SKIP: C:\ drive not found)
if exist "D:\" (echo   MOUNT: D:\ drive will be available as /mnt/d) else (echo   SKIP: D:\ drive not found)
if exist "E:\" (echo   MOUNT: E:\ drive will be available as /mnt/e) else (echo   SKIP: E:\ drive not found)
if exist "F:\" (echo   MOUNT: F:\ drive will be available as /mnt/f) else (echo   SKIP: F:\ drive not found)
echo   OK: Starting Cortex Suite with all detected drives...

REM Check for NVIDIA GPU support before adding --gpus flag
echo DEBUG: Checking for NVIDIA GPU support...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo DEBUG: Testing Docker GPU access...
    docker run --rm --gpus all hello-world >nul 2>&1
    if not errorlevel 1 (
        echo OK: NVIDIA GPU detected and Docker GPU support available
        set DOCKER_CMD=docker run -d --name cortex-suite --gpus all -p 8501:8501 -p 8000:8000 -v cortex_data:/data -v cortex_logs:/home/cortex/app/logs -v cortex_ollama:/home/cortex/.ollama
    ) else (
        echo INFO: NVIDIA GPU detected but Docker GPU support unavailable, using CPU-only mode
        set DOCKER_CMD=docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 -v cortex_data:/data -v cortex_logs:/home/cortex/app/logs -v cortex_ollama:/home/cortex/.ollama
    )
) else (
    echo INFO: No NVIDIA GPU detected, using CPU-only mode
    echo INFO: This is optimal for ARM64 PCs, Apple Silicon, and Intel systems without NVIDIA GPUs
    set DOCKER_CMD=docker run -d --name cortex-suite -p 8501:8501 -p 8000:8000 -v cortex_data:/data -v cortex_logs:/home/cortex/app/logs -v cortex_ollama:/home/cortex/.ollama
)

REM Add host bind mounts for AI database and Knowledge Source if configured
if not defined ENV_AI_DB_PATH goto skip_ai_db_mount
echo   MOUNT: Mapping AI database to host: !ENV_AI_DB_PATH! -^> /data/ai_databases
set DOCKER_CMD=!DOCKER_CMD! -v "!ENV_AI_DB_PATH!:/data/ai_databases"
:skip_ai_db_mount

if not defined ENV_SOURCE_PATH goto skip_src_mount
echo   MOUNT: Mapping Knowledge Source to host (read-only): !ENV_SOURCE_PATH! -^> /data/knowledge_base
set DOCKER_CMD=!DOCKER_CMD! -v "!ENV_SOURCE_PATH!:/data/knowledge_base:ro"
:skip_src_mount

REM Mount available drives (read-only for safety)
if exist "C:\" set DOCKER_CMD=!DOCKER_CMD! -v "C:\:/mnt/c:ro"
if exist "D:\" set DOCKER_CMD=!DOCKER_CMD! -v "D:\:/mnt/d:ro"
if exist "E:\" set DOCKER_CMD=!DOCKER_CMD! -v "E:\:/mnt/e:ro"
if exist "F:\" set DOCKER_CMD=!DOCKER_CMD! -v "F:\:/mnt/f:ro"

REM Complete the docker command with host networking for Ollama access
set DOCKER_CMD=!DOCKER_CMD! --add-host=host.docker.internal:host-gateway --env-file .env --restart unless-stopped cortex-suite

REM Execute the dynamically built command
!DOCKER_CMD!

if errorlevel 1 goto start_failed

echo SETUP: Fixing data directory permissions...
docker exec -u root cortex-suite chown -R cortex:cortex /data 2>nul || echo "   Permission fix completed"

echo WAIT: Waiting for services to fully start (60 seconds)...
echo    This includes downloading and setting up AI models...

for /l %%i in (1,1,12) do (
    timeout /t 5 /nobreak >nul
    set /a elapsed=%%i*5
    echo    ... !elapsed! seconds elapsed
)

echo CHECK: Checking if services are ready...
curl -s http://localhost:8501/_stcore/health >nul 2>&1
if errorlevel 1 goto ui_check_failed
echo OK: Streamlit UI is ready!
goto api_check

:ui_check_failed
echo WARNING: UI might still be starting up...

:api_check
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto api_check_failed
echo OK: API is ready!
goto success

:api_check_failed
echo WARNING: API might still be starting up...

:success
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
exit /b 0

:build_failed
echo.
echo ===============================================
echo BUILD FAILED
echo ===============================================
echo ERROR: Build failed! This could be due to:
echo    - Network connectivity issues (download interrupted)
echo    - Insufficient disk space (need approx 10GB)
echo    - Docker permission issues
echo    - Corrupted Docker build cache
echo    - Windows system folder access (RECYCLE.BIN issue)
echo.
echo RECOMMENDED FIX - Run these commands then retry:
echo    docker system prune -a -f
echo    docker builder prune -a -f
echo.
echo If error mentions "short read" or "unexpected EOF":
echo    This is a network/download issue - just retry the build
echo    The download was interrupted and needs to restart
echo.
echo If error mentions "hash mismatch":
echo    The PyTorch packages may have been updated on their servers.
echo    Try running: docker builder prune -a -f
echo    Then retry the build.
pause
exit /b 1

:container_start_failed
echo WARNING: Failed to start existing container, trying rebuild...
goto first_time_setup

:start_failed
echo ERROR: Failed to start Cortex Suite!
echo This might be because the ports are already in use.
echo Try running: docker stop cortex-suite ^&^& docker rm cortex-suite
pause
exit /b 1
