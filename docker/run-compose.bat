@echo off
setlocal enabledelayedexpansion

echo.
echo ===============================================
echo    CORTEX SUITE - Docker Compose Launcher
echo    Date: %date% %time%
echo ===============================================
echo.

REM Parse arguments
set "STORAGE_MODE=portable"
set "ACTION=start"
set "EXECUTION_MODE=auto"
set "UI_PORT=8501"
set "API_PORT=8000"
set "OLLAMA_PORT=11434"

:parse_args
if "%~1"=="" goto done_parsing
if /I "%~1"=="--external" (
    set "STORAGE_MODE=external"
    shift
    goto parse_args
)
if /I "%~1"=="-e" (
    set "STORAGE_MODE=external"
    shift
    goto parse_args
)
if /I "%~1"=="--stop" (
    set "ACTION=stop"
    shift
    goto parse_args
)
if /I "%~1"=="--down" (
    set "ACTION=down"
    shift
    goto parse_args
)
if /I "%~1"=="--gpu" (
    set "EXECUTION_MODE=gpu"
    shift
    goto parse_args
)
if /I "%~1"=="--cpu" (
    set "EXECUTION_MODE=cpu"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h" goto show_help
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --external, -e    Use external storage (host filesystem)
echo   --gpu             Force GPU profile (fail-fast if unavailable)
echo   --cpu             Force CPU profile
echo   --stop            Stop all services
echo   --down            Stop and remove containers
echo   --help, -h        Show this help
echo.
echo Storage Modes:
echo   Portable (default): Data stored in Docker volumes - fully transportable
echo   External:           Data stored on host filesystem - easy access/backup
exit /b 0

:done_parsing

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo OK: Docker is running

REM Handle stop/down actions
if "%ACTION%"=="stop" (
    echo Stopping Cortex Suite services...
    docker compose stop
    echo Services stopped
    pause
    exit /b 0
)

if "%ACTION%"=="down" (
    echo Stopping and removing Cortex Suite containers...
    docker compose down
    echo Containers removed
    pause
    exit /b 0
)

REM Check/create .env file
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo INFO: Created .env configuration file
    ) else (
        echo ERROR: .env.example not found!
        pause
        exit /b 1
    )
)

REM Load optional host port overrides from .env
for /f "tokens=1* delims==" %%A in ('findstr /B "CORTEX_UI_PORT=" .env 2^>nul') do set "UI_PORT=%%B"
for /f "tokens=1* delims==" %%A in ('findstr /B "CORTEX_API_PORT=" .env 2^>nul') do set "API_PORT=%%B"
for /f "tokens=1* delims==" %%A in ('findstr /B "CORTEX_OLLAMA_PORT=" .env 2^>nul') do set "OLLAMA_PORT=%%B"

REM Handle storage mode
if "%STORAGE_MODE%"=="external" goto external_mode

:portable_mode
echo.
echo ===============================================
echo    PORTABLE STORAGE MODE (Default)
echo ===============================================
echo.
echo Data will be stored in Docker volumes:
echo   - cortex_ai_databases (vector store, knowledge graph)
echo   - cortex_knowledge_base (source documents)
echo.
echo To backup your data, see the Docker volume backup commands
echo in the .env.example file.
echo.
set "COMPOSE_CMD=docker compose"
goto check_gpu

:external_mode
echo.
echo ===============================================
echo    EXTERNAL STORAGE MODE
echo ===============================================
echo.

REM Check if external paths are configured
set "EXTERNAL_AI_PATH="
set "EXTERNAL_KB_PATH="
for /f "tokens=1* delims==" %%A in ('findstr /B "EXTERNAL_AI_DATABASE_PATH=" .env 2^>nul') do set "EXTERNAL_AI_PATH=%%B"
for /f "tokens=1* delims==" %%A in ('findstr /B "EXTERNAL_KNOWLEDGE_PATH=" .env 2^>nul') do set "EXTERNAL_KB_PATH=%%B"

if not defined EXTERNAL_AI_PATH (
    echo EXTERNAL_AI_DATABASE_PATH not set in .env
    echo.
    set /p "EXTERNAL_AI_PATH=Enter path for AI databases, for example C:\ai_databases: "

    if not defined EXTERNAL_AI_PATH (
        echo ERROR: Path required for external storage mode
        pause
        exit /b 1
    )

    REM Create directory if needed
    if not exist "!EXTERNAL_AI_PATH!" (
        echo Creating directory: !EXTERNAL_AI_PATH!
        mkdir "!EXTERNAL_AI_PATH!" 2>nul
        if errorlevel 1 (
            echo ERROR: Failed to create directory
            pause
            exit /b 1
        )
    )

    REM Save to .env
    >>".env" echo EXTERNAL_AI_DATABASE_PATH=!EXTERNAL_AI_PATH!
)

if not defined EXTERNAL_KB_PATH (
    echo.
    set /p "EXTERNAL_KB_PATH=Enter path for knowledge source, for example C:\KB_Test [optional]: "

    if defined EXTERNAL_KB_PATH (
        if not exist "!EXTERNAL_KB_PATH!" (
            echo Creating directory: !EXTERNAL_KB_PATH!
            mkdir "!EXTERNAL_KB_PATH!" 2>nul
        )
        >>".env" echo EXTERNAL_KNOWLEDGE_PATH=!EXTERNAL_KB_PATH!
    )
)

echo.
echo External storage configuration:
echo   AI Database: !EXTERNAL_AI_PATH!
if defined EXTERNAL_KB_PATH echo   Knowledge Base: !EXTERNAL_KB_PATH!
echo.

set "COMPOSE_CMD=docker compose -f docker-compose.yml -f docker-compose.external.yml"

:check_gpu
set "HAS_GPU_CLI=false"
set "HAS_GPU_RUNTIME=false"

nvidia-smi >nul 2>&1
if not errorlevel 1 set "HAS_GPU_CLI=true"

docker info 2>nul | findstr /I "nvidia" >nul
if not errorlevel 1 set "HAS_GPU_RUNTIME=true"

if /I "%EXECUTION_MODE%"=="cpu" (
    echo INFO: CPU mode forced by --cpu
    set "PROFILE=--profile cpu"
    goto run_compose
)

if /I "%EXECUTION_MODE%"=="gpu" (
    echo GPU mode requested --gpu. Validating NVIDIA runtime...
    if /I not "%HAS_GPU_CLI%"=="true" (
        echo ERROR: GPU mode failed: nvidia-smi not available or GPU not detected.
        echo Install NVIDIA drivers ^(or run with --cpu^).
        pause
        exit /b 1
    )
    if /I not "%HAS_GPU_RUNTIME%"=="true" (
        echo ERROR: GPU mode failed: Docker NVIDIA runtime is not detected.
        echo Install/configure NVIDIA Container Toolkit, then retry ^(or run with --cpu^).
        pause
        exit /b 1
    )
    echo OK: NVIDIA GPU + Docker runtime detected
    set "PROFILE=--profile gpu"
    goto run_compose
)

echo Checking for NVIDIA GPU...
if /I "%HAS_GPU_CLI%"=="true" (
    if /I "%HAS_GPU_RUNTIME%"=="true" (
        echo OK: NVIDIA GPU + Docker runtime detected
        set "PROFILE=--profile gpu"
    ) else (
        echo WARN: NVIDIA GPU detected but Docker NVIDIA runtime missing; falling back to CPU.
        echo Tip: install NVIDIA Container Toolkit or use --gpu after setup.
        set "PROFILE=--profile cpu"
    )
) else (
    echo INFO: No NVIDIA GPU - using CPU mode
    set "PROFILE=--profile cpu"
)

:run_compose

REM Stop opposite profile container to avoid Cortex-to-Cortex port collisions
if "%PROFILE%"=="--profile gpu" (
    docker compose --profile cpu stop cortex-suite-cpu >nul 2>&1
) else (
    docker compose --profile gpu stop cortex-suite-gpu >nul 2>&1
)

call :check_port_in_use "%UI_PORT%" "Streamlit UI"
if errorlevel 1 call :find_next_free_port "%UI_PORT%" UI_PORT "Streamlit UI"
call :check_port_in_use "%API_PORT%" "API"
if errorlevel 1 call :find_next_free_port "%API_PORT%" API_PORT "API"
call :check_port_in_use "%OLLAMA_PORT%" "Ollama"
if errorlevel 1 call :find_next_free_port "%OLLAMA_PORT%" OLLAMA_PORT "Ollama"
goto after_port_checks

:after_port_checks
set "CORTEX_UI_PORT=%UI_PORT%"
set "CORTEX_API_PORT=%API_PORT%"
set "CORTEX_OLLAMA_PORT=%OLLAMA_PORT%"

REM Build and start services
echo.
echo Starting Cortex Suite services...
echo Command: %COMPOSE_CMD% %PROFILE% up -d --build
echo.

%COMPOSE_CMD% %PROFILE% up -d --build

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start services
    echo Check the error messages above for details.
    pause
    exit /b 1
)

echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service health
echo.
echo Checking service status...
%COMPOSE_CMD% ps

echo.
echo ===============================================
echo    Cortex Suite is starting!
echo ===============================================
echo.
echo Access your Cortex Suite at:
echo   Main Application: http://localhost:%UI_PORT%
echo   API Documentation: http://localhost:%API_PORT%/docs
echo.
echo Useful commands:
echo   View logs:     docker compose logs -f
echo   Stop:          %~nx0 --stop
echo   Remove:        %~nx0 --down
echo.
if "%STORAGE_MODE%"=="portable" (
    echo Storage: PORTABLE - Docker volumes
) else (
    echo Storage: EXTERNAL - !EXTERNAL_AI_PATH!
)
echo.
pause
exit /b 0

:check_port_in_use
set "_CHK_PORT=%~1"
set "_CHK_LABEL=%~2"
netstat -ano | findstr /R /C:":%_CHK_PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo ERROR: Cannot start: host port %_CHK_PORT% - %_CHK_LABEL% is already in use.
    exit /b 1
)
exit /b 0

:find_next_free_port
set "_START_PORT=%~1"
set "_TARGET_VAR=%~2"
set "_TARGET_LABEL=%~3"
set /a "_CANDIDATE=%_START_PORT%+1"
set /a "_MAX_PORT=%_START_PORT%+200"

:find_next_free_port_loop
if %_CANDIDATE% GTR %_MAX_PORT% (
    echo ERROR: Could not find free port for %_TARGET_LABEL% near %_START_PORT%.
    echo Please set CORTEX_UI_PORT / CORTEX_API_PORT / CORTEX_OLLAMA_PORT manually in .env
    pause
    exit /b 1
)

call :check_port_in_use "%_CANDIDATE%" "%_TARGET_LABEL%"
if errorlevel 1 (
    set /a "_CANDIDATE+=1"
    goto find_next_free_port_loop
)

for %%V in ("%_TARGET_VAR%") do set "%%~V=%_CANDIDATE%"
echo WARN: %_TARGET_LABEL% port %_START_PORT% is busy; using %_CANDIDATE% instead.
exit /b 0
