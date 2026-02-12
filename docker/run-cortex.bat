@echo off
REM Backward-compatible alias to the maintained compose launcher.
setlocal
call "%~dp0run-compose.bat" %*
endlocal
