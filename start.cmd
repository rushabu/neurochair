@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Create the project environment first. See README.md Setup.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0server.py" %*
