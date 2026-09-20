@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Llama Studio

set "PY="
rem Test interpreters by running them - Microsoft Store stubs answer `where`
rem but fail to execute, so a PATH lookup alone is not enough.
for %%C in ("py -3.12" "py -3.11" "py -3.10" "py -3" "python") do (
  if not defined PY (
    %%~C -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "PY=%%~C"
  )
)

if not defined PY (
  echo.
  echo   Python 3.10 or newer was not found.
  echo   Install it from https://www.python.org/downloads/ and tick
  echo   "Add python.exe to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)

echo   Using: %PY%
%PY% -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 (
  echo   Could not install the Python packages. Check your internet connection.
  pause
  exit /b 1
)

%PY% server.py
pause
