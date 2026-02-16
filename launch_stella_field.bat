@echo off
title Stella Octangula Field Agent
echo ========================================================================
echo  STELLA OCTANGULA FIELD AGENT
echo  The novelty signal is the poppit that moves through cube-space.
echo ========================================================================
echo.

:: Resolve Python — try Sanctuary venv, fall back to system python
set PYEXE=C:\Sanctuary\venv\Scripts\python.exe
if not exist "%PYEXE%" (
    echo [WARN] Sanctuary venv not found at %PYEXE%
    set PYEXE=python
)

:: Ensure we're in the MBD-Framework root
cd /d "%~dp0"

:: Inject the project root into sys.path so dynamics/ imports resolve
"%PYEXE%" -c "import sys; sys.path.insert(0, '.'); exec(open('dynamics/field_agent.py').read())"

echo.
echo ========================================================================
echo  Field agent complete. Plastic deformation is permanent.
echo ========================================================================
pause
