@echo off
title Stella Octangula Field Agent
echo ========================================================================
echo  STELLA OCTANGULA FIELD AGENT
echo  The novelty signal is the poppit that moves through cube-space.
echo ========================================================================
echo.

:: Resolve Python — prefer a local venv, fall back to PATH
set "PYEXE=python"
if exist "%~dp0venv\Scripts\python.exe" set "PYEXE=%~dp0venv\Scripts\python.exe"

:: Ensure we're in the MBD-Framework root
cd /d "%~dp0"

"%PYEXE%" visualize_stella.py

echo.
echo ========================================================================
echo  Field agent complete. Plastic deformation is permanent.
echo ========================================================================
pause
