@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  Stella Octangula Field Diagram
REM  Interactive 3D visualization of the Influence Cube
REM
REM  Nature · Nurture · Heaven · Home  ↔  Displacement · Fixation · Degeneration · Capture
REM
REM  Drag to rotate | Scroll to zoom | Close window to exit
REM ═══════════════════════════════════════════════════════════════════════

cd /d "%~dp0"

REM --- Locate Python ---
set "PYEXE=C:\Sanctuary\venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] Python not found at %PYEXE%
    echo Please set PYEXE in this script to your Python environment.
    pause
    exit /b 1
)

echo.
echo   ╔═══════════════════════════════════════════════╗
echo   ║   THE STELLA OCTANGULA: INFLUENCE CUBE FIELD  ║
echo   ║                                               ║
echo   ║   Nature  Nurture  Heaven  Home               ║
echo   ║     ↕       ↕        ↕      ↕                ║
echo   ║   Displ.  Fixation  Degen. Capture            ║
echo   ║                                               ║
echo   ║   Drag to rotate · Scroll to zoom             ║
echo   ╚═══════════════════════════════════════════════╝
echo.

"%PYEXE%" visualize_stella.py

if errorlevel 1 (
    echo.
    echo [ERROR] Visualization failed. Check Python dependencies.
    pause
)
