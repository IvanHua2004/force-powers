@echo off
rem Force Powers launcher: installs the libraries on first run, then starts the game.
cd /d "%~dp0"

if exist ".venv\installed" goto :play

echo First run: setting up Python libraries. This takes a few minutes...
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :fail
echo ok> ".venv\installed"

:play
".venv\Scripts\python.exe" game.py %*
exit /b %errorlevel%

:fail
echo.
echo Setup failed. Install Python 3.12 from https://www.python.org/downloads/ and run this again.
pause
exit /b 1
