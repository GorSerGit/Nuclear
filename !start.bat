@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set "MTMC_PY=.venv\Scripts\python.exe"
if not exist "%MTMC_PY%" call :bootstrap
if not exist "%MTMC_PY%" goto failed
"%MTMC_PY%" -c "import sys,struct; sys.exit(sys.version_info[:2] != (3,13) or struct.calcsize('P') != 8)"
if errorlevel 1 goto failed
"%MTMC_PY%" scripts\office_dependencies.py --prepare
if errorlevel 1 goto failed
if /i "%~1"=="--dependencies" goto dependencies
if not "%~1"=="" goto cli
"%MTMC_PY%" scripts\office_web.py
goto done
:dependencies
"%MTMC_PY%" scripts\office_web.py --dependencies
goto done
:cli
"%MTMC_PY%" scripts\office.py %*
:done
set "MTMC_EXIT=%errorlevel%"
pause
exit /b %MTMC_EXIT%

:bootstrap
call :create_venv
if exist "%MTMC_PY%" exit /b
where winget >nul 2>&1
if errorlevel 1 exit /b
winget install --id Python.Python.3.13 --exact --scope user --architecture x64 --accept-package-agreements --accept-source-agreements
call :create_venv
exit /b

:create_venv
set "MTMC_LOCAL_PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if exist "%MTMC_LOCAL_PY%" (
    "%MTMC_LOCAL_PY%" -c "import struct; assert struct.calcsize('P') == 8" >nul 2>&1
    if not errorlevel 1 "%MTMC_LOCAL_PY%" -m venv .venv
)
if exist "%MTMC_PY%" exit /b
py -3.13-64 -m venv .venv
if exist "%MTMC_PY%" exit /b
python -c "import sys,struct; sys.exit(sys.version_info[:2] != (3,13) or struct.calcsize('P') != 8)" >nul 2>&1
if not errorlevel 1 python -m venv .venv
exit /b

:failed
echo Нужен Python 3.13 x64. Если .venv перенесена с другой машины, переименуйте её и запустите снова.
pause
exit /b 1
