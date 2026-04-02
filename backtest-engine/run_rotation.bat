@echo off
setlocal

if "%~1"=="" (
  python "%~dp0..\scripts\rotation_backtest.py" --pick
) else (
  python "%~dp0..\scripts\rotation_backtest.py" %*
)

pause
