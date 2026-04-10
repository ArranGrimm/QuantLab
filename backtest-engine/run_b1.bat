@echo off
setlocal

if "%~1"=="" (
  python "%~dp0..\scripts\b1_backtest.py" --pick
) else (
  python "%~dp0..\scripts\b1_backtest.py" %*
)

pause
