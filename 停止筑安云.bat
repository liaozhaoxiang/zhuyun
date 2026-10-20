@echo off
chcp 936 >nul
cd /d "%~dp0"
title 筑安云 - 停止

netstat -ano | findstr /r /c:":8011 .*LISTENING" >nul
if errorlevel 1 (
    echo 端口 8011 上没有服务在跑，不用停。
    timeout /t 3 /nobreak >nul
    exit /b 0
)

echo 正在停止筑安云，端口 8011...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8011" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)
echo 已停止。
timeout /t 2 /nobreak >nul
