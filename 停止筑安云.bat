@echo off
chcp 65001 >nul
echo 正在停止筑安云（端口 8011）...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8011" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)
echo 已停止。
timeout /t 2 /nobreak >nul
