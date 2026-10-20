@echo off
chcp 936 >nul
cd /d "%~dp0"
title 筑安云 - 启动

netstat -ano | findstr /r /c:":8011 .*LISTENING" >nul
if not errorlevel 1 (
    echo 服务已经在运行：http://127.0.0.1:8011/
    start "" "http://127.0.0.1:8011/"
    timeout /t 3 /nobreak >nul
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 未找到虚拟环境，正在创建...
    D:\Anaconda\python.exe -m venv .venv
    echo [2/3] 安装依赖，首次约 3-8 分钟...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo [3/3] 启动筑安云，正在等端口就绪...
start "" ".venv\Scripts\pythonw.exe" main.py

for /l %%i in (1,1,40) do (
    netstat -ano | findstr /r /c:":8011 .*LISTENING" >nul
    if not errorlevel 1 goto ready
    timeout /t 1 /nobreak >nul
)

echo.
echo 启动失败：等了 40 秒端口 8011 仍没起来。
echo 下面是 logs\service.log 的最后 25 行：
echo ------------------------------------------------------------
if exist "logs\service.log" (
    powershell -NoProfile -Command "Get-Content 'logs\service.log' -Tail 25"
) else (
    echo 没有生成日志文件，说明 python 进程根本没跑起来。
)
echo ------------------------------------------------------------
pause
exit /b 1

:ready
start "" "http://127.0.0.1:8011/"
echo 服务已启动：http://127.0.0.1:8011/
echo 关闭服务请双击「停止筑安云.bat」。
timeout /t 4 /nobreak >nul
exit /b 0
