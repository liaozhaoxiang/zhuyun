@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 未找到虚拟环境，正在创建...
    D:\Anaconda\python.exe -m venv .venv
    echo [2/3] 安装依赖（首次约 3-8 分钟）...
    .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo [3/3] 启动筑安云...
start "" ".venv\Scripts\pythonw.exe" main.py
timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:8011/"

echo.
echo 服务已启动：http://127.0.0.1:8011/
echo 首次启动需要构建索引，页面右上角会显示进度。
echo 关闭服务请双击「停止筑安云.bat」。
timeout /t 3 /nobreak >nul
