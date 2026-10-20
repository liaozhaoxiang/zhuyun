"""入口：python main.py"""
from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pythonw（启动脚本用）没有 stdout/stderr，uvicorn 日志写 None 会直接崩；重定向到日志文件
if sys.stdout is None or sys.stderr is None:
    _log_dir = Path(__file__).resolve().parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    _log_f = open(_log_dir / "service.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log_f
    sys.stderr = sys.stderr or _log_f

import uvicorn

from app.config import settings

if __name__ == "__main__":
    if "--open" in sys.argv:
        import threading

        threading.Timer(2.5, lambda: webbrowser.open(f"http://{settings.host}:{settings.port}/")).start()
    print(f"{settings.app_name} {settings.version} -> http://{settings.host}:{settings.port}")
    uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=False, log_level="info", ws_ping_interval=30)
