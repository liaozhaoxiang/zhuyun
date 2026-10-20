"""入口：python main.py"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from app.config import settings

if __name__ == "__main__":
    if "--open" in sys.argv:
        import threading

        threading.Timer(2.5, lambda: webbrowser.open(f"http://{settings.host}:{settings.port}/")).start()
    print(f"{settings.app_name} {settings.version} -> http://{settings.host}:{settings.port}")
    uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=False, log_level="info", ws_ping_interval=30)
