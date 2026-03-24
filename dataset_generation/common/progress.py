from __future__ import annotations

from datetime import datetime


def log_progress(stage: str, message: str) -> None:
    """输出带时间戳的阶段进度日志，避免长流程静默。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{stage}] {message}", flush=True)