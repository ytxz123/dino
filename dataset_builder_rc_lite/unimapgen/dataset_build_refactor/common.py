"""通用 IO 和 ShareGPT helper。"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_event(stage: str, message: str) -> None:
    print(f"[{timestamp_text()}] [{str(stage)}] {str(message)}", flush=True)


def log_warning(stage: str, message: str) -> None:
    log_event(stage, f"WARNING: {message}")


def log_error(stage: str, message: str) -> None:
    log_event(stage, f"ERROR: {message}")


def format_progress(current: int, total: int) -> str:
    safe_current = max(0, int(current))
    safe_total = max(0, int(total))
    if safe_total <= 0:
        return f"{safe_current}/?"
    ratio = 100.0 * float(safe_current) / float(safe_total)
    return f"{safe_current}/{safe_total} ({ratio:.1f}%)"


def validate_ratio(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def require_existing_path(path: Path, *, kind: str) -> Path:
    resolved = Path(path).resolve()
    if kind == "file" and not resolved.is_file():
        raise FileNotFoundError(f"Required file does not exist: {resolved}")
    if kind == "dir" and not resolved.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {resolved}")
    return resolved


def load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"JSON file does not exist: {Path(path).resolve()}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {Path(path).resolve()}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sanitize_name(name: str) -> str:
    output = []
    for char in str(name):
        output.append(char if char.isalnum() or char in ("_", "-") else "_")
    return "".join(output).strip("_") or "dataset"


def extract_message_content(row: Dict[str, Any], role: str) -> str:
    wanted_role = str(role).strip().lower()
    for message in row.get("messages", []):
        if str(message.get("role", "")).strip().lower() == wanted_role:
            return str(message.get("content", ""))
    return ""


def resolve_optional_text(*, inline_text: str = "", file_path: str = "", fallback: str = "") -> str:
    if str(file_path).strip():
        return Path(str(file_path)).read_text(encoding="utf-8").strip()
    if str(inline_text).strip():
        return str(inline_text).strip()
    return str(fallback).strip()


def link_or_copy_images(input_root: Path, output_root: Path, mode: str) -> str:
    src = input_root / "images"
    dst = output_root / "images"
    if not src.exists() or str(mode) == "none":
        return "none"
    if dst.exists() or dst.is_symlink():
        return "existing"
    if str(mode) == "symlink":
        try:
            dst.symlink_to(src, target_is_directory=True)
            return "symlink"
        except OSError:
            shutil.copytree(src, dst)
            return "copy_fallback"
    shutil.copytree(src, dst)
    return "copy"


def build_sharegpt_dataset_info(output_root: Path, splits: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    prefix = sanitize_name(output_root.name)
    info: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        info[f"unimapgen_{prefix}_{split}"] = {
            "file_name": str((output_root / f"{split}.jsonl").resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return info


def make_sharegpt_record(
    *,
    sample_id: str,
    image_rel_path: str,
    user_text: str,
    assistant_payload: Any,
    system_prompt: str = "",
) -> Dict[str, Any]:
    assistant_text = assistant_payload if isinstance(assistant_payload, str) else json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict[str, str]] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(user_text)})
    messages.append({"role": "assistant", "content": assistant_text})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}
