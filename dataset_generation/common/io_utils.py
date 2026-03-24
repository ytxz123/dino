from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def ensure_directory(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    """读取单个 JSON 文件。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Dict) -> None:
    """写出带缩进的 JSON 文件。"""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict]:
    """读取 JSONL 文件。"""
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    """写出 JSONL 文件，并返回写出的记录数。"""
    ensure_directory(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_images_field(images_value) -> List[str]:
    """把 ShareGPT 的 images 字段统一成字符串列表。"""
    if images_value is None:
        return []
    if isinstance(images_value, str):
        text = images_value.strip()
        return [text] if text else []
    if isinstance(images_value, Sequence):
        out: List[str] = []
        for value in images_value:
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
        return out
    return []


def build_sharegpt_dataset_info(dataset_root: Path, prefix: str, splits: Sequence[str]) -> Dict[str, Dict]:
    """生成 LLaMAFactory 兼容的 dataset_info.json 内容。"""
    registry: Dict[str, Dict] = {}
    for split in splits:
        split_name = str(split)
        data_file = dataset_root / f"{split_name}.jsonl"
        if not data_file.is_file():
            continue
        registry[f"{prefix}_{split_name}"] = {
            "file_name": str(data_file.resolve()),
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
    return registry