from __future__ import annotations

import json
from typing import Dict, List, Sequence


def build_patch_only_record(image_rel_path: str, target_lines: Sequence[Dict], sample_id: str, system_prompt: str, prompt_template: str) -> Dict:
    """构造 stage_a 的 ShareGPT 样本。"""
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_template)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}



def build_state_record(image_rel_path: str, state_lines: Sequence[Dict], target_lines: Sequence[Dict], sample_id: str, system_prompt: str, prompt_template: str) -> Dict:
    """构造 stage_b 的 ShareGPT 样本。"""
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_template).format(state_json=state_json)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}
