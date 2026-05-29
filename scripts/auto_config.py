#!/usr/bin/env python3
"""自动检测可用 GPU，输出 JSON 配置供后续脚本使用。"""

import json
import sys
from typing import List, Dict, Any


def detect_gpus() -> Dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpus": [], "count": 0, "error": "CUDA not available"}

        count = torch.cuda.device_count()
        gpus: List[Dict[str, Any]] = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "total_memory_mb": props.total_memory // (1024 * 1024),
            })

        return {"gpus": gpus, "count": count}
    except Exception as e:
        return {"gpus": [], "count": 0, "error": str(e)}


def main() -> int:
    info = detect_gpus()
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if info["count"] == 0:
        print("[WARN] No GPU detected", file=sys.stderr)
        return 1

    print(f"[OK] Detected {info['count']} GPU(s):", file=sys.stderr)
    for g in info["gpus"]:
        print(f"  GPU {g['id']}: {g['name']} ({g['total_memory_mb']} MB)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
