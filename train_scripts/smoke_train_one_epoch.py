import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Task:
    key: str
    model: str
    kind: str


def run_prepare_dataset(root: Path) -> None:
    cmd = [
        sys.executable,
        str(root / "scripts" / "prepare_rgb_yolo.py"),
        "--root",
        str(root),
        "--clear",
        "--limit-dut",
        "2",
        "--limit-dronedet",
        "2",
        "--limit-ard",
        "2",
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "write_data_yaml.py"),
            "--root",
            str(root),
            "--out",
            str(root / "configs" / "drone_rgb_abs.yaml"),
        ],
        check=True,
    )


def safe_import_detectron2() -> bool:
    try:
        import detectron2  # noqa: F401

        return True
    except Exception:
        return False


def train_ultralytics(
    task: Task,
    root: Path,
    seed: int,
    epochs: int,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
) -> Dict:
    from ultralytics import YOLO

    data_yaml = str(root / "configs" / "drone_rgb_abs.yaml")
    runs_dir = root / "runs_smoke"
    runs_dir.mkdir(parents=True, exist_ok=True)

    name = f"{task.key}_seed{seed}"
    model = YOLO(task.model)
    r = model.train(
        data=data_yaml,
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        seed=int(seed),
        device=device,
        workers=int(workers),
        project=str(runs_dir),
        name=name,
        verbose=False,
        plots=False,
    )

    best = runs_dir / name / "weights" / "best.pt"
    val_res = model.val(
        model=str(best) if best.exists() else task.model,
        data=data_yaml,
        split="val",
        device=device,
        workers=int(workers),
        verbose=False,
        plots=False,
    )
    test_res = model.val(
        model=str(best) if best.exists() else task.model,
        data=data_yaml,
        split="test",
        device=device,
        workers=int(workers),
        verbose=False,
        plots=False,
    )

    return {
        "train": {"result": str(r)},
        "best": str(best) if best.exists() else None,
        "val": getattr(val_res, "results_dict", None) or {},
        "test": getattr(test_res, "results_dict", None) or {},
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    epochs = int(os.environ.get("EPOCHS", "1"))
    device = os.environ.get("DEVICE", "cpu")
    imgsz = int(os.environ.get("IMGSZ", "320"))
    batch = int(os.environ.get("BATCH", "2"))
    workers = int(os.environ.get("WORKERS", "0"))
    seeds = [1, 2, 3]

    run_prepare_dataset(root)

    models_root = root.parent / "models"
    tasks: List[Task] = [
        Task("yolo11n", str(models_root / "yolo" / "yolo11n.pt"), "ultralytics"),
        Task("yolov8n", str(models_root / "yolo" / "yolov8n.pt"), "ultralytics"),
        Task("yolov10n", str(models_root / "yolov10" / "yolov10n.pt"), "ultralytics"),
        Task("rtdetr_l", "rtdetr-l.pt", "ultralytics"),
        Task("ddw_yolo", str(models_root / "yolo" / "yolo11n.pt"), "ultralytics"),
        Task("faster_rcnn", "detectron2", "detectron2"),
    ]

    summary: Dict[str, Dict] = {
        "root": str(root),
        "epochs": epochs,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "seeds": seeds,
        "tasks": {},
    }

    for task in tasks:
        summary["tasks"][task.key] = {"kind": task.kind, "model": task.model, "runs": {}}
        for seed in seeds:
            run_key = str(seed)
            try:
                if task.kind == "detectron2":
                    ok = safe_import_detectron2()
                    if not ok:
                        summary["tasks"][task.key]["runs"][run_key] = {"status": "failed", "error": "detectron2 not installed"}
                        continue
                    summary["tasks"][task.key]["runs"][run_key] = {"status": "skipped", "reason": "detectron2 training not executed in smoke test"}
                    continue

                res = train_ultralytics(
                    task=task,
                    root=root,
                    seed=seed,
                    epochs=epochs,
                    device=device,
                    imgsz=imgsz,
                    batch=batch,
                    workers=workers,
                )
                summary["tasks"][task.key]["runs"][run_key] = {"status": "ok", "metrics": res}
            except Exception as e:
                summary["tasks"][task.key]["runs"][run_key] = {"status": "failed", "error": repr(e)}

    out = root / "runs_smoke" / "smoke_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
