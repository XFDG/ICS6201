#!/usr/bin/env python3
"""多 GPU 并行训练启动器 — 将 18 个训练任务分配到 N 张 GPU 并行执行。"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainTask:
    run_key: str
    kind: str       # "ultralytics" | "detectron2"
    model: str
    epochs: int
    seed: int
    gpu_id: int
    imgsz: int = 640
    batch: int = -1
    workers: int = 16


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def detect_gpus() -> List[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return []


def build_tasks(
    gpu_ids: List[int],
    mode: str,
    seeds: List[int],
    data_yaml: str,
    ddw_model: Optional[str],
    frcnn_available: bool,
) -> List[TrainTask]:
    epochs_yolo = int(os.environ.get("EPOCHS_YOLO", "200" if mode == "formal" else "1"))
    epochs_rtdetr = int(os.environ.get("EPOCHS_RTDETR", "120" if mode == "formal" else "1"))
    epochs_frcnn = int(os.environ.get("EPOCHS_FASTER_RCNN", "120" if mode == "formal" else "1"))

    yolo_models = [
        ("yolo11", "ultralytics", os.environ.get("YOLO11_MODEL", "yolo11m.pt")),
        ("yolov8", "ultralytics", os.environ.get("YOLOV8_MODEL", "yolov8n.pt")),
        ("yolov10", "ultralytics", os.environ.get("YOLOV10_MODEL", "yolov10n.pt")),
        ("rtdetr", "ultralytics", os.environ.get("RTDETR_MODEL", "rtdetr-l.pt")),
    ]

    if ddw_model and Path(ddw_model).exists():
        yolo_models.append(("ddw_yolo", "ultralytics", ddw_model))

    epoch_map = {
        "yolo11": epochs_yolo, "yolov8": epochs_yolo, "yolov10": epochs_yolo,
        "ddw_yolo": epochs_yolo, "rtdetr": epochs_rtdetr,
    }

    tasks: List[TrainTask] = []
    for i, seed in enumerate(seeds):
        gpu = gpu_ids[i % len(gpu_ids)]
        for key, kind, model in yolo_models:
            tasks.append(TrainTask(
                run_key=f"{key}_seed{seed}",
                kind=kind,
                model=model,
                epochs=epoch_map.get(key, epochs_yolo),
                seed=seed,
                gpu_id=gpu,
                batch=int(os.environ.get("BATCH", "-1")),
                workers=int(os.environ.get("WORKERS", "16")),
            ))

    if frcnn_available:
        for i, seed in enumerate(seeds):
            gpu = gpu_ids[(i + len(yolo_models)) % len(gpu_ids)]
            tasks.append(TrainTask(
                run_key=f"faster_rcnn_seed{seed}",
                kind="detectron2",
                model="detectron2",
                epochs=epochs_frcnn,
                seed=seed,
                gpu_id=gpu,
            ))

    return tasks


def run_ultralytics_task(task: TrainTask, data_yaml: str, runs_dir: Path, log_dir: Path) -> Tuple[int, str]:
    log_path = log_dir / f"{task.run_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "train_scripts" / "run_ultralytics_task.py"),
        "--model", task.model,
        "--data", data_yaml,
        "--epochs", str(task.epochs),
        "--seed", str(task.seed),
        "--project", str(runs_dir),
        "--name", task.run_key,
        "--imgsz", str(task.imgsz),
        "--batch", str(task.batch),
        "--device", str(task.gpu_id),
        "--workers", str(task.workers),
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {now_ts()} START gpu={task.gpu_id} =====\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=f)
        f.write(f"\n===== {now_ts()} END code={p.returncode} =====\n")

    return p.returncode, str(log_path)


def run_detectron2_task(task: TrainTask, data_yaml: str, runs_dir: Path, log_dir: Path) -> Tuple[int, str]:
    log_path = log_dir / f"{task.run_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(task.gpu_id)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_detectron2_fasterrcnn.py"),
        "--root", str(ROOT),
        "--seed", str(task.seed),
        "--epochs", str(task.epochs),
        "--ims-per-batch", os.environ.get("D2_IMS_PER_BATCH", "8"),
        "--num-workers", os.environ.get("D2_NUM_WORKERS", "8"),
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {now_ts()} START gpu={task.gpu_id} =====\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=f, env=env)
        f.write(f"\n===== {now_ts()} END code={p.returncode} =====\n")

    return p.returncode, str(log_path)


def ensure_data(root: Path, skip_ard: bool) -> str:
    yolo_train = root / "yolo" / "images" / "train"
    if not yolo_train.exists():
        print("[prepare] Generating YOLO dataset...")
        cmd = [sys.executable, str(root / "scripts" / "prepare_rgb_yolo.py"), "--root", str(root), "--clear"]
        if skip_ard:
            cmd.append("--skip-ard")
        subprocess.run(cmd, check=True)

    data_yaml = root / "configs" / "drone_rgb_abs.yaml"
    subprocess.run(
        [sys.executable, str(root / "scripts" / "write_data_yaml.py"), "--root", str(root), "--out", str(data_yaml)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(root / "scripts" / "prepare_rgb_coco.py"), "--root", str(root)],
        check=False,
    )
    return str(data_yaml)


def check_detectron2() -> bool:
    try:
        import detectron2  # noqa: F401
        return True
    except Exception:
        return False


def last_json_line(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    last = s
        if not last:
            return None
        return json.loads(last)
    except Exception:
        return None


def process_tasks(
    tasks: List[TrainTask],
    data_yaml: str,
    runs_dir: Path,
    log_dir: Path,
    state_path: Path,
) -> Dict[str, Any]:
    max_workers = len(set(t.gpu_id for t in tasks))
    state: Dict[str, Any] = {
        "started_at": now_ts(),
        "total_tasks": len(tasks),
        "gpus_used": max_workers,
        "data_yaml": data_yaml,
        "runs_dir": str(runs_dir),
        "items": {},
    }

    for t in tasks:
        state["items"][t.run_key] = {
            "task": t.run_key, "kind": t.kind, "model": t.model,
            "seed": t.seed, "gpu": t.gpu_id, "epochs": t.epochs,
            "status": "pending",
        }

    def write_state():
        state["updated_at"] = now_ts()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_state()

    def run_one(task: TrainTask) -> Tuple[str, int, str, Optional[Dict]]:
        item = state["items"][task.run_key]
        item["status"] = "running"
        item["start_at"] = now_ts()
        item["gpu"] = task.gpu_id
        write_state()

        t0 = time.time()
        if task.kind == "detectron2":
            code, log = run_detectron2_task(task, data_yaml, runs_dir, log_dir)
        else:
            code, log = run_ultralytics_task(task, data_yaml, runs_dir, log_dir)
        elapsed = time.time() - t0

        item["elapsed_sec"] = round(elapsed, 1)
        item["end_at"] = now_ts()
        item["status"] = "ok" if code == 0 else "failed"
        if code != 0:
            item["error"] = f"exit_code={code}"

        metrics = last_json_line(Path(log))
        if metrics:
            item["metrics"] = metrics
        write_state()
        return task.run_key, code, item["status"], metrics

    print(f"\n[launch] {len(tasks)} tasks on {max_workers} GPU(s)\n")
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for f in as_completed(futures):
            key, code, status, _ = f.result()
            results[key] = {"code": code, "status": status}
            ok = sum(1 for v in state["items"].values() if v["status"] == "ok")
            fail = sum(1 for v in state["items"].values() if v["status"] == "failed")
            done = ok + fail
            print(f"  [{done}/{len(tasks)}] {key}: {status}")

    state["summary"] = {
        "ok": sum(1 for v in state["items"].values() if v["status"] == "ok"),
        "failed": sum(1 for v in state["items"].values() if v["status"] == "failed"),
        "completed_at": now_ts(),
    }
    write_state()
    return state


def print_summary(state: Dict[str, Any]) -> None:
    s = state.get("summary", {})
    print(f"\n{'='*60}")
    print(f"Training complete: {s.get('ok',0)} OK, {s.get('failed',0)} failed")
    print(f"State file: {state.get('_state_path','')}")
    for key, item in sorted(state.get("items", {}).items()):
        status = item.get("status", "?")
        elapsed = item.get("elapsed_sec", "?")
        gpu = item.get("gpu", "?")
        mark = "OK" if status == "ok" else "FAIL"
        print(f"  [{mark}] {key}  gpu={gpu}  {elapsed}s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "formal"], required=True)
    p.add_argument("--skip-ard", action="store_true")
    p.add_argument("--seeds", default=os.environ.get("SEEDS", "1 2 3"))
    p.add_argument("--data", default=os.environ.get("DATA_YAML", ""))
    args = p.parse_args()

    gpu_ids = detect_gpus()
    if not gpu_ids:
        print("[FATAL] No GPU detected", file=sys.stderr)
        return 1

    print(f"[GPU] Available: {gpu_ids}")

    data_yaml = args.data or ensure_data(ROOT, skip_ard=args.skip_ard)
    seeds = [int(x) for x in args.seeds.split() if x.strip()]
    ddw_model = os.environ.get("DDW_MODEL")
    if not ddw_model:
        candidate = ROOT / "models" / "ddw_yolo11m_p2_bifpn_eca.yaml"
        if candidate.exists():
            ddw_model = str(candidate.resolve())

    frcnn_ok = check_detectron2()
    if not frcnn_ok:
        print("[WARN] detectron2 not available, skipping Faster R-CNN")

    tasks = build_tasks(gpu_ids, args.mode, seeds, data_yaml, ddw_model, frcnn_ok)

    suffix = "formal" if args.mode == "formal" else "smoke"
    runs_dir = ROOT / f"runs_{suffix}"
    log_dir = ROOT / "logs" / suffix
    state_path = ROOT / "logs" / f"state_{suffix}.json"

    state = process_tasks(tasks, data_yaml, runs_dir, log_dir, state_path)
    state["_state_path"] = str(state_path)
    print_summary(state)

    failed = state["summary"].get("failed", 0)
    return 2 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
