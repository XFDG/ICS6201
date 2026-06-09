#!/usr/bin/env python3
"""Multi-GPU training launcher with per-GPU worker queues and resume support.

Tasks are pulled by per-GPU worker threads from a single shared queue. A task
runs on whichever GPU's worker picks it up — tasks are never pre-bound to a
future GPU, which prevents OOM when a short task on a busy card finishes first.

Recovery mode (``--target primary``, ``--resume-existing``, ``--skip-complete``)
consumes a manifest produced by ``scripts/recovery_manifest.py`` and only
schedules tasks that are not yet complete. Primary tasks
(``rtdetr_seed{1,2,3}``, ``faster_rcnn_seed{1,2,3}``) are scheduled before
secondary; ``--no-secondary`` blocks secondary entirely.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_EPOCH_RE = re.compile(r"^\s*(\d+)/(\d+)\s+\S+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


ROOT = Path(__file__).resolve().parents[1]


PRIMARY_FAMILIES = {"rtdetr", "faster_rcnn"}
PRIMARY_ORDER = ["rtdetr", "faster_rcnn"]
SECONDARY_ORDER = ["yolo11", "ddw_yolo", "yolov10", "yolov8"]


@dataclass
class TrainTask:
    run_key: str
    kind: str       # "ultralytics" | "detectron2"
    model: str
    epochs: int
    seed: int
    family: str
    priority: str   # "primary" | "secondary"
    imgsz: int = 640
    batch: int = -1
    workers: int = 16
    optimizer: str = ""
    resume_path: Optional[str] = None     # ultralytics last.pt
    run_dir: Optional[str] = None         # ultralytics canonical run dir name
    detectron_resume: bool = False        # use Detectron2 resume_or_load(resume=True)


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


def parse_gpu_ids(value: str) -> List[int]:
    ids: List[int] = []
    for part in str(value or "").replace(" ", ",").split(","):
        s = part.strip()
        if not s:
            continue
        ids.append(int(s))
    return ids


def _family_of(run_key: str) -> str:
    if run_key.startswith("faster_rcnn"):
        return "faster_rcnn"
    return run_key.split("_seed")[0]


def _priority_of(family: str) -> str:
    return "primary" if family in PRIMARY_FAMILIES else "secondary"


def _model_for_family(family: str) -> str:
    return {
        "rtdetr": os.environ.get("RTDETR_MODEL", "rtdetr-l.pt"),
        "yolo11": os.environ.get("YOLO11_MODEL", "yolo11m.pt"),
        "yolov10": os.environ.get("YOLOV10_MODEL", "yolov10n.pt"),
        "yolov8": os.environ.get("YOLOV8_MODEL", "yolov8n.pt"),
    }.get(family, family)


def build_tasks_fresh(
    mode: str,
    seeds: List[int],
    ddw_model: Optional[str],
    frcnn_available: bool,
) -> List[TrainTask]:
    epochs_yolo = int(os.environ.get("EPOCHS_YOLO", "200" if mode == "formal" else "1"))
    epochs_rtdetr = int(os.environ.get("EPOCHS_RTDETR", "120" if mode == "formal" else "1"))
    epochs_frcnn = int(os.environ.get("EPOCHS_FASTER_RCNN", "120" if mode == "formal" else "1"))

    epoch_map = {
        "yolo11": epochs_yolo, "yolov8": epochs_yolo, "yolov10": epochs_yolo,
        "ddw_yolo": epochs_yolo, "rtdetr": epochs_rtdetr, "faster_rcnn": epochs_frcnn,
    }

    families = list(PRIMARY_ORDER) + list(SECONDARY_ORDER)
    if not frcnn_available and "faster_rcnn" in families:
        families.remove("faster_rcnn")
    if not (ddw_model and Path(ddw_model).exists()) and "ddw_yolo" in families:
        families.remove("ddw_yolo")

    tasks: List[TrainTask] = []
    for family in families:
        if family == "faster_rcnn":
            for seed in seeds:
                tasks.append(TrainTask(
                    run_key=f"faster_rcnn_seed{seed}",
                    kind="detectron2",
                    model="detectron2",
                    epochs=epochs_frcnn,
                    seed=seed,
                    family=family,
                    priority="primary",
                ))
            continue
        model = ddw_model if family == "ddw_yolo" else _model_for_family(family)
        optimizer = "AdamW" if family == "ddw_yolo" else os.environ.get("OPTIMIZER", "")
        for seed in seeds:
            tasks.append(TrainTask(
                run_key=f"{family}_seed{seed}",
                kind="ultralytics",
                model=model,
                epochs=epoch_map.get(family, epochs_yolo),
                seed=seed,
                family=family,
                priority=_priority_of(family),
                batch=int(os.environ.get("BATCH", "-1")),
                workers=int(os.environ.get("WORKERS", "16")),
                optimizer=optimizer,
            ))
    return tasks


def build_tasks_from_manifest(
    manifest_path: Path,
    target: str,
    no_secondary: bool,
    skip_complete: bool,
    resume_existing: bool,
    workers_override: Optional[int] = None,
) -> List[TrainTask]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks: List[TrainTask] = []
    epochs_map = {
        "rtdetr": int(os.environ.get("EPOCHS_RTDETR", "120")),
        "yolo11": int(os.environ.get("EPOCHS_YOLO", "200")),
        "yolov10": int(os.environ.get("EPOCHS_YOLO", "200")),
        "yolov8": int(os.environ.get("EPOCHS_YOLO", "200")),
        "ddw_yolo": int(os.environ.get("EPOCHS_YOLO", "200")),
        "faster_rcnn": int(os.environ.get("EPOCHS_FASTER_RCNN", "120")),
    }

    for run_key, item in obj.get("items", {}).items():
        family = item.get("family") or _family_of(run_key)
        priority = item.get("priority") or _priority_of(family)
        if target == "primary" and priority != "primary":
            continue
        if target == "secondary" and priority != "secondary":
            continue
        if no_secondary and priority == "secondary":
            continue
        if skip_complete and item.get("complete"):
            continue

        seed = item["seed"]
        kind = item["kind"]
        resume_path = item.get("last_checkpoint") if resume_existing else None
        run_dir_name = Path(item["canonical_dir"]).name if item.get("canonical_dir") else None

        if kind == "ultralytics":
            model = _model_for_family(family)
            optimizer = "AdamW" if family == "ddw_yolo" else os.environ.get("OPTIMIZER", "")
            tasks.append(TrainTask(
                run_key=run_key,
                kind="ultralytics",
                model=model,
                epochs=epochs_map.get(family, 200),
                seed=seed,
                family=family,
                priority=priority,
                batch=int(os.environ.get("BATCH", "-1")),
                workers=int(workers_override if workers_override is not None else os.environ.get("WORKERS", "16")),
                optimizer=optimizer,
                resume_path=resume_path if resume_existing else None,
                run_dir=run_dir_name,
            ))
        else:  # detectron2
            tasks.append(TrainTask(
                run_key=run_key,
                kind="detectron2",
                model="detectron2",
                epochs=epochs_map.get("faster_rcnn", 120),
                seed=seed,
                family=family,
                priority=priority,
                detectron_resume=bool(resume_existing and item.get("last_checkpoint")),
            ))

    # Sort: primary first, then by family order, then by seed
    family_order_index = {f: i for i, f in enumerate(PRIMARY_ORDER + SECONDARY_ORDER)}
    tasks.sort(key=lambda t: (
        0 if t.priority == "primary" else 1,
        family_order_index.get(t.family, 99),
        t.seed,
    ))
    return tasks


def run_ultralytics_task(
    task: TrainTask, data_yaml: str, runs_dir: Path, log_dir: Path, gpu_id: int,
) -> Tuple[int, str]:
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
        "--name", task.run_dir or task.run_key,
        "--imgsz", str(task.imgsz),
        "--batch", str(task.batch),
        "--device", str(gpu_id),
        "--workers", str(task.workers),
    ]
    if task.optimizer:
        cmd.extend(["--optimizer", task.optimizer])
    if task.resume_path:
        cmd.extend(["--resume-path", task.resume_path, "--exist-ok"])
    elif task.run_dir:
        cmd.extend(["--exist-ok"])

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {now_ts()} START physical_gpu={gpu_id} resume={task.resume_path or 'none'} run_dir={task.run_dir or task.run_key} =====\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=f, env=env)
        f.write(f"\n===== {now_ts()} END code={p.returncode} =====\n")
    return p.returncode, str(log_path)


def run_detectron2_task(
    task: TrainTask, data_yaml: str, runs_dir: Path, log_dir: Path, gpu_id: int,
) -> Tuple[int, str]:
    log_path = log_dir / f"{task.run_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_detectron2_fasterrcnn.py"),
        "--root", str(ROOT),
        "--seed", str(task.seed),
        "--epochs", str(task.epochs),
        "--ims-per-batch", os.environ.get("D2_IMS_PER_BATCH", "8"),
        "--num-workers", os.environ.get("D2_NUM_WORKERS", "8"),
    ]
    if task.detectron_resume:
        cmd.append("--resume")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {now_ts()} START physical_gpu={gpu_id} visible_device=0 resume={task.detectron_resume} =====\n")
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
        last: Optional[str] = None
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
    gpu_ids: List[int],
    data_yaml: str,
    runs_dir: Path,
    log_dir: Path,
    state_path: Path,
    stagger_sec: int,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "started_at": now_ts(),
        "total_tasks": len(tasks),
        "gpus_used": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "data_yaml": data_yaml,
        "runs_dir": str(runs_dir),
        "items": {},
    }
    state_lock = threading.Lock()

    for t in tasks:
        state["items"][t.run_key] = {
            "task": t.run_key, "kind": t.kind, "model": t.model,
            "seed": t.seed, "epochs": t.epochs, "family": t.family,
            "priority": t.priority,
            "optimizer": t.optimizer or None,
            "status": "pending",
            "resume_path": t.resume_path,
            "run_dir": t.run_dir,
        }

    def write_state():
        with state_lock:
            state["updated_at"] = now_ts()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_state()

    pending: "queue.Queue[TrainTask]" = queue.Queue()
    for t in tasks:
        pending.put(t)

    results_lock = threading.Lock()
    results: Dict[str, Any] = {}
    start_barrier = threading.Lock()
    next_start_time = [time.time()]

    def acquire_start_slot():
        if stagger_sec <= 0:
            return
        with start_barrier:
            now = time.time()
            wait = max(0.0, next_start_time[0] - now)
            next_start_time[0] = max(now, next_start_time[0]) + stagger_sec
        if wait > 0:
            time.sleep(wait)

    def gpu_worker(gpu_id: int):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                acquire_start_slot()
                item = state["items"][task.run_key]
                item["status"] = "running"
                item["start_at"] = now_ts()
                item["gpu"] = gpu_id
                write_state()

                t0 = time.time()
                if task.kind == "detectron2":
                    code, log = run_detectron2_task(task, data_yaml, runs_dir, log_dir, gpu_id)
                else:
                    code, log = run_ultralytics_task(task, data_yaml, runs_dir, log_dir, gpu_id)
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

                with results_lock:
                    results[task.run_key] = {"code": code, "status": item["status"], "gpu": gpu_id}
                    done = len(results)
                print(f"  [{done}/{len(tasks)}] {task.run_key} gpu={gpu_id}: {item['status']}")
            finally:
                pending.task_done()

    print(f"\n[launch] {len(tasks)} tasks across {len(gpu_ids)} GPU(s): {gpu_ids}\n")

    threads = [threading.Thread(target=gpu_worker, args=(g,), daemon=True) for g in gpu_ids]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

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
        mark = "OK" if status == "ok" else ("FAIL" if status == "failed" else "?")
        print(f"  [{mark}] {key}  gpu={gpu}  {elapsed}s")


def print_dry_run(tasks: List[TrainTask], gpu_ids: List[int]) -> None:
    print(f"\n[dry-run] {len(tasks)} task(s) would be scheduled on GPUs {gpu_ids}\n")
    print(f"  {'Run Key':<22} {'Pri':<10} {'Kind':<11} {'Resume?':<8} Detail")
    print(f"  {'-'*22} {'-'*10} {'-'*11} {'-'*8} {'-'*40}")
    for t in tasks:
        resume = "yes" if (t.resume_path or t.detectron_resume) else "no"
        detail = ""
        if t.kind == "ultralytics":
            detail = f"epochs={t.epochs} run_dir={t.run_dir or t.run_key}"
            if t.resume_path:
                detail += f" resume={Path(t.resume_path).name}"
        else:
            detail = f"epochs={t.epochs} resume={t.detectron_resume}"
        print(f"  {t.run_key:<22} {t.priority:<10} {t.kind:<11} {resume:<8} {detail}")
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "formal"], required=True)
    p.add_argument("--skip-ard", action="store_true")
    p.add_argument("--seeds", default=os.environ.get("SEEDS", "1 2 3"))
    p.add_argument("--data", default=os.environ.get("DATA_YAML", ""))
    p.add_argument("--gpus", default=os.environ.get("TRAIN_GPU_IDS", ""))

    p.add_argument("--target", choices=["primary", "secondary", "all"],
                   default=os.environ.get("TRAIN_TARGET", "all"))
    p.add_argument("--resume-existing", action="store_true",
                   help="Resume from manifest canonical checkpoints")
    p.add_argument("--skip-complete", action="store_true",
                   help="Skip tasks already marked complete in manifest")
    p.add_argument("--no-secondary", action="store_true",
                   help="Block secondary tasks (yolo11/ddw_yolo/yolov10/yolov8)")
    p.add_argument("--manifest", default="",
                   help="Path to recovery manifest JSON; auto-generated if omitted in recovery mode")
    p.add_argument("--start-stagger-sec", type=int,
                   default=int(os.environ.get("START_STAGGER_SEC", "60")),
                   help="Seconds between starting consecutive tasks to spread I/O")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the task schedule and exit without launching")
    p.add_argument("--detectron-target-iter", type=int,
                   default=int(os.environ.get("DETECTRON_TARGET_ITER", "0")),
                   help="Override Detectron2 target max_iter for completeness check")
    args = p.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus else detect_gpus()
    if not gpu_ids:
        print("[FATAL] No GPU detected", file=sys.stderr)
        return 1

    print(f"[GPU] Using physical GPUs: {gpu_ids}")
    if 0 in gpu_ids:
        print("[WARN] GPU0 is in the GPU list; recovery plan reserves it. Continuing as requested.")

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

    suffix = "formal" if args.mode == "formal" else "smoke"
    runs_dir = ROOT / f"runs_{suffix}"
    log_dir = ROOT / "logs" / suffix
    state_path = ROOT / "logs" / f"state_{suffix}.json"

    recovery_mode = args.resume_existing or args.skip_complete or args.target != "all"
    if recovery_mode:
        manifest_path = Path(args.manifest) if args.manifest else None
        if manifest_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            manifest_path = ROOT / "logs" / f"recovery_{ts}" / f"manifest_{args.target}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(ROOT / "scripts" / "recovery_manifest.py"),
                "--runs-dir", str(runs_dir),
                "--detectron-dir", str(ROOT / "runs_detectron2"),
                "--target", args.target,
                "--out", str(manifest_path),
            ]
            if args.detectron_target_iter:
                cmd.extend(["--detectron-target-iter", str(args.detectron_target_iter)])
            print(f"[recovery] generating manifest: {manifest_path}")
            subprocess.run(cmd, check=True)
        else:
            print(f"[recovery] using manifest: {manifest_path}")

        tasks = build_tasks_from_manifest(
            manifest_path=manifest_path,
            target=args.target,
            no_secondary=args.no_secondary,
            skip_complete=args.skip_complete,
            resume_existing=args.resume_existing,
        )
        if not tasks:
            print("[recovery] no tasks to schedule (everything complete or filtered out)")
            return 0
    else:
        tasks = build_tasks_fresh(args.mode, seeds, ddw_model, frcnn_ok)
        if args.no_secondary:
            tasks = [t for t in tasks if t.priority != "secondary"]

    if args.dry_run:
        print_dry_run(tasks, gpu_ids)
        return 0

    state = process_tasks(
        tasks, gpu_ids, data_yaml, runs_dir, log_dir, state_path,
        stagger_sec=args.start_stagger_sec,
    )
    state["_state_path"] = str(state_path)
    print_summary(state)

    failed = state["summary"].get("failed", 0)
    return 2 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
