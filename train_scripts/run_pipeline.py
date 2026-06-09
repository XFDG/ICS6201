import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Task:
    key: str
    kind: str
    model: str
    epochs: int


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cmd(cmd: List[str], log_path: Path, cwd: Path) -> Tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {now_ts()} START =====\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=f)
        f.write(f"===== {now_ts()} END code={p.returncode} =====\n")
    return p.returncode, time.time() - t0


def last_json_line(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        last = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    last = s
        if not last:
            return None
        obj = json.loads(last)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return int(default)
    s = str(v).strip()
    if not s:
        return int(default)
    try:
        return int(s)
    except Exception:
        return int(default)


def env_int_optional(name: str) -> Optional[int]:
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s in ("auto", "none", "null"):
        return None
    try:
        return int(s)
    except Exception:
        return None


def choose_ultralytics_batch(task_key: str) -> int:
    global_batch = env_int_optional("BATCH")
    if global_batch is not None:
        return int(global_batch)
    if task_key == "rtdetr":
        return env_int("BATCH_RTDETR", 48)
    if task_key == "ddw_yolo":
        return env_int("BATCH_DDW", env_int("BATCH_YOLO", 64))
    return env_int("BATCH_YOLO", 64)


def ensure_dataset(root: Path, skip_ard: bool) -> None:
    yolo_dir = root / "yolo" / "images" / "train"
    if not yolo_dir.exists():
        cmd = [
            sys.executable,
            str(root / "scripts" / "prepare_rgb_yolo.py"),
            "--root",
            str(root),
            "--clear",
        ]
        if skip_ard:
            cmd.append("--skip-ard")
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

    subprocess.run([sys.executable, str(root / "scripts" / "prepare_rgb_coco.py"), "--root", str(root)], check=False)


def detectron2_available() -> bool:
    try:
        import detectron2  # noqa: F401

        return True
    except Exception:
        return False


def build_tasks(mode: str, models_dir: Path, ddw_model: Optional[str]) -> List[Task]:
    yolo11 = os.environ.get("YOLO11_MODEL", "yolo11m.pt")
    yolo8 = os.environ.get("YOLOV8_MODEL", "yolov8n.pt")
    yolo10 = os.environ.get("YOLOV10_MODEL", "yolov10n.pt")
    rtdetr = os.environ.get("RTDETR_MODEL", "rtdetr-l.pt")

    if mode == "smoke":
        epochs_yolo = int(os.environ.get("EPOCHS_SMOKE", "1"))
        epochs_rtdetr = int(os.environ.get("EPOCHS_SMOKE", "1"))
        epochs_frcnn = int(os.environ.get("EPOCHS_SMOKE", "1"))
    else:
        epochs_yolo = int(os.environ.get("EPOCHS_YOLO", "200"))
        epochs_rtdetr = int(os.environ.get("EPOCHS_RTDETR", "120"))
        epochs_frcnn = int(os.environ.get("EPOCHS_FASTER_RCNN", "120"))

    tasks: List[Task] = [
        Task("yolo11", "ultralytics", yolo11, epochs_yolo),
        Task("yolov8", "ultralytics", yolo8, epochs_yolo),
        Task("yolov10", "ultralytics", yolo10, epochs_yolo),
        Task("rtdetr", "ultralytics", rtdetr, epochs_rtdetr),
    ]

    if ddw_model:
        tasks.append(Task("ddw_yolo", "ultralytics", ddw_model, epochs_yolo))
    else:
        tasks.append(Task("ddw_yolo", "skip", "ddw_yolo.py", epochs_yolo))

    tasks.append(Task("faster_rcnn", "detectron2", "detectron2", epochs_frcnn))
    return tasks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "formal"], required=True)
    p.add_argument("--skip-ard", action="store_true")
    p.add_argument("--seeds", default=os.environ.get("SEEDS", "1 2 3"))
    p.add_argument("--only-failed", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs_dir = root / "logs" / args.mode
    runs_dir = root / ("runs_formal" if args.mode == "formal" else "runs_smoke")
    state_path = root / "logs" / f"state_{args.mode}.json"

    device = env_str("DEVICE", "0")
    imgsz = env_int("IMGSZ", 640)
    workers = env_int("WORKERS", 16)
    d2_ims_per_batch = env_int("D2_IMS_PER_BATCH", 8)
    d2_num_workers = env_int("D2_NUM_WORKERS", 8)

    models_dir = Path(os.environ.get("MODELS_DIR", str(root.parent / "models"))).resolve()
    ddw_model = os.environ.get("DDW_MODEL")
    if not ddw_model:
        candidate = root / "models" / "ddw_yolo11m_p2_bifpn_eca.yaml"
        if candidate.exists():
            ddw_model = str(candidate.resolve())

    seeds = [int(x) for x in args.seeds.split() if x.strip()]

    ensure_dataset(root, skip_ard=args.skip_ard)
    data_yaml = str((root / "configs" / "drone_rgb_abs.yaml").resolve())

    tasks = build_tasks(args.mode, models_dir=models_dir, ddw_model=ddw_model)

    state = read_json(state_path)
    state.setdefault("mode", args.mode)
    state.setdefault("root", str(root))
    state.setdefault("created_at", now_ts())
    state.setdefault("runs_dir", str(runs_dir))
    state.setdefault("data_yaml", data_yaml)
    state.setdefault("items", {})

    for task in tasks:
        for seed in seeds:
            run_key = f"{task.key}_seed{seed}"
            state["items"].setdefault(run_key, {})
            item = state["items"][run_key]

            if args.only_failed and item.get("status") == "ok":
                continue
            if not args.only_failed and item.get("status") in ("ok", "skipped"):
                continue

            item["task"] = task.key
            item["seed"] = seed
            item["kind"] = task.kind
            item["model"] = task.model
            item["epochs"] = task.epochs
            item["device"] = device
            item["imgsz"] = imgsz
            item["start_at"] = now_ts()
            item["status"] = "running"
            item.pop("error", None)
            item.pop("metrics", None)
            write_json(state_path, state)

            log_path = logs_dir / f"{run_key}.log"

            if task.kind == "skip":
                item["status"] = "skipped"
                item["error"] = "DDW_MODEL not provided"
                item["end_at"] = now_ts()
                write_json(state_path, state)
                continue

            if task.kind == "detectron2":
                if not detectron2_available():
                    item["status"] = "skipped"
                    item["error"] = "detectron2 not installed"
                    item["end_at"] = now_ts()
                    write_json(state_path, state)
                    continue

                eval_max_batches = 200 if args.mode == "smoke" else 0
                cmd = [
                    sys.executable,
                    str(root / "scripts" / "train_detectron2_fasterrcnn.py"),
                    "--root",
                    str(root),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(task.epochs),
                    "--ims-per-batch",
                    str(d2_ims_per_batch),
                    "--num-workers",
                    str(d2_num_workers),
                    "--eval-max-batches",
                    str(eval_max_batches),
                ]
                item["d2_ims_per_batch"] = d2_ims_per_batch
                item["d2_num_workers"] = d2_num_workers
                item["d2_eval_max_batches"] = eval_max_batches
                code, dt = run_cmd(cmd, log_path, cwd=root)
                item["elapsed_sec"] = dt
                item["end_at"] = now_ts()
                if code == 0:
                    item["status"] = "ok"
                    item.pop("error", None)
                    m = last_json_line(log_path)
                    if m:
                        item["metrics"] = m
                else:
                    item["status"] = "failed"
                    item["error"] = f"exit_code={code}"
                write_json(state_path, state)
                continue

            batch = choose_ultralytics_batch(task.key)
            item["batch"] = batch
            item["workers"] = workers
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_ultralytics_task.py"),
                "--model",
                task.model,
                "--data",
                data_yaml,
                "--epochs",
                str(task.epochs),
                "--seed",
                str(seed),
                "--project",
                str(runs_dir),
                "--name",
                run_key,
                "--imgsz",
                str(imgsz),
                "--batch",
                str(batch),
                "--device",
                device,
                "--workers",
                str(workers),
            ]
            code, dt = run_cmd(cmd, log_path, cwd=root)
            item["elapsed_sec"] = dt
            item["end_at"] = now_ts()

            if code == 0:
                item["status"] = "ok"
                item.pop("error", None)
                m = last_json_line(log_path)
                if m:
                    item["metrics"] = m
            else:
                item["status"] = "failed"
                item["error"] = f"exit_code={code}"

            write_json(state_path, state)

    for v in state["items"].values():
        if v.get("status") == "ok":
            v.pop("error", None)
        if v.get("status") in ("skipped", "failed"):
            v.pop("metrics", None)

    ok = sum(1 for v in state["items"].values() if v.get("status") == "ok")
    failed = sum(1 for v in state["items"].values() if v.get("status") == "failed")
    skipped = sum(1 for v in state["items"].values() if v.get("status") == "skipped")

    state["summary"] = {"ok": ok, "failed": failed, "skipped": skipped, "updated_at": now_ts()}
    write_json(state_path, state)
    print(str(state_path))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
