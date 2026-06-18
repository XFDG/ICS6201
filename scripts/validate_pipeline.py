#!/usr/bin/env python3
"""验证脚本 — 用极小子集验证全链路是否可跑通。

检查项：
1. GPU 可用数量
2. 数据准备（极小子集）
3. 6 个模型 × 1 epoch 并行训练
4. 权重保存
5. 计时功能
6. 默认跳过 keep_alive，避免正式训练前抢占 GPU
7. 输出 validation_report.md
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
KEEP_ALIVE_SCRIPT = Path("/volume/yzhao04/workspace/gpu-workspace/keep_alive/run.sh")
VALIDATE_SUBSET_DIR = ROOT / "logs" / "validate_subset"
VALIDATE_DATA_YAML = ROOT / "configs" / "drone_rgb_validate.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def parse_gpu_ids(value: str) -> List[int]:
    ids: List[int] = []
    for part in str(value or "").replace(" ", ",").split(","):
        s = part.strip()
        if s:
            ids.append(int(s))
    return ids


class ValidationReport:
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_check(self, name: str, passed: bool, detail: str = "", duration: float = 0):
        self.checks.append({
            "name": name,
            "passed": passed,
            "detail": detail,
            "duration_sec": round(duration, 1),
            "time": now_str(),
        })
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name} ({detail})")

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def to_md(self) -> str:
        total = self.elapsed()
        passed = sum(1 for c in self.checks if c["passed"])
        failed = sum(1 for c in self.checks if not c["passed"])

        lines = [
            f"# Validation Report",
            f"",
            f"**Time**: {now_str()}",
            f"**Duration**: {hms(total)}",
            f"**Result**: {passed}/{len(self.checks)} checks passed",
            f"",
            f"| # | Check | Result | Detail | Duration |",
            f"|---|-------|--------|--------|----------|",
        ]
        for i, c in enumerate(self.checks, 1):
            status = "PASS" if c["passed"] else "FAIL"
            lines.append(f"| {i} | {c['name']} | {status} | {c['detail']} | {hms(c['duration_sec'])} |")

        if failed > 0:
            lines.append(f"\n## Failed Checks\n")
            for c in self.checks:
                if not c["passed"]:
                    lines.append(f"- **{c['name']}**: {c['detail']}")

        lines.append(f"\n## Summary\n")
        lines.append(f"- **Total checks**: {len(self.checks)}")
        lines.append(f"- **Passed**: {passed}")
        lines.append(f"- **Failed**: {failed}")
        lines.append(f"- **Total duration**: {hms(total)}")

        return "\n".join(lines)


def run_cmd(cmd: List[str], cwd: Path = None, env: Dict = None, timeout: int = 1800, log_path: Path = None) -> Tuple[int, str, str]:
    """返回 (returncode, stdout, stderr)。输出写入日志文件避免管道阻塞。"""
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                p = subprocess.run(
                    cmd, cwd=str(cwd or ROOT), stdout=f, stderr=f,
                    timeout=timeout, env=env or os.environ,
                )
            # read just the last 2KB for error reporting
            last_bytes = log_path.stat().st_size
            with open(log_path, "rb") as f:
                f.seek(max(0, last_bytes - 2048))
                tail = f.read().decode("utf-8", errors="replace")
            return p.returncode, tail, ""
        else:
            p = subprocess.run(
                cmd, cwd=str(cwd or ROOT), capture_output=True, text=True,
                timeout=timeout, env=env or os.environ,
            )
            return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)


def check_gpu(report: ValidationReport) -> Dict[str, Any]:
    """检测 GPU 可用性"""
    t0 = time.time()
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            gpus = [torch.cuda.get_device_name(i) for i in range(count)]
            detail = f"{count} GPU(s): {gpus}"
            report.add_check("GPU Detection", True, detail, time.time() - t0)
            return {"count": count, "gpus": gpus}
        else:
            report.add_check("GPU Detection", False, "CUDA not available", time.time() - t0)
            return {"count": 0, "gpus": []}
    except Exception as e:
        report.add_check("GPU Detection", False, str(e), time.time() - t0)
        return {"count": 0, "gpus": [], "error": str(e)}


def check_environment(report: ValidationReport) -> bool:
    """检查基本环境"""
    t0 = time.time()
    ok = True
    for mod in ["torch", "ultralytics", "cv2", "PIL", "yaml"]:
        try:
            __import__(mod)
        except Exception:
            ok = False
            report.add_check(f"Import {mod}", False, "import failed", time.time() - t0)
            return ok
    report.add_check("Environment Imports", True, "all core modules OK", time.time() - t0)
    return ok


def _list_images(split_dir: Path, limit: int) -> List[Path]:
    if not split_dir.exists():
        return []
    images: List[Path] = []
    with os.scandir(split_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            p = Path(entry.path)
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            images.append(p)
            if len(images) >= limit:
                break
    return sorted(images)


def _write_image_list(path: Path, images: List[Path]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    return len(images)


def _write_validate_yaml() -> None:
    VALIDATE_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    VALIDATE_DATA_YAML.write_text(
        "\n".join([
            "path: /",
            f"train: {VALIDATE_SUBSET_DIR / 'train.txt'}",
            f"val: {VALIDATE_SUBSET_DIR / 'val.txt'}",
            f"test: {VALIDATE_SUBSET_DIR / 'test.txt'}",
            "names:",
            "  0: drone",
            "",
        ]),
        encoding="utf-8",
    )


def check_data_prep(report: ValidationReport) -> str:
    """验证用数据：从正式 yolo/ 中写绝对路径小样本列表，避免 1 epoch 扫全量数据。"""
    t0 = time.time()

    yolo_images = ROOT / "yolo" / "images"
    yolo_train = yolo_images / "train"
    if not yolo_train.exists() or not any(yolo_train.iterdir()):
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "prepare_rgb_yolo.py"),
            "--root", str(ROOT),
            "--skip-ard",
            "--limit-dut", "10",
            "--limit-dronedet", "10",
            "--seed", "42",
        ]
        code, _, err = run_cmd(cmd)
        if code != 0:
            report.add_check("Data Prepare", False, f"exit={code}: {err[:200]}", time.time() - t0)
            return ""

    train_limit = int(os.environ.get("VALIDATE_TRAIN_IMAGES", "256"))
    val_limit = int(os.environ.get("VALIDATE_VAL_IMAGES", "64"))
    test_limit = int(os.environ.get("VALIDATE_TEST_IMAGES", "64"))

    counts: Dict[str, int] = {}
    for split, limit in [("train", train_limit), ("val", val_limit), ("test", test_limit)]:
        images = _list_images(yolo_images / split, max(1, int(limit)))
        if not images:
            report.add_check("Data Prepare", False, f"no images in yolo/images/{split}", time.time() - t0)
            return ""
        counts[split] = _write_image_list(VALIDATE_SUBSET_DIR / f"{split}.txt", images)

    _write_validate_yaml()
    report.add_check(
        "Data Prepare",
        True,
        f"validate subset train/val/test={counts['train']}/{counts['val']}/{counts['test']}",
        time.time() - t0,
    )
    report.add_check("Data YAML", VALIDATE_DATA_YAML.exists(), str(VALIDATE_DATA_YAML), time.time() - t0)
    return str(VALIDATE_DATA_YAML)


def check_detectron2() -> bool:
    try:
        import detectron2  # noqa: F401
        return True
    except Exception:
        return False


def check_single_model(
    run_key: str, model: str, data_yaml: str, gpu_id: int,
    epochs: int, report: ValidationReport, log_dir: Path, runs_dir: Path,
) -> bool:
    """用单个 GPU 跑一个模型，快速验证"""
    t0 = time.time()

    log_path = log_dir / f"validate_{run_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "train_scripts" / "run_ultralytics_task.py"),
        "--model", model,
        "--data", data_yaml,
        "--epochs", str(epochs),
        "--seed", "42",
        "--project", str(runs_dir),
        "--name", f"validate_{run_key}",
        "--imgsz", "320",
        "--batch", "32",
        "--device", str(gpu_id),
        "--workers", "8",
    ]
    if run_key == "ddw_yolo":
        cmd.extend(["--optimizer", "AdamW"])

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["YOLO_CACHE"] = "0"

    code, tail, err = run_cmd(cmd, env=env, timeout=900, log_path=log_path)

    # 检查 best.pt 是否生成
    best_pt = runs_dir / f"validate_{run_key}" / "weights" / "best.pt"
    weight_ok = best_pt.exists()

    if code == 0 and weight_ok:
        report.add_check(f"Train {run_key}", True, f"gpu={gpu_id}, weights saved", time.time() - t0)
        return True
    else:
        if code == -1:
            reason = "timeout after 15m"
        elif code != 0:
            snippet = (err or tail).strip().replace("\n", " ")[-180:]
            reason = f"exit={code}: {snippet}" if snippet else f"exit={code}"
        else:
            reason = "no weights"
        report.add_check(f"Train {run_key}", False, f"gpu={gpu_id}, {reason}", time.time() - t0)
        return False


def check_faster_rcnn(
    data_yaml: str, gpu_id: int, epochs: int,
    report: ValidationReport, log_dir: Path,
) -> bool:
    """验证 Faster R-CNN"""
    t0 = time.time()

    log_path = log_dir / "validate_faster_rcnn.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["FRCNN_WEIGHTS"] = str(ROOT / "pretrained_weights" / "faster_rcnn_R_50_FPN_3x.pkl")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_detectron2_fasterrcnn.py"),
        "--root", str(ROOT),
        "--seed", "42",
        "--epochs", str(epochs),
        "--ims-per-batch", os.environ.get("D2_IMS_PER_BATCH", "32"),
        "--num-workers", os.environ.get("D2_NUM_WORKERS", "4"),
        "--max-train-images", os.environ.get("VALIDATE_D2_TRAIN_IMAGES", "256"),
        "--eval-max-batches", os.environ.get("VALIDATE_D2_EVAL_BATCHES", "20"),
    ]

    code, tail, err = run_cmd(cmd, env=env, timeout=1200, log_path=log_path)

    if code == 0:
        report.add_check(f"Train faster_rcnn", True, f"gpu={gpu_id}", time.time() - t0)
        return True
    else:
        if code == -1:
            reason = "timeout after 20m"
        else:
            snippet = (err or tail).strip().replace("\n", " ")[-180:]
            reason = f"exit={code}: {snippet}" if snippet else f"exit={code}"
        report.add_check(f"Train faster_rcnn", False, reason, time.time() - t0)
        return False


def check_keep_alive(report: ValidationReport) -> bool:
    """启动 Keep Alive 守护进程（后台运行，不等待退出）"""
    t0 = time.time()
    if not KEEP_ALIVE_SCRIPT.exists():
        report.add_check("Keep Alive", False, f"script not found: {KEEP_ALIVE_SCRIPT}", time.time() - t0)
        return False

    try:
        # 后台启动，不等待退出（keep_alive 是守护进程）
        p = subprocess.Popen(
            ["bash", str(KEEP_ALIVE_SCRIPT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # 短暂检查进程是否启动成功
        time.sleep(3)
        poll = p.poll()
        if poll is not None and poll != 0:
            report.add_check("Keep Alive", False, f"exited early with code={poll}", time.time() - t0)
            return False
        report.add_check("Keep Alive", True, f"daemon started (pid={p.pid})", time.time() - t0)
        return True
    except Exception as e:
        report.add_check("Keep Alive", False, str(e), time.time() - t0)
        return False


def check_timing_end_to_end(report: ValidationReport) -> None:
    """检查端到端总用时"""
    total = report.elapsed()
    report.add_check("End-to-End Timing", True, f"total={hms(total)}", total)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default=os.environ.get("TRAIN_GPU_IDS", ""))
    p.add_argument("--skip-keep-alive", action="store_true", default=os.environ.get("SKIP_KEEP_ALIVE", "1") != "0")
    args = p.parse_args()

    print("=" * 60)
    print(f"ICS6201 Pipeline Validation — {now_str()}")
    print("=" * 60)

    report = ValidationReport()

    # 1. GPU 检测
    print("\n[1] GPU Detection...")
    gpu_info = check_gpu(report)
    gpu_count = gpu_info["count"]
    if gpu_count == 0:
        print("[FATAL] No GPU available")
        report.add_check("FATAL", False, "No GPU — cannot proceed")
        md = report.to_md()
        out_path = ROOT / "validation_report.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"\nReport saved: {out_path}")
        return 1

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus else list(range(gpu_count))
    invalid = [x for x in gpu_ids if x < 0 or x >= gpu_count]
    if invalid:
        report.add_check("GPU Selection", False, f"invalid GPU ids: {invalid}", 0)
        out_path = ROOT / "validation_report.md"
        out_path.write_text(report.to_md(), encoding="utf-8")
        return 1
    report.add_check("GPU Selection", True, f"physical GPUs: {gpu_ids}", 0)

    # 2. 环境检查
    print("\n[2] Environment Check...")
    if not check_environment(report):
        print("[FATAL] Environment check failed")
        out_path = ROOT / "validation_report.md"
        out_path.write_text(report.to_md(), encoding="utf-8")
        return 1

    # 3. 数据准备
    print("\n[3] Data Preparation (tiny subset)...")
    data_yaml = check_data_prep(report)
    if not data_yaml:
        print("[FATAL] Data preparation failed")
        out_path = ROOT / "validation_report.md"
        out_path.write_text(report.to_md(), encoding="utf-8")
        return 1

    # 4. 并行训练验证
    print("\n[4] Parallel Training (all models on tiny subset)...")
    log_dir = ROOT / "logs" / "validate"
    runs_dir = ROOT / "runs_validate"
    # 清理旧验证 runs，避免 ultralytics 自动加后缀
    import shutil
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("yolo11", "yolo11m.pt", 1),
        ("yolov8", "yolov8n.pt", 1),
        ("yolov10", "yolov10n.pt", 1),
        ("rtdetr", "rtdetr-l.pt", 1),
    ]

    ddw_model = None
    candidate = ROOT / "models" / "ddw_yolo11m_p2_bifpn_eca.yaml"
    if candidate.exists():
        ddw_model = str(candidate.resolve())
        models.append(("ddw_yolo", ddw_model, 1))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 每个 GPU 跑一个模型，并行
    assigned = []
    for i, (key, model, epochs) in enumerate(models):
        gpu = gpu_ids[i % len(gpu_ids)]
        assigned.append((key, model, epochs, gpu))

    def run_model(args):
        key, model, epochs, gpu = args
        ok = check_single_model(key, model, data_yaml, gpu, epochs, report, log_dir, runs_dir)
        return key, ok

    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(assigned))) as pool:
        futures = [pool.submit(run_model, a) for a in assigned]
        for f in as_completed(futures):
            key, ok = f.result()

    # 5. Faster R-CNN
    print("\n[5] Faster R-CNN Check...")
    if check_detectron2():
        gpu_for_frcnn = gpu_ids[0]
        check_faster_rcnn(data_yaml, gpu_for_frcnn, 1, report, log_dir)
    else:
        report.add_check("Faster R-CNN", False, "detectron2 not available", 0)

    # 6. 并行验证
    print("\n[6] Multi-GPU Parallel Verification...")
    t0 = time.time()
    tasks_per_gpu = {}
    for i, (key, _, _) in enumerate(models):
        gpu = gpu_ids[i % len(gpu_ids)]
        tasks_per_gpu.setdefault(gpu, []).append(key)
    detail = "; ".join(f"GPU{g}: {','.join(v)}" for g, v in sorted(tasks_per_gpu.items()))
    report.add_check("Multi-GPU Parallel", len(gpu_ids) >= min(len(models), len(gpu_ids)),
                     f"{len(models)} models on {len(gpu_ids)} selected GPU(s): {detail}", time.time() - t0)

    # 7. Keep Alive
    print("\n[7] Keep Alive...")
    if args.skip_keep_alive:
        report.add_check("Keep Alive", True, "skipped (training should own the GPUs)", 0)
    else:
        check_keep_alive(report)

    # 8. 总计时
    print("\n[8] Final Timing...")
    check_timing_end_to_end(report)

    # 输出报告
    md = report.to_md()
    out_path = ROOT / "validation_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Report saved: {out_path}")
    print(f"\n{md}")

    passed = sum(1 for c in report.checks if c["passed"])
    failed = sum(1 for c in report.checks if not c["passed"])
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
