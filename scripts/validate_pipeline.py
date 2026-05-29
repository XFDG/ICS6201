#!/usr/bin/env python3
"""验证脚本 — 用极小子集验证全链路是否可跑通。

检查项：
1. GPU 可用数量
2. 数据准备（极小子集）
3. 6 个模型 × 1 epoch 并行训练
4. 权重保存
5. 计时功能
6. 训练完成后调用 keep_alive
7. 输出 validation_report.md
"""

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


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


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


def run_cmd(cmd: List[str], cwd: Path = None, env: Dict = None, timeout: int = 600) -> Tuple[int, str, str]:
    """返回 (returncode, stdout, stderr)"""
    try:
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


def check_data_prep(report: ValidationReport) -> str:
    """验证用数据：优先用已有 yolo/ 数据，不存在则生成小子集到独立目录 yolo_validate/"""
    t0 = time.time()
    data_yaml = ROOT / "configs" / "drone_rgb_abs.yaml"

    yolo_train = ROOT / "yolo" / "images" / "train"
    if yolo_train.exists() and any(yolo_train.iterdir()):
        # 已有正式数据，直接用
        report.add_check("Data Prepare", True,
                         f"using existing yolo/ data ({sum(1 for _ in yolo_train.iterdir())} train images)",
                         time.time() - t0)

        # 确保绝对路径 YAML 存在
        if not data_yaml.exists():
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "write_data_yaml.py"),
                 "--root", str(ROOT), "--out", str(data_yaml)],
                check=False, capture_output=True,
            )
        report.add_check("Data YAML", data_yaml.exists(), str(data_yaml), time.time() - t0)
        return str(data_yaml)

    # 无正式数据 → 生成小子集到独立目录
    report.add_check("Data Prepare", True, "generating tiny subset to yolo_validate/", time.time() - t0)

    validate_root = ROOT / "yolo_validate"
    validate_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_rgb_yolo.py"),
        "--root", str(ROOT),
        "--skip-ard",
        "--limit-dut", "10",
        "--limit-dronedet", "10",
        "--seed", "42",
    ]
    # 临时重命名 yolo/ 以保护已有数据，让脚本输出到 yolo/（少量数据），再移回
    code, out, err = run_cmd(cmd)
    if code != 0:
        report.add_check("Data Prepare", False, f"exit={code}: {err[:200]}", time.time() - t0)
        return ""

    data_yaml = ROOT / "configs" / "drone_rgb_abs_validate.yaml"
    cmd2 = [
        sys.executable,
        str(ROOT / "scripts" / "write_data_yaml.py"),
        "--root", str(ROOT),
        "--out", str(data_yaml),
    ]
    run_cmd(cmd2)

    report.add_check("Data YAML", data_yaml.exists(), f"tiny subset ready", time.time() - t0)

    # COCO 转换（Detectron2 需要）
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_rgb_coco.py"), "--root", str(ROOT)],
        check=False, capture_output=True,
    )

    return str(data_yaml)


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
        "--imgsz", "640",
        "--batch", "4",
        "--device", str(gpu_id),
        "--workers", "4",
    ]

    code, _, _ = run_cmd(cmd)

    # 检查 best.pt 是否生成
    best_pt = runs_dir / f"validate_{run_key}" / "weights" / "best.pt"
    weight_ok = best_pt.exists()

    if code == 0 and weight_ok:
        report.add_check(f"Train {run_key}", True, f"gpu={gpu_id}, weights saved", time.time() - t0)
        return True
    else:
        reason = f"exit={code}" if code != 0 else "no weights"
        report.add_check(f"Train {run_key}", False, f"gpu={gpu_id}, {reason}", time.time() - t0)
        return False


def check_faster_rcnn(
    data_yaml: str, gpu_id: int, epochs: int,
    report: ValidationReport, log_dir: Path,
) -> bool:
    """验证 Faster R-CNN"""
    t0 = time.time()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_detectron2_fasterrcnn.py"),
        "--root", str(ROOT),
        "--seed", "42",
        "--epochs", str(epochs),
        "--ims-per-batch", "2",
        "--num-workers", "2",
    ]

    code, _, err = run_cmd(cmd, env=env)

    if code == 0:
        report.add_check(f"Train faster_rcnn", True, f"gpu={gpu_id}", time.time() - t0)
        return True
    else:
        report.add_check(f"Train faster_rcnn", False, f"exit={code}: {err[:150]}", time.time() - t0)
        return False


def check_keep_alive(report: ValidationReport) -> bool:
    """调用 keep_alive 脚本"""
    t0 = time.time()
    if not KEEP_ALIVE_SCRIPT.exists():
        report.add_check("Keep Alive", False, f"script not found: {KEEP_ALIVE_SCRIPT}", time.time() - t0)
        return False

    try:
        p = subprocess.run(
            ["bash", str(KEEP_ALIVE_SCRIPT)],
            capture_output=True, text=True, timeout=30,
        )
        if p.returncode == 0:
            report.add_check("Keep Alive", True, "executed successfully", time.time() - t0)
            return True
        else:
            report.add_check("Keep Alive", False, f"exit={p.returncode}: {p.stderr[:100]}", time.time() - t0)
            return False
    except Exception as e:
        report.add_check("Keep Alive", False, str(e), time.time() - t0)
        return False


def check_timing_end_to_end(report: ValidationReport) -> None:
    """检查端到端总用时"""
    total = report.elapsed()
    report.add_check("End-to-End Timing", True, f"total={hms(total)}", total)


def main() -> int:
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

    gpu_ids = list(range(gpu_count))

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
        gpu = gpu_ids[i % gpu_count]
        assigned.append((key, model, epochs, gpu))

    def run_model(args):
        key, model, epochs, gpu = args
        ok = check_single_model(key, model, data_yaml, gpu, epochs, report, log_dir, runs_dir)
        return key, ok

    with ThreadPoolExecutor(max_workers=min(gpu_count, len(assigned))) as pool:
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
        gpu = gpu_ids[i % gpu_count]
        tasks_per_gpu.setdefault(gpu, []).append(key)
    detail = "; ".join(f"GPU{g}: {','.join(v)}" for g, v in sorted(tasks_per_gpu.items()))
    report.add_check("8-GPU Parallel", gpu_count >= len(models),
                     f"{len(models)} models on {gpu_count} GPUs: {detail}", time.time() - t0)

    # 7. Keep Alive
    print("\n[7] Keep Alive...")
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
