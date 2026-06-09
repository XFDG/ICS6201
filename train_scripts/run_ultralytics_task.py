import argparse
import csv
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_cache(s: str):
    v = str(s).strip().lower()
    if v in ("", "0", "false", "no", "off", "none"):
        return None
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("ram", "disk"):
        return v
    return None


def _maybe_register_ddw_modules(model_path: str) -> None:
    p = Path(model_path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        return
    if not p.exists():
        return
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    if "ECA" not in text and "BiFPNFuse" not in text:
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import ddw_modules  # type: ignore

        from ultralytics.nn import modules as ul_modules
        from ultralytics.nn import tasks as ul_tasks

        ul_tasks.ECA = ddw_modules.ECA
        ul_tasks.BiFPNFuse = ddw_modules.BiFPNFuse
        ul_modules.ECA = ddw_modules.ECA
        ul_modules.BiFPNFuse = ddw_modules.BiFPNFuse
    except Exception:
        return


def _patch_ultralytics_cache_lock() -> None:
    """Serialize Ultralytics label cache writes across parallel training jobs."""
    try:
        from ultralytics.data import dataset as ds_utils
        from ultralytics.data import utils as data_utils
    except Exception:
        return

    original = data_utils.save_dataset_cache_file

    def locked_save_dataset_cache_file(prefix, path, x, version):  # type: ignore[no-untyped-def]
        lock_path = Path(str(path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return original(prefix, path, x, version)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    data_utils.save_dataset_cache_file = locked_save_dataset_cache_file
    ds_utils.save_dataset_cache_file = locked_save_dataset_cache_file


def _read_args_yaml(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "args.yaml"
    if not p.exists():
        return {}
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_results_csv_last_row(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "results.csv"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            last: Optional[Dict[str, str]] = None
            for row in reader:
                last = row
        if not last:
            return {}
        out: Dict[str, Any] = {}
        for k, v in last.items():
            if v is None:
                continue
            s = str(v).strip()
            if s == "":
                continue
            try:
                out[k] = float(s)
                continue
            except Exception:
                pass
            try:
                out[k] = int(s)
                continue
            except Exception:
                pass
            out[k] = s
        return out
    except Exception:
        return {}


def _model_param_count(yolo) -> Optional[int]:
    try:
        m = getattr(yolo, "model", None)
        if m is None:
            return None
        params = list(m.parameters())
        return int(sum(p.numel() for p in params))
    except Exception:
        return None


def _env_info(device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import ultralytics

        out["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    except Exception:
        pass
    try:
        import torch

        out["torch_version"] = getattr(torch, "__version__", None)
        out["cuda_is_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                out["cuda_device_count"] = int(torch.cuda.device_count())
                out["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        out["device_arg"] = device
    except Exception:
        out["device_arg"] = device
    return out


def _results_dict_and_speed(res) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    d = getattr(res, "results_dict", None) or {}
    s = getattr(res, "speed", None) or {}
    if not isinstance(d, dict):
        d = {}
    if not isinstance(s, dict):
        s = {}
    return d, s


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default=os.environ.get("YOLO_CACHE", ""))
    p.add_argument("--amp", default=os.environ.get("YOLO_AMP", "1"))
    p.add_argument("--optimizer", default=os.environ.get("OPTIMIZER", ""))
    p.add_argument("--resume-path", default="",
                   help="Path to last.pt to resume from; loads weights and passes resume=True")
    p.add_argument("--exist-ok", action="store_true",
                   help="Allow writing into an existing run directory (project/name)")
    args = p.parse_args()

    from ultralytics import YOLO

    _patch_ultralytics_cache_lock()
    _maybe_register_ddw_modules(args.model)

    resume_path = str(args.resume_path or "").strip()
    model_path = resume_path if resume_path and Path(resume_path).exists() else args.model
    model = YOLO(model_path)
    cache = _parse_cache(args.cache)
    amp = _parse_bool(args.amp)
    train_kwargs: Dict[str, Any] = {
        "data": args.data,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "seed": int(args.seed),
        "device": args.device,
        "workers": int(args.workers),
        "amp": amp,
        "project": args.project,
        "name": args.name,
        "plots": False,
        "exist_ok": bool(args.exist_ok),
    }
    if resume_path and Path(resume_path).exists():
        train_kwargs["resume"] = True
    if cache is not None:
        train_kwargs["cache"] = cache
    optimizer = str(args.optimizer).strip()
    if optimizer:
        train_kwargs["optimizer"] = optimizer

    r = model.train(**train_kwargs)

    run_dir = Path(args.project).resolve() / args.name
    best = run_dir / "weights" / "best.pt"
    best_model = str(best) if best.exists() else args.model

    val_res = model.val(model=best_model, data=args.data, split="val", device=args.device, workers=int(args.workers), plots=False)
    test_res = model.val(model=best_model, data=args.data, split="test", device=args.device, workers=int(args.workers), plots=False)
    val_dict, val_speed = _results_dict_and_speed(val_res)
    test_dict, test_speed = _results_dict_and_speed(test_res)

    out: Dict[str, Any] = {
        "train_result": str(r),
        "run_dir": str(run_dir),
        "best": str(best) if best.exists() else None,
        "val": val_dict,
        "val_speed": val_speed,
        "test": test_dict,
        "test_speed": test_speed,
        "args": _read_args_yaml(run_dir),
        "results_last_row": _read_results_csv_last_row(run_dir),
        "env": _env_info(args.device),
        "model_param_count": _model_param_count(model),
        "train_kwargs": train_kwargs,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
