import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def require_detectron2() -> None:
    try:
        import detectron2  # noqa: F401
    except Exception as e:
        raise RuntimeError("detectron2 not available") from e


def read_num_images(coco_json: Path) -> int:
    obj = json.loads(coco_json.read_text(encoding="utf-8"))
    return int(len(obj.get("images", [])))


def compute_max_iter(num_images: int, ims_per_batch: int, epochs: int) -> int:
    ims_per_batch = max(1, int(ims_per_batch))
    iters_per_epoch = max(1, (num_images + ims_per_batch - 1) // ims_per_batch)
    return int(max(1, epochs * iters_per_epoch))


class _LimitedLoader:
    def __init__(self, loader, max_batches: int):
        self._loader = loader
        self._max_batches = int(max(0, max_batches))

    def __iter__(self):
        if self._max_batches <= 0:
            yield from iter(self._loader)
            return
        yield from itertools.islice(iter(self._loader), self._max_batches)

    def __len__(self) -> int:
        if self._max_batches <= 0:
            try:
                return len(self._loader)
            except Exception:
                return 0
        try:
            return min(len(self._loader), self._max_batches)
        except Exception:
            return self._max_batches


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--ims-per-batch", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-max-batches", type=int, default=0)
    args = p.parse_args()

    require_detectron2()

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.utils.env import seed_all_rng

    root = Path(args.root).resolve()
    yolo_root = root / "yolo"
    coco_ann = root / "coco" / "annotations"

    train_json = coco_ann / "instances_train.json"
    val_json = coco_ann / "instances_val.json"
    test_json = coco_ann / "instances_test.json"

    train_imgs = yolo_root / "images" / "train"
    val_imgs = yolo_root / "images" / "val"
    test_imgs = yolo_root / "images" / "test"

    register_coco_instances("drone_train", {}, str(train_json), str(train_imgs))
    register_coco_instances("drone_val", {}, str(val_json), str(val_imgs))
    register_coco_instances("drone_test", {}, str(test_json), str(test_imgs))

    num_train = read_num_images(train_json)
    max_iter = compute_max_iter(num_train, args.ims_per_batch, args.epochs)

    out_dir = root / "runs_detectron2" / f"faster_rcnn_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("drone_train",)
    cfg.DATASETS.TEST = ("drone_val",)
    cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    weights_path = os.environ.get(
        "FRCNN_WEIGHTS",
        str(root / "pretrained_weights" / "faster_rcnn_R_50_FPN_3x.pkl"),
    )
    if Path(weights_path).exists():
        cfg.MODEL.WEIGHTS = str(Path(weights_path).resolve())
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = int(args.ims_per_batch)
    cfg.SOLVER.MAX_ITER = int(max_iter)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(max(1, max_iter // 5))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = str(out_dir)
    cfg.SEED = int(args.seed)

    seed_all_rng(int(args.seed))

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator_val = COCOEvaluator("drone_val", cfg, False, output_dir=str(out_dir / "eval_val"))
    val_loader = build_detection_test_loader(cfg, "drone_val")
    val_loader = _LimitedLoader(val_loader, args.eval_max_batches)
    val_metrics = inference_on_dataset(trainer.model, val_loader, evaluator_val)

    evaluator_test = COCOEvaluator("drone_test", cfg, False, output_dir=str(out_dir / "eval_test"))
    test_loader = build_detection_test_loader(cfg, "drone_test")
    test_loader = _LimitedLoader(test_loader, args.eval_max_batches)
    test_metrics = inference_on_dataset(trainer.model, test_loader, evaluator_test)

    env: Dict[str, Any] = {}
    try:
        import detectron2

        env["detectron2_version"] = getattr(detectron2, "__version__", None)
    except Exception:
        pass
    try:
        import torch

        env["torch_version"] = getattr(torch, "__version__", None)
        env["cuda_is_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                env["cuda_device_count"] = int(torch.cuda.device_count())
                env["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        pass

    out: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "ims_per_batch": int(args.ims_per_batch),
        "num_workers": int(args.num_workers),
        "max_iter": int(max_iter),
        "eval_max_batches": int(args.eval_max_batches),
        "val_metrics": val_metrics if isinstance(val_metrics, dict) else {},
        "test_metrics": test_metrics if isinstance(test_metrics, dict) else {},
        "env": env,
    }
    print(json.dumps(out, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
