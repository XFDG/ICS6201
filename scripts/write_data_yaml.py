import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    yolo_dir = (root / "yolo").resolve()

    def norm(p: Path) -> str:
        return p.as_posix()

    text = "\n".join(
        [
            f"path: {norm(yolo_dir)}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: drone",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

