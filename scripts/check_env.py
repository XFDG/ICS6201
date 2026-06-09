import shutil
import subprocess
import sys
from typing import List, Tuple


def check_import(name: str) -> Tuple[bool, str]:
    try:
        __import__(name)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ok = True
    lines: List[str] = []

    lines.append(f"python: {sys.version.split()[0]}")

    for mod in ["ultralytics", "torch", "PIL", "yaml"]:
        exists, err = check_import(mod)
        if exists:
            lines.append(f"{mod}: OK")
        else:
            ok = False
            lines.append(f"{mod}: MISSING ({err})")

    try:
        import torch

        lines.append(f"torch.version: {getattr(torch, '__version__', 'unknown')}")
        lines.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                lines.append(f"cuda.device_count: {torch.cuda.device_count()}")
                lines.append(f"cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        p = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=False)
        first = (p.stdout or p.stderr or "").splitlines()[:1]
        lines.append(f"ffmpeg: OK ({first[0] if first else 'unknown'})")
    else:
        lines.append("ffmpeg: MISSING (only required when not using --skip-ard)")

    for line in lines:
        print(line)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
