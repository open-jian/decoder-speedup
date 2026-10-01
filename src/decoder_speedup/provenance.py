from pathlib import Path
import hashlib
import json
import os
import subprocess
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_identity(source):
    source = Path(source).resolve()
    def git(*args):
        proc = subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    directory = source / "wan/modules"
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "files": {p.name: sha256(p) for p in sorted(directory.glob("*.py"))
                      if p.name == "vae2_2.py" or p.name.startswith("winograd")}}


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def framework_identity():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}
