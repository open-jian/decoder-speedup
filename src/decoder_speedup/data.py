"""Frozen video manifests, source-disjoint splits and resumable, deterministic batches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import cv2
import numpy as np
import torch
from .provenance import sha256, write_json


def vidgen_source_id(path):
    # HD-VILA/VidGen names use an 11-character YouTube id followed by Scene/split.
    stem = Path(path).stem
    match = re.match(r"^([A-Za-z0-9_-]{11})(?:[-_]Scene[-_]?\d+|[-_]\d+)(?:.*)$", stem, re.I)
    if match is None:
        raise ValueError(f"Cannot infer source id from {path}; provide a JSONL manifest with explicit source_id")
    return match.group(1)


def split_manifest(rows, validation_fraction=0.05, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0,1)")
    train, val, paths = [], [], set()
    for row in sorted(rows, key=lambda r: r["path"]):
        if not row.get("source_id") or row["path"] in paths:
            raise ValueError("Rows require a source_id and unique path")
        paths.add(row["path"])
        value = int(hashlib.sha256(f"{seed}:{row['source_id']}".encode()).hexdigest()[:16], 16) / 2**64
        (val if value < validation_fraction else train).append(row)
    return train, val


def make_manifests(root, output, *, input_manifest=None, limit=None, seed=42, validation_fraction=0.05):
    root, output = Path(root).resolve(), Path(output)
    if limit is not None and limit < 2:
        raise ValueError("limit must be at least 2")
    if input_manifest:
        rows = [json.loads(line) for line in Path(input_manifest).read_text().splitlines() if line.strip()]
    else:
        scan_root = root / "videos" if (root / "videos").is_dir() else root
        paths = sorted(p for p in scan_root.rglob("*.mp4")
                       if not any(part.startswith(".") or part in {"extraction_in_progress", "archives_in_progress"}
                                  for part in p.relative_to(root).parts))
        rows = [{"path": str(p.relative_to(root)), "source_id": vidgen_source_id(p)} for p in paths]
    # Sample a fixed subset without a quality filter. Split by source AFTER sampling.
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    train, val = split_manifest(rows, validation_fraction, seed)
    if not train or not val:
        raise ValueError("Empty train/validation split; use more sources or another split seed")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("train", train), ("val", val)):
        path = output / (name + ".jsonl")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen manifest: {path}")
    for name, values in (("train", train), ("val", val)):
        (output / (name + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
    write_json({"seed": seed, "validation_fraction": validation_fraction, "root": str(root),
                "train_clips": len(train), "val_clips": len(val),
                "train_sha256": sha256(output / "train.jsonl"), "val_sha256": sha256(output / "val.jsonl")}, output / "split.json")
    return train, val


class VideoDataset:
    def __init__(self, config, manifest, training=True):
        self.config = config
        self.root = Path(config.root).resolve()
        self.manifest = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.training = training
        if not self.rows:
            raise ValueError("Empty dataset manifest")
        seen = set()
        for row in self.rows:
            if not isinstance(row.get("source_id"), str) or not row["source_id"]:
                raise ValueError("Every manifest row requires a nonempty source_id")
            if "decoded_frames" in row and (type(row["decoded_frames"]) is not int or row["decoded_frames"] < 1):
                raise ValueError("decoded_frames must be a positive verified integer")
            path = (self.root / row["path"]).resolve()
            if not path.is_relative_to(self.root) or path in seen:
                raise ValueError(f"Duplicate path or path outside dataset root: {path}")
            seen.add(path)
        self.identity = {"manifest_sha256": sha256(self.manifest), "rows": len(self.rows)}

    def __len__(self):
        return len(self.rows)

    def get(self, index, seed):
        cfg = self.config
        rng = random.Random(seed)
        path = self.root / self.rows[index]["path"]
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            # Some MP4 header counts exceed the actual decodable timeline.
            # Prefer the frozen preflight count.
            total = self.rows[index].get("decoded_frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            needed = 1 + (cfg.frames - 1) * cfg.frame_stride
            if total < needed:
                raise RuntimeError(f"Video too short ({total} < {needed}): {path}")
            start = rng.randrange(total - needed + 1) if self.training else (total - needed) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            # One crop/flip for the whole clip, never different temporal transforms.
            crop = None
            flip = self.training and cfg.horizontal_flip and rng.random() < 0.5
            for i in range(needed):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at frame {start+i}: {path}")
                if i % cfg.frame_stride:
                    continue
                h, w = frame.shape[:2]
                ratio = max(cfg.height / h, cfg.width / w)
                rh, rw = max(cfg.height, round(h * ratio)), max(cfg.width, round(w * ratio))
                if crop is None:
                    y = rng.randrange(rh - cfg.height + 1) if self.training else (rh - cfg.height) // 2
                    x = rng.randrange(rw - cfg.width + 1) if self.training else (rw - cfg.width) // 2
                    crop = (y, x)
                y, x = crop
                frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y:y + cfg.height, x:x + cfg.width]
                if flip:
                    frame = frame[:, ::-1]
                frames.append(frame.copy())
            return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div_(127.5).sub_(1)
        finally:
            cap.release()


def assert_disjoint(train, val):
    if {r['source_id'] for r in train.rows} & {r['source_id'] for r in val.rows}:
        raise ValueError("Train/validation contain overlapping source videos")
    if {r['path'] for r in train.rows} & {r['path'] for r in val.rows}:
        raise ValueError("Train/validation contain overlapping clips")


class BatchStream:
    """No worker prefetch: cursor exactly identifies the next consumed sample.

    Clip augmentations are keyed by epoch/index, independent of global RNG state.
    CPU video loading is intentionally simple for the initial single-GPU trainer.
    """
    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch, self.cursor = 0, 0
        self._order = None

    def _indices(self):
        if self._order is None:
            self._order = list(range(len(self.dataset)))
            random.Random(self.seed + self.epoch).shuffle(self._order)
        return self._order

    def next(self, decode=True):
        clips = []
        for _ in range(self.batch_size):
            if self.cursor == len(self.dataset):
                self.epoch += 1
                self.cursor = 0
                self._order = None
            index = self._indices()[self.cursor]
            token = f"{self.seed}:{self.epoch}:{index}"
            seed = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
            if decode:
                clips.append(self.dataset.get(index, seed))
            self.cursor += 1
        return torch.stack(clips) if decode else None

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed,
                "batch_size": self.batch_size, "dataset": self.dataset.identity}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("seed", "batch_size", "dataset"):
            if state[key] != expected[key]:
                raise ValueError(f"Cannot resume different data stream: {key}")
        if not 0 <= state["cursor"] <= len(self.dataset) or state["epoch"] < 0:
            raise ValueError("Invalid saved data cursor")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self._order = None
