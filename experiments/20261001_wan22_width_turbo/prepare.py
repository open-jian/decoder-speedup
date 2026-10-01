"""Freeze a seeded 10k train / 256 validation split from completed VidGen shards."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import time
import cv2
import av
import yaml
from decoder_speedup.data import vidgen_source_id
from decoder_speedup.provenance import sha256, write_json

base = Path('/data2/jian/vae-speedup')
experiment = base / 'experiments/20261001_wan22_width_turbo'
root = base / 'datasets/vidgen-1m'
output = experiment / 'manifests'
output.mkdir(parents=True, exist_ok=True)
assert not (output / 'train.jsonl').exists(), 'Never replace a frozen training manifest'
cv2.setNumThreads(1)
status = json.loads((root / 'status.json').read_text())
write_json(status, experiment / 'download-snapshot.json')
shards = sorted(p for p in (root / 'videos').iterdir() if (p / '.vidgen_complete.json').is_file())
train_pool, val_pool = [], []
for shard in shards:
    metadata = root / 'metadata' / (shard.name + '.files.jsonl')
    for line in metadata.read_text().splitlines():
        item = json.loads(line)
        row = {'path': item['relative_path'], 'source_id': vidgen_source_id(item['relative_path'])}
        fraction = int(hashlib.sha256(f"42:{row['source_id']}".encode()).hexdigest()[:16], 16) / 2**64
        (val_pool if fraction < .05 else train_pool).append(row)
print(json.dumps({'completed_shards':len(shards), 'train_pool':len(train_pool), 'val_pool':len(val_pool)}), flush=True)
random.Random(42).shuffle(train_pool)
random.Random(43).shuffle(val_pool)

def check(row):
    try:
        with av.open(str(root / row['path'])) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            header_count = stream.frames
            count = sum(1 for _ in container.decode(video=0))
    except Exception as exc:
        return f'decode_error:{type(exc).__name__}:{exc}'
    row['decoded_frames'] = count
    row['header_frames'] = header_count
    cap = cv2.VideoCapture(str(root / row['path']))
    try:
        if not cap.isOpened():
            return 'cannot_open'
        if count < 17:
            return f'too_short:{count}'
        for index in (0, count // 2, count - 17):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            for offset in range(17 if index == count - 17 else 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    return f'unreadable_frame:{index+offset}'
        return None
    finally:
        cap.release()

invalid = []
selected = {}
with ThreadPoolExecutor(max_workers=12) as pool:
    for name, rows, wanted in [('train', train_pool, 10000), ('val', val_pool, 256)]:
        selected[name] = []
        for start in range(0, len(rows), 256):
            batch = rows[start:start + min(256, wanted-len(selected[name]))]
            for row, error in zip(batch, pool.map(check, batch)):
                if error:
                    invalid.append(dict(row, split=name, reason=error))
                else:
                    selected[name].append(row)
            print(json.dumps({'split':name,'valid':len(selected[name]),'invalid':len(invalid)}), flush=True)
            if len(selected[name]) == wanted:
                break
        assert len(selected[name]) == wanted
assert not {r['source_id'] for r in selected['train']} & {r['source_id'] for r in selected['val']}
for name, rows in selected.items():
    (output / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(output / 'invalid.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in invalid))
write_json({'seed':42,'validation_sample_seed':43, 'source_split_fraction':.05,
    'available_completed_shards':[p.name for p in shards], 'dataset_revision':status['revision'],
    'train_pool':len(train_pool),'val_pool':len(val_pool),
    'train_clips':10000,'val_clips':256,'invalid_count':len(invalid),
    'checks':'count actual frames with full PyAV decode; minimum 17 frames; OpenCV first/middle/last 17-frame checks; no quality filtering',
    'header_count_mismatches':sum(r['header_frames'] != r['decoded_frames'] for rows in selected.values() for r in rows),
    'train_sha256':sha256(output/'train.jsonl'),'val_sha256':sha256(output/'val.jsonl')}, output/'split.json')
cfg=yaml.safe_load((base/'decoder-speedup/configs/wan22/width.yaml').read_text())
cfg['model']['source']=str(base/'Wan2.2')
cfg['model']['weights']=str(base/'assets/wan22/Wan2.2_VAE.pth')
cfg['data'].update(root=str(root),train_manifest=str(output/'train.jsonl'),val_manifest=str(output/'val.jsonl'))
cfg['output']=str(base/'decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001')
cfg['wandb']['name']='wan22-amd-width-turbo-10k-20261001'
cfg['wandb']['tags'] += ['subset:train10k-val256','campaign:first-recovery']
(experiment/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
print(json.dumps({'prepared':True,'output':cfg['output'],'train':10000,'validation':256,'invalid':len(invalid)}),flush=True)
