# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Small immutable checkpoint store composed from pathlib, JSON, NumPy, and hashes."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


class Store:
    def __init__(self, root, binding):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.binding = binding
        self.index_path = self.root / 'index.json'
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {'binding': binding, 'entries': []}
        if self.index['binding'] != binding:
            raise ValueError('Checkpoint freeze differs')
        self.entries = {e['job']: e for e in self.index['entries']}
        if len(self.entries) != len(self.index['entries']):
            raise ValueError('Duplicate checkpoint identity')
        for e in self.entries.values():
            if e['file'] != e['job'] + '.npz' or '/' in e['file']:
                raise ValueError('Unsafe checkpoint path')
            if sha(self.root/e['file']) != e['sha256']:
                raise ValueError('Checkpoint integrity failure')

    def has(self, job):
        return job in self.entries

    def get(self, job):
        e = self.entries[job]
        if sha(self.root/e['file']) != e['sha256']:
            raise ValueError('Checkpoint changed')
        with np.load(self.root/e['file'], allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            if metadata['binding'] != self.binding or metadata['job'] != job:
                raise ValueError('Checkpoint metadata differs')
            return {k: z[k].copy() for k in z.files if k != 'metadata'}, metadata

    def put(self, job, arrays, metadata):
        if self.has(job) or '/' in job or '..' in job:
            raise ValueError('Checkpoint is immutable or identity invalid')
        for value in arrays.values():
            a = np.asarray(value)
            if a.dtype.kind in 'fc' and not np.isfinite(a).all():
                raise ValueError('Nonfinite checkpoint')
        meta = dict(metadata, job=job, binding=self.binding, saved_unix=time.time())
        path = self.root/(job+'.npz')
        if path.exists():
            raise ValueError('Unindexed checkpoint requires reconciliation')
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, metadata=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)), **arrays)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(path)
        entry = {'job': job, 'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
        self.index['entries'].append(entry); self.entries[job] = entry
        atomic_json(self.index_path, self.index)
        return entry


def acknowledge(store, job, deadline):
    entry = store.entries[job]
    stop = min(deadline, time.time()+300)
    while time.time() < stop:
        p = store.root/'ack.json'
        if p.exists():
            ack = json.loads(p.read_text())
            if ack.get('job') == job and ack.get('sha256') == entry['sha256']:
                return
        time.sleep(1)
    raise TimeoutError('Host has not retained the last checkpoint')
