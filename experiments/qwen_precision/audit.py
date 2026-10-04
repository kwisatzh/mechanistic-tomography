# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Observe the unchanged original Qwen hook; no scientific endpoint scoring."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from qwen_refusal import QwenRefusalPlant, load_directions

ROOT = Path(__file__).resolve().parent


def save(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def summarize(before, after, intended):
    delta = after.astype(np.float64) - before.astype(np.float64)
    intended = np.broadcast_to(intended, delta.shape)
    norm = np.linalg.norm(intended, axis=-1)
    actual = np.linalg.norm(delta, axis=-1)
    active = norm > 0
    if not all(np.isfinite(x).all() for x in (before, after, intended)):
        raise ValueError('Nonfinite edit array')
    if np.any(delta[~active] != 0):
        raise ValueError('Inactive edit changed residual')
    error = np.linalg.norm(delta - intended, axis=-1)[active] / norm[active]
    cosine = np.sum(delta * intended, axis=-1)[active] / np.maximum(actual[active]*norm[active], 1e-300)
    return {'count': int(active.sum()), 'relative_error': error.tolist(),
            'norm_ratio': (actual[active]/norm[active]).tolist(), 'cosine': cosine.tolist()}


def observe(plant, bundle, action, ids, mask):
    """Observers bracket the actual original hook, including in-place precision."""
    torch = plant.torch
    before_full, before, after = {}, {}, {}
    handles, off_position, dtypes = [], [], []
    rows = torch.arange(ids.shape[0], device=ids.device)
    initial_counts = [len(block._forward_hooks) for block in plant.blocks]
    try:
        for layer in plant.layers:
            def remember(_m, _a, output, layer=layer):
                h = output[0] if isinstance(output, tuple) else output
                before_full[layer] = h.detach().clone()
                before[layer] = h[rows, plant._positions].detach().float().cpu().numpy()
                dtypes.append(str(h.dtype))
            handles.append(plant.blocks[layer].register_forward_hook(remember))
        with plant._edit_hooks(bundle, action):
            for layer in plant.layers:
                def inspect(_m, _a, output, layer=layer):
                    h = output[0] if isinstance(output, tuple) else output
                    old = before_full.pop(layer)
                    difference = h != old
                    difference[rows, plant._positions] = False
                    off_position.append(int(difference.sum().item()))
                    after[layer] = h[rows, plant._positions].detach().float().cpu().numpy()
                handles.append(plant.blocks[layer].register_forward_hook(inspect))
            with torch.inference_mode():
                plant.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    if initial_counts != [len(block._forward_hooks) for block in plant.blocks]:
        raise ValueError('Leaked hook')
    if any(off_position) or len(off_position) != len(plant.layers):
        raise ValueError('Edit position or coverage differs')
    if set(before) != set(plant.layers) or set(after) != set(plant.layers):
        raise ValueError('Incomplete layer coverage')
    return np.stack([before[k] for k in plant.layers]), np.stack([after[k] for k in plant.layers]), dtypes


def main():
    import torch
    freeze = json.loads((ROOT/'freeze.json').read_text())
    for name, expected in freeze['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen file differs')
    config = json.loads((ROOT/'config.json').read_text())
    fixtures = json.loads((ROOT/'fixtures.json').read_text())
    lease = json.loads((ROOT/'lease.json').read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    plant = QwenRefusalPlant(config['model']['id'], config['model']['revision'],
        config['actuator']['layers'], **config['task'], **config['runtime'],
        position=config['actuator']['position'])
    save('ready.json', {'unix':time.time(), 'versions':{k:importlib.metadata.version(k) for k in
         ['torch','transformers','numpy','accelerate']}, 'cuda':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(), 'parameter_dtypes':sorted({str(p.dtype) for p in plant.model.parameters()})})
    if time.time() > lease['requested_unix']+1800:
        raise TimeoutError('Setup allowance exceeded')
    bundle = load_directions(ROOT/'directions.npz')
    prefixes = [plant._prefix_ids(t) for t in fixtures['texts']]
    stems = [plant.refusal_stems[0], plant.compliance_stems[0]]
    seq, lengths = [], []
    for prefix in prefixes:
        for stem in stems:
            tail = plant._stem_token_ids[stem]
            seq.append(prefix+list(tail)); lengths.append(len(tail))
    ids, mask = plant._padded_batch(seq)
    plant._positions = plant._last_prompt_positions(ids.shape[1], lengths)
    if max(map(len,seq)) > plant.max_length:
        raise ValueError('Context bound exceeded')
    actions = json.loads((ROOT/'actions.json').read_text())
    index, records = [], []
    for i, row in enumerate(actions):
        if time.time() > lease['worker_deadline']:
            raise TimeoutError('Worker allowance exhausted')
        action = np.asarray(row['action'])
        before, after, dtypes = observe(plant,bundle,action,ids,mask)
        if set(dtypes) != {'torch.bfloat16'}:
            raise ValueError('Unexpected activation precision')
        intended = bundle.directions[:,None,:]*action[:,None,None]
        stats = summarize(before,after,intended)
        name = f'checkpoints/action-{i:03d}.npz'
        (ROOT/'checkpoints').mkdir(exist_ok=True)
        np.savez_compressed(ROOT/name,before=before,after=after,intended=intended,action=action,
            layers=np.asarray(plant.layers),positions=plant._positions.cpu().numpy())
        raw=(ROOT/name).read_bytes()
        index.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        record=dict(row,**stats,activation_dtypes=dtypes,off_position_changes=0)
        records.append(record)
        save('index.json',{'entries':index})
        save('progress.json',{'completed':len(index),'total':len(actions),'unix':time.time()})
    errors=[v for r in records for v in r['relative_error']]
    report={'scope':'prospective engineering fixtures only; no historical endpoint rerun',
        'passed':max(errors)<=0.20,'tolerance':0.20,'max_relative_error':max(errors),
        'median_relative_error':float(np.median(errors)), 'records':records,
        'sequence_count':len(seq),'prefix_lengths':list(map(len,prefixes)),
        'all_position_checks_pass':True,'no_leaked_hooks':True,'finished_unix':time.time()}
    save('report.json',report)
    print(json.dumps({k:report[k] for k in ['passed','max_relative_error','median_relative_error']}))


if __name__ == '__main__':
    main()
