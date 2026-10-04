# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Resumable deterministic acquisition using the qualified Gemma adapter."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import numpy as np
from adapter import make_plant, retained
from fp32_entry import set_precision
from state import Store, acknowledge, atomic_json, sha
from analysis import commit, evaluate

ROOT=Path(__file__).resolve().parent


def load_inputs(root=ROOT):
    config=json.loads((root/'config.json').read_text())
    with np.load(root/'data/designs.npz',allow_pickle=False) as z:
        designs={k:z[k] for k in z.files}
    records=retained.load_prompt_records(root/'data/prompts.jsonl')
    groups={s:retained.select_records(records,source) for s,source in [('direction','direction'),('fit','fit'),('test','test_id'),('collateral','collateral_id')]}
    if [len(groups[s]) for s in groups]!=[32,32,96,32]: raise ValueError('Unexpected prompt counts')
    return config,designs,groups


def jobs(config, designs, groups):
    result=[{'id':'directions','kind':'directions','phase':'fit'}]
    def add(prefix,kind,split,count,actions=None):
        for i in range(count):
            for start in range(0,len(groups[split]),8):
                result.append({'id':f'{prefix}-{i:03d}-{start:03d}','kind':kind,'split':split,
                               'start':start,'index':i,'pool':actions,'phase':'fit' if split=='fit' else 'heldout'})
    add('fit-clean','clean','fit',1)
    add('fit-coordinate','margin','fit',64,'coordinate')
    for seed in config['design_seeds']: add(f'fit-aggregate_{seed}','margin','fit',64,f'aggregate_{seed}')
    add('fit-validation','margin','fit',8,'validation')
    add('fit-search','margin','fit',64,'search')
    result.append({'id':'commitment','kind':'commitment','phase':'fit'})
    for split in ['test','collateral']:
        add(split+'-clean','clean',split,1)
        add(split+'-menu','margin',split,64,'menu')
    add('kl-clean','kl_clean','collateral',1)
    add('kl-menu','kl','collateral',64,'menu')
    return result


def action_for(job,config,designs):
    pool=job['pool']
    if pool=='coordinate': pool='coordinate_'+str(config['design_seeds'][0])
    if pool=='search': pool='menu'
    return designs[pool][job['index']]*config['primary_scale']


def matrix(store,prefix,actions,prompts,key='margin'):
    return np.stack([np.concatenate([store.get(f'{prefix}-{i:03d}-{j:03d}')[0][key]
                                     for j in range(0,prompts,8)]) for i in range(actions)])


def make_commitment(store,config,designs):
    clean=matrix(store,'fit-clean',1,32)[0]
    fit={pool:matrix(store,'fit-'+pool,n,32)-clean[None,:]
         for pool,n in [('coordinate',64),*[(f'aggregate_{seed}',64) for seed in config['design_seeds']],('validation',8),('search',64)]}
    return commit(config,designs,fit)


def results(store,groups):
    document=json.loads(str(store.get('commitment')[0]['document']))
    target=matrix(store,'test-menu',64,96)-matrix(store,'test-clean',1,96)
    benign=matrix(store,'collateral-menu',64,32)-matrix(store,'collateral-clean',1,32)
    kl=matrix(store,'kl-menu',64,32,'kl')
    target=np.vstack([np.zeros((1,96)),target]); benign=np.vstack([np.zeros((1,32)),benign]);kl=np.vstack([np.zeros((1,32)),kl])
    return evaluate(document,target,benign,kl,[r.family for r in groups['test']],[r.family for r in groups['collateral']])


def distribution(plant,rows,bundle=None,action=None):
    torch=plant.torch
    seq=[plant._prefix_ids(r.text) for r in rows]
    ids,mask=plant._padded_batch(seq)
    plant._positions=plant._last_prompt_positions(ids.shape[1],[0]*len(rows))
    context=plant._edit_hooks(bundle,action) if bundle is not None else nullcontext()
    with torch.inference_mode(),context:
        logits=plant.model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[:,-1,:]
        return torch.log_softmax(logits.float(),dim=-1).cpu().numpy()


def divergence(clean,edited):
    # Accumulation is float64; probabilities originate in the declared FP32 log-softmax.
    clean=np.asarray(clean,dtype=np.float64);edited=np.asarray(edited,dtype=np.float64)
    values=np.sum(np.exp(clean)*(clean-edited),axis=1)
    if not np.isfinite(values).all() or (values < -1e-6).any(): raise ValueError('Invalid KL')
    return np.maximum(values,0.)


def qualify_kl(plant,config):
    from preflight import FIXTURES
    rows=[retained.PromptRecord(f'kl-engineering-{i}',t,'benign',f'kl-engineering-{i}','engineering') for i,t in enumerate(FIXTURES[:2])]
    captures={l:[] for l in plant.layers}
    seq=[plant._prefix_ids(r.text) for r in rows];ids,mask=plant._padded_batch(seq)
    plant._positions=plant._last_prompt_positions(ids.shape[1],[0,0])
    with plant.torch.inference_mode(),plant._capture_hooks(captures):
        plant.model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
    norms=np.array([np.median(np.linalg.norm(np.concatenate(captures[l]),axis=1)) for l in plant.layers])
    random=np.random.default_rng(10399).normal(size=(len(plant.layers),config['runtime']['expected_hidden_size']))
    vectors=random/np.linalg.norm(random,axis=1)[:,None]*(config['base_fraction']*norms[:,None])
    bundle=retained.DirectionBundle(plant.layers,vectors,norms,config['base_fraction'],plant.model_id,plant.revision)
    clean=distribution(plant,rows)
    zero=distribution(plant,rows,bundle,np.zeros(len(plant.layers)))
    action=np.ones(len(plant.layers))*.5/np.sqrt(len(plant.layers))
    edited=distribution(plant,rows,bundle,action)
    single=np.concatenate([distribution(plant,[r],bundle,action) for r in rows])
    restored=distribution(plant,rows)
    k=divergence(clean,edited);ks=divergence(clean,single)
    checks={'zero':bool(np.allclose(clean,zero,atol=1e-6,rtol=0)),
            'restored':bool(np.allclose(clean,restored,atol=1e-6,rtol=0)),
            'batch_kl':bool(np.allclose(k,ks,atol=1e-5,rtol=1e-3)),
            'normalized':bool(np.allclose(np.exp(clean.astype(float)).sum(axis=1),1,atol=1e-6,rtol=0)),
            'hooks_removed':all(not b._forward_hooks for b in plant.blocks),
            'finite':bool(np.isfinite(k).all())}
    return {'checks':checks,'passed':all(checks.values()),'kl':k.tolist(),'batch_kl':ks.tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);p.add_argument('--lease',required=True)
    args=p.parse_args(); deadline=args.deadline
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for name,digest in freeze['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed')
    binding=sha(ROOT/'freeze.json');store=Store(ROOT/'checkpoints',binding)
    config,designs,groups=load_inputs(); plan=jobs(config,designs,groups)
    ids=[j['id'] for j in plan]
    existing=[e['job'] for e in store.index['entries']]
    if existing!=ids[:len(existing)]: raise ValueError('Saved jobs are not a plan prefix')
    if len(existing)==len(plan):
        atomic_json(ROOT/'worker_terminal.json',{'status':'complete','completed':len(plan),'total':len(plan)});return
    import torch
    precision=set_precision(torch)
    plant=make_plant(config)
    if {str(p.dtype) for p in plant.model.parameters() if p.is_floating_point()}!={'torch.float32'}:raise ValueError('FP32 required')
    if 'A100' not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(0).total_memory < 75*1024**3:raise ValueError('A100 80 GB required')
    qualification=qualify_kl(plant,config)
    atomic_json(ROOT/'kl_qualification.json',dict(qualification,precision=precision))
    if not qualification['passed']:raise ValueError('KL engineering qualification failed')
    atomic_json(ROOT/'ready.json',{'unix':time.time(),'completed_at_start':len(existing),'total':len(plan)})
    audit=[{'id':r.prompt_id,'full_tokens':len(plant._chat_prefix_ids(r.text)),
            'used_tokens':len(plant._prefix_ids(r.text)),'truncated':len(plant._chat_prefix_ids(r.text))>len(plant._prefix_ids(r.text))}
           for rows in groups.values() for r in rows]
    atomic_json(ROOT/'token_audit.json',audit)
    bundle=None
    counts={'calls':0,'sequences':0,'input_tokens':0}
    def counter(_m,_a,kw):
        counts['calls']+=1;counts['sequences']+=int(kw['input_ids'].shape[0]);counts['input_tokens']+=int(kw['attention_mask'].sum().item())
    handle=plant.model.register_forward_pre_hook(counter,with_kwargs=True)
    try:
        for ordinal,job in enumerate(plan):
            if store.has(job['id']):continue
            if time.time()+180>deadline:
                atomic_json(ROOT/'worker_terminal.json',{'status':'lease_boundary','completed':len(store.entries),'total':len(plan)});return
            if job['phase']=='heldout' and not store.has('commitment'):raise ValueError('Held-out collection before commitment')
            if bundle is None and store.has('directions'):
                a,_=store.get('directions')
                bundle=retained.DirectionBundle(tuple(config['layers']),a['directions'],a['norms'],config['base_fraction'],config['model_id'],config['revision']);bundle.validate()
            atomic_json(ROOT/'progress.json',{'job':job['id'],'ordinal':ordinal,'total':len(plan),'unix':time.time()})
            start=time.monotonic(); before=dict(counts)
            if job['kind']=='directions':
                bundle=plant.capture_directions(groups['direction'],config['base_fraction'])
                arrays={'directions':bundle.directions,'norms':bundle.residual_norms}
            elif job['kind']=='commitment':
                arrays={'document':np.asarray(json.dumps(make_commitment(store,config,designs),sort_keys=True,allow_nan=False))}
            else:
                rows=groups[job['split']][job['start']:job['start']+8]
                if job['kind']=='clean': arrays={'margin':plant.refusal_margin(rows)}
                elif job['kind']=='margin': arrays={'margin':plant.refusal_margin(rows,bundle,action_for(job,config,designs))}
                elif job['kind']=='kl_clean':arrays={'log_probs':distribution(plant,rows)}
                elif job['kind']=='kl':
                    clean,_=store.get(f"kl-clean-000-{job['start']:03d}")
                    arrays={'kl':divergence(clean['log_probs'],distribution(plant,rows,bundle,action_for(job,config,designs)))}
                else:raise ValueError('Unknown job')
            torch.cuda.synchronize()
            store.put(job['id'],arrays,{'specification':job,'lease':args.lease,'seconds':time.monotonic()-start,
                     'counts':{k:counts[k]-before[k] for k in counts},'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                     'direction_sha256':None if job['id']=='directions' else store.entries['directions']['sha256']})
            acknowledge(store,job['id'],deadline)
        atomic_json(ROOT/'worker_terminal.json',{'status':'complete','completed':len(plan),'total':len(plan)})
    finally:handle.remove()


if __name__=='__main__':main()
