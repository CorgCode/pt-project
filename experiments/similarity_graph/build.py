"""Construct graph matcher and exact/compact vectors from frozen learned matrices."""
import argparse
import json
from pathlib import Path
import joblib
import shutil
import numpy as np
from .prepare import config_file
from .data import Window,read_window,signature
from .matching import GraphMatcher
from .matrix import exact_vectors,compact_vectors,psd_edge_vectors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reuse-initial-built',type=Path)
    parser.add_argument('--psd-edge-control',action='store_true')
    parser.add_argument('--skip-full-rank',action='store_true')
    parser.add_argument('--compact-steps',type=int)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    training=json.loads((args.trained/'frozen_training.json').read_text())
    config=training['config'];hosts=training['hosts']
    compact_steps=args.compact_steps if args.compact_steps is not None else config.get('compact_steps',80)
    reusable=None
    if args.reuse_initial_built:
        reusable=json.loads((args.reuse_initial_built/'frozen_build.json').read_text())
        old_training=json.loads((Path(reusable['training'])/'frozen_training.json').read_text())
        if (old_training['train']!=training['train'] or old_training['training_files']!=training['training_files'] or
            old_training['matrix_sha256']['initial']!=training['matrix_sha256']['initial'] or reusable['hosts']!=hosts):
            raise ValueError('reused initial/matcher must have identical training history and initial matrix')
    observations=[read_window(Window(**w),hosts,config['cache']) for w in config['train']]
    print('fitting graph profiles',flush=True)
    matcher=GraphMatcher().fit(observations,hosts)
    joblib.dump(matcher,args.output/'matcher.joblib',compress=3)
    reports={}
    for name in ('initial','learned'):
        if not args.skip_full_rank and (args.psd_edge_control or config.get('psd_edge_control',False)):
            print('exact PSD-edge control',name,flush=True)
            z,info=psd_edge_vectors(np.load(args.trained/f'{name}.npy'))
            label=f'{name}_psd_edge';path=args.output/f'{label}.npy';np.save(path,z);del z
            reports[label]={'matrix':name,'dim':len(hosts),'factorization':info,'sha256':signature(path)}
        if name=='initial' and reusable is not None:
            for label,metadata in reusable['candidates'].items():
                if metadata['matrix']!='initial' or label in reports:continue
                if args.skip_full_rank and metadata['dim']>=len(hosts):continue
                original=args.reuse_initial_built/f'{label}.npy'
                if signature(original)!=metadata['sha256']:raise ValueError('reused initial vectors changed')
                shutil.copyfile(original,args.output/original.name);reports[label]=dict(metadata)
                reports[label].setdefault('compression_loss',reusable.get('compact_loss','soft'))
            continue
        if signature(args.trained/f'{name}.npy')!=training['matrix_sha256'][name]:raise ValueError('training matrix changed')
        S=np.load(args.trained/f'{name}.npy')
        if not args.skip_full_rank:
            print('exact factorization',name,flush=True)
            z,info=exact_vectors(S)
            path=args.output/f'{name}_exact.npy';np.save(path,z);del z
            reports[f'{name}_exact']={'matrix':name,'dim':len(hosts),'factorization':info,'sha256':signature(path)}
        for dim in config.get('compact_dims',[384,512,1024]):
            print('compact distillation',name,dim,flush=True)
            loss_kind=config.get('initial_compact_loss',config.get('compact_loss','soft')) if name=='initial' else config.get('compact_loss','soft')
            actual_steps=config.get('initial_compact_steps',compact_steps) if name=='initial' else compact_steps
            z,log=compact_vectors(S,dim,steps=actual_steps,loss_kind=loss_kind)
            if loss_kind=='top10':
                log=[{'step':r['step'],'total_distillation_loss':r['neighbor_kl']} for r in log]
            label=f'{name}_{dim}';path=args.output/f'{label}.npy';np.save(path,z)
            reports[label]={'matrix':name,'dim':z.shape[1],'compression_history':log,'sha256':signature(path),
                            'compression_loss':loss_kind,'compression_steps':actual_steps}
    record={'training':str(args.trained),'hosts':hosts,'training_end':training['training_end'],
            'candidates':reports,'matcher_sha256':signature(args.output/'matcher.joblib'),
            'development_or_test_read':False,'psd_edge_control':args.psd_edge_control or config.get('psd_edge_control',False),
            'full_rank_skipped':args.skip_full_rank,'compact_loss':config.get('compact_loss','soft'),'compact_steps':compact_steps}
    (args.output/'frozen_build.json').write_text(json.dumps(record,indent=2)+'\n')
    print('vectors and graph matcher ready',flush=True)


if __name__=='__main__':main()
