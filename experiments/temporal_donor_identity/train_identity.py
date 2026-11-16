"""Train first-stage linear residual using original codebook; choose on day13.

The frozen base has seen days03–12; they are training, not rolling tests.
No fitting or tuning uses days14–16. See TRAINING.md for full protocol.
"""
import argparse, hashlib, json, os, pickle, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluation.identity import identity_retrieval
from .model import ModelConfig, TemporalDonorIdentity
from .objectives import episodic_identity_loss

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'PT_PROJECT_ML'
EXPECTED='a1c4bfa15ed0c227aefd981f59b87769aece146efb232da90617e4f561b10211'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('experiments.fingerprint_'):
            module='experiments.multiday.'+module.removeprefix('experiments.fingerprint_')
        if module.startswith('experiments.multiday_gain'):
            module=module.replace('experiments.multiday_gain','experiments.multiday.gain')
        return super().find_class(module,name)


def prepare(out, archive):
    import experiments
    experiments.__path__.insert(0,str(archive/'experiments'));sys.path.insert(0,str(archive))
    os.environ['PT_ML_ROOT']=str(DATA)
    checkpoint=DATA/'fingerprint_codebook_recent/final/model.pkl'
    if sha(checkpoint)!=EXPECTED:raise ValueError('unexpected original codebook checkpoint')
    with checkpoint.open('rb') as f:base=LegacyUnpickler(f).load()
    hosts=json.loads((DATA/'fingerprint_aligned/hosts.json').read_text())
    idx=pd.Index(hosts,name='host_id')
    source=base.source.loc[idx].to_numpy(dtype=np.float32)
    original=pd.read_csv(DATA/'late_test/vectors/multiday_codebook/embeddings.csv').set_index('host_id').loc[idx]
    delta=float(np.abs(source-original.to_numpy()).max())
    if delta>1e-7:raise ValueError('source mismatch')
    del original
    hist=DATA/'fingerprint_codebook_recent/history';aligned=DATA/'fingerprint_aligned'
    episodes=[];fingerprints={}
    for day in range(3,13):
        p=(aligned if day<=9 else hist)/f'historical{day:02}.npy'
        mask_path=(aligned if day<=9 else hist)/f'future_active{day:02}.npy'
        raw=np.load(p,mmap_mode='r');active=np.load(mask_path)
        if raw.shape!=source.shape or active.shape!=(len(idx),):raise ValueError(f'day{day}: shape mismatch')
        rows=np.flatnonzero(active)
        if len(rows)<100:raise ValueError(f'day{day}: too few active hosts')
        late=base.adapter.transform(raw[rows]).astype(np.float32)
        episodes.append({'day':day,'rows':rows,'late':late})
        fingerprints[str(p.relative_to(ROOT))]=sha(p);fingerprints[str(mask_path.relative_to(ROOT))]=sha(mask_path)
        print('prepared historical day',day,'active',len(rows),flush=True)
    devfile=DATA/'fingerprint_codebook_recent/candidates/codebook1_e400/later.csv'
    late=pd.read_csv(devfile,dtype={'host_id':str}).set_index('host_id')
    eval_hosts=pd.read_csv(DATA/'fingerprint_recent/reference_validation/validation/eval_hosts.csv',dtype=str).host_id.tolist()
    if set(late.index)!=set(eval_hosts):raise ValueError('day13 active host mismatch')
    late=late.loc[eval_hosts];source_dev=base.source.loc[eval_hosts]
    meta={'base_checkpoint_sha256':EXPECTED,'source_csv_max_abs_error':delta,'training_days':list(range(3,13)),
          'development_day':13,'development_in_training':False,'test_days':[14,15,16],
          'training_windows':'prepared 12:30–13:30 historical windows; day02 source only',
          'base_is_fitted_on_training_days':'in-sample base; historical episodes are not temporal holdouts',
          'features':'old adapter output only; context zero; no query-ID input',
          'input_sha256':fingerprints,'development_late_sha256':sha(devfile),
          'source_coordinates':'original codebook 896, unchanged'}
    (out/'inputs.json').write_text(json.dumps(meta,indent=2))
    return source,idx,episodes,source_dev,late


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--archive',type=Path,default=Path('/private/tmp/pt-original-multiday'))
    ap.add_argument('--steps',type=int,default=24);ap.add_argument('--batch',type=int,default=512)
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--lr',type=float,default=0.00003)
    ap.add_argument('--no-bias',action='store_true')
    ap.add_argument('--contact-context',type=Path,help='prefix-fitted contact context directory')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<1 or args.batch<20:raise ValueError('steps>=1, batch>=20 required')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    source,hosts,episodes,source_dev,late_dev=prepare(args.output,args.archive)
    context_dev = None
    if args.contact_context:
        context_manifest=json.loads((args.contact_context/'manifest.json').read_text())
        if context_manifest['train_days']!=list(range(3,13)) or context_manifest['development_day']!=13 or not context_manifest['train_only_fit']:
            raise ValueError('contact features must be fitted on days03–12 only')
        for ep in episodes:
            path=args.contact_context/f"day{ep['day']:02}.npy"
            values=np.load(path,mmap_mode='r')
            if values.shape!=(len(hosts),64):raise ValueError('bad historical context shape')
            ep['context']=np.asarray(values[ep['rows']],dtype=np.float32)
        dev_values=np.load(args.contact_context/'day13.npy',mmap_mode='r')
        if dev_values.shape!=(len(hosts),64):raise ValueError('bad development context shape')
        dev_positions=hosts.get_indexer(late_dev.index)
        if (dev_positions<0).any():raise ValueError('unknown validation host')
        context_dev=torch.from_numpy(np.asarray(dev_values[dev_positions],dtype=np.float32))
    config={'seed':args.seed,'steps':args.steps,'batch':args.batch,'learning_rate':args.lr,
      'temperature':.1,'residual_penalty':10.,'weight_decay':.0001,'alignment_gradient':'stop','identity_bias':not args.no_bias,
      'feature_family':'contacts_tfidf_svd' if args.contact_context else 'baseline_vector_only',
      'contact_manifest_sha256':sha(args.contact_context/'manifest.json') if args.contact_context else None,
      'svd_regularization':0.,'checkpoint_steps':[0]+list(range(8,args.steps+1,8))+([] if args.steps%8==0 else [args.steps]),
      'selection':'highest official identity_top1 day13; exact tie uses baseline/earlier checkpoint',
      'note':'batch sub-gallery and stop-gradient are approximations; selection uses full day13 gallery',
      'code_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'experiments/temporal_donor_identity/model.py',ROOT/'experiments/temporal_donor_identity/objectives.py']}}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    model=TemporalDonorIdentity(ModelConfig(identity_bias=not args.no_bias));opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    xdev=torch.from_numpy(late_dev.to_numpy(dtype=np.float32));zeros_dev=context_dev if context_dev is not None else torch.zeros(len(xdev),64)
    source_dev=source_dev.astype(float)
    rows=[];best=-1.;best_step=None
    def assess(step):
        nonlocal best,best_step
        with torch.no_grad():out=model.late_vectors(xdev,zeros_dev).numpy()
        df=pd.DataFrame(out,index=late_dev.index,columns=source_dev.columns)
        score=identity_retrieval(source_dev,df,alignment='procrustes',seed=0,n_boot=0)
        row={'step':step,'identity_top1':score['top1'],'n_eval':score['n_eval_hosts']}
        rows.append(row);print('development',row,flush=True)
        torch.save({'state_dict':model.state_dict(),'step':step,'model_config':vars(model.config),'experiment':config},args.output/f'checkpoint_{step:04}.pt')
        if score['top1']>best+1e-12:
            best=score['top1'];best_step=step
            torch.save({'state_dict':model.state_dict(),'step':step,'model_config':vars(model.config),'experiment':config},args.output/'model.pt')
        (args.output/'development.json').write_text(json.dumps(rows,indent=2))
    assess(0)
    for step in range(1,args.steps+1):
        ep=episodes[(step-1)%len(episodes)]
        chosen=rng.choice(len(ep['rows']),size=min(args.batch,len(ep['rows'])),replace=False)
        order=rng.permutation(len(chosen));half=len(chosen)//2
        src=torch.from_numpy(source[ep['rows'][chosen]])
        late=torch.from_numpy(ep['late'][chosen]);context=torch.from_numpy(ep['context'][chosen]) if args.contact_context else torch.zeros(len(late),64)
        y=model.late_vectors(late,context)
        loss,_=episodic_identity_loss(src,y,torch.from_numpy(order[half:]),torch.from_numpy(order[:half]),
                                      temperature=.1,alignment_gradient='stop',baseline=late,residual_weight=10.)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        print('step',step,'day',ep['day'],'loss',round(float(loss.detach()),6),flush=True)
        if step%8==0 or step==args.steps:assess(step)
    bestmodel=torch.load(args.output/'model.pt',weights_only=True);model.load_state_dict(bestmodel['state_dict'])
    with torch.no_grad():chosen=model.late_vectors(xdev,zeros_dev).numpy()
    pd.DataFrame(chosen,index=late_dev.index,columns=source_dev.columns).rename_axis('host_id').to_csv(args.output/'validation.csv')
    result={'baseline_day13_identity':rows[0]['identity_top1'],'selected_day13_identity':best,
            'selected_step':best_step,'gain':best-rows[0]['identity_top1'],'test_read':False,
            'training_completed':True,'day13_is_development_and_already_studied':True,
            'checkpoint_sha256':sha(args.output/'model.pt'),
            'status':'no_improvement' if best_step==0 else 'development_improvement_requires_test'}
    (args.output/'frozen_selection.json').write_text(json.dumps(result,indent=2));print('frozen',result,flush=True)

if __name__=='__main__':main()
