"""Обучение preprocessing и контрольных моделей только на объявленном train."""
from .config import load as load_config,versions
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from .data import prepare,sha
from .features import BlockFeatures
from .model import pretrain,encode,plain,pooled_source


def code_shas():return {str(p.relative_to(Path(__file__).parent)):sha(p) for p in Path(__file__).parent.rglob('*.py') if not p.name.startswith('test_')}


def verify_training(folder,metadata=None):
    """Не принимать старые преобразованные строки после смены признаков/словаря."""
    folder=Path(folder)
    metadata=metadata or json.loads((folder/'frozen_training.json').read_text())
    if sha(folder/'transforms.joblib')!=metadata['transforms_sha256']:
        raise ValueError('Преобразования/словарь изменились после fit')
    for name,digest in metadata.get('train_feature_sha256',{}).items():
        if sha(folder/(name+'.npy'))!=digest:raise ValueError('Старый или изменённый кэш признаков: '+name)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-transforms',type=Path)
    args=parser.parse_args();c=load_config(args.config);code_at_start=code_shas()
    if args.output.exists():raise FileExistsError(args.output)
    windows=c['train'];dev=c['development']
    if max(w['end'] for w in windows)>dev['start']:raise ValueError('training exceeds development cutoff')
    if any(w['day']>12 for w in windows):raise ValueError('benchmark history cutoff12 exceeded')
    if len({w['day'] for w in windows})!=len(windows):raise ValueError('first experiment requires distinct train days')
    hosts=pd.read_csv(c['cohort'],dtype=str).host_id.tolist();args.output.mkdir(parents=True)
    observations=[joblib.load(prepare(w,hosts,c['feature_dir'])) for w in windows]
    if args.reuse_transforms:
        transforms=joblib.load(args.reuse_transforms)
        if transforms.train_windows!=windows:raise ValueError('reused transforms have different history')
    else:transforms=BlockFeatures().fit(observations).calibrate(observations)
    joblib.dump(transforms,args.output/'transforms.joblib',compress=3)
    x=[transforms.transform(o) for o in observations];active=[o['events']>=300 for o in observations]
    for name,values in zip([w['name'] for w in windows],x):np.save(args.output/f'{name}.npy',values)
    flow_sha256={o['window']['name']:o['flow_sha256'] for o in observations}
    del observations
    frame=lambda z:pd.DataFrame(np.asarray(z,float),index=pd.Index(hosts,name='host_id')).add_prefix('emb_')
    frame(plain(x[0])).to_csv(args.output/'plain_source.csv')
    candidates={'plain':{'mode':'plain','dim':512,'source_sha256':sha(args.output/'plain_source.csv')}}
    for mode in ['no_graph','graph']:
        model,log=pretrain(x,active,mode,steps=c.get('steps',600),seed=c.get('seed',17))
        checkpoint={'state':model.state_dict(),'dim':model.dim,'mode':mode,'hidden':model.hidden}
        torch.save(checkpoint,args.output/f'{mode}.pt');frame(encode(model,x[0])).to_csv(args.output/f'{mode}_source.csv')
        (args.output/f'{mode}_history.json').write_text(json.dumps(log,indent=2)+'\n')
        pooled=pooled_source(model,x,[w['day'] for w in windows],presence=active)
        frame(pooled).to_csv(args.output/f'{mode}_recent_source.csv')
        import shutil
        shutil.copyfile(args.output/f'{mode}.pt',args.output/f'{mode}_recent.pt')
        candidates[mode+'_recent']={'mode':mode,'source_strategy':'history_pool','dim':512,'source_sha256':sha(args.output/f'{mode}_recent_source.csv'),'checkpoint_sha256':sha(args.output/f'{mode}_recent.pt')}
        candidates[mode]={'mode':mode,'dim':512,'source_sha256':sha(args.output/f'{mode}_source.csv'),'checkpoint_sha256':sha(args.output/f'{mode}.pt')}
    freeze={'config':c,'train':windows,'training_end':max(w['end'] for w in windows),'hosts':hosts,'candidates':candidates,
            'transforms_sha256':sha(args.output/'transforms.joblib'),'code_sha256':code_at_start,'train_flow_sha256':flow_sha256,
            'development_test_read':False,'versions':versions(),'train_feature_sha256':{w['name']:sha(args.output/f"{w['name']}.npy") for w in windows}}
    (args.output/'frozen_training.json').write_text(json.dumps(freeze,indent=2)+'\n');print('training frozen',args.output,flush=True)


if __name__=='__main__':main()
