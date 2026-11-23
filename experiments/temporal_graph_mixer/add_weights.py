"""Небольшая сетка весов stable/dynamic на dev без градиентного обучения."""
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np
from .data import prepare,sha
from .evaluate import checkpoint,frame
from .model import pooled_source
from .fit import code_shas,verify_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--trained',type=Path,required=True);a=p.parse_args()
    f=verify_training(a.trained);c=f['config'];hosts=f['hosts']
    history=[np.load(a.trained/f"{w['name']}.npy") for w in c['train']]
    active=[joblib.load(prepare(w,hosts,c['feature_dir']))['events']>=300 for w in c['train']]
    for mode in ['no_graph_global','graph_global']:
        model=checkpoint(a.trained/f'{mode}.pt')
        for i,weights in enumerate([(1.,2.,.5),(1.,2.,1.),(1.,1.,1.)]):
            label=mode+f'_w{i}';path=a.trained/f'{label}.pt'
            if path.exists():raise FileExistsError(path)
            shutil.copyfile(a.trained/f'{mode}.pt',path)
            source=pooled_source(model,history,[w['day'] for w in c['train']],weights=weights,presence=active)
            frame(source,hosts).to_csv(a.trained/f'{label}_source.csv',float_format='%.17g')
            f['candidates'][label]={'mode':model.mode,'source_strategy':'history_pool','weights':weights,'dim':512,'source_sha256':sha(a.trained/f'{label}_source.csv'),'checkpoint_sha256':sha(path)}
    f['weight_grid']=[[1.,2.,.5],[1.,2.,1.],[1.,1.,1.]];f['code_sha256']=code_shas()
    (a.trained/'frozen_training.json').write_text(json.dumps(f,indent=2)+'\n')


if __name__=='__main__':main()
