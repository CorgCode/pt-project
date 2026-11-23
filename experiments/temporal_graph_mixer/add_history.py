"""Добавление source из train-истории до выбора на development."""
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np
from .data import sha,prepare
from .evaluate import checkpoint,frame
from .model import pooled_source
from .fit import code_shas,verify_training


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trained',required=True,type=Path)
    a=p.parse_args();f=verify_training(a.trained);c=f['config'];hosts=f['hosts']
    # Принимается только результат обучения, без внешних dev/test входов.
    history=[np.load(a.trained/f"{w['name']}.npy") for w in c['train']]
    presence=[joblib.load(prepare(w,hosts,c['feature_dir']))['events']>=300 for w in c['train']]
    for label,meta in list(f['candidates'].items()):
        if label=='plain' or meta.get('source_strategy')=='history_pool' or label+'_recent' in f['candidates']:continue
        model=checkpoint(a.trained/f'{label}.pt');values=pooled_source(model,history,[w['day'] for w in c['train']],presence=presence)
        name=label+'_recent';path=a.trained/f'{name}_source.csv'
        frame(values,hosts).to_csv(path,float_format='%.17g');shutil.copyfile(a.trained/f'{label}.pt',a.trained/f'{name}.pt')
        f['candidates'][name]={'mode':meta['mode'],'source_strategy':'history_pool','dim':512,'source_sha256':sha(path),'checkpoint_sha256':sha(a.trained/f'{name}.pt')}
        print('added historical profile',name,flush=True)
    f['code_sha256']=code_shas();(a.trained/'frozen_training.json').write_text(json.dumps(f,indent=2)+'\n')


if __name__=='__main__':main()
