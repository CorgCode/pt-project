"""Отдельная адаптация голосования с сохранением исходного checkpoint."""
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np,torch
from .data import sha,prepare
from .evaluate import checkpoint,frame
from .model import encode
from .finetune import tune
from .fit import code_shas,verify_training


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trained',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    a=p.parse_args();f=verify_training(a.trained);c=f['config'];torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for file in a.trained.iterdir():
        if file.is_file():shutil.copyfile(file,a.output/file.name)
    x=[np.load(a.trained/f"{w['name']}.npy") for w in c['train']]
    obs=[joblib.load(prepare(w,f['hosts'],c['feature_dir'])) for w in c['train']]
    model=checkpoint(a.trained/'graph.pt');model,history=tune(model,x,obs,steps=a.steps,seed=c.get('seed',17))
    torch.save({'state':model.state_dict(),'mode':model.mode,'dim':model.dim,'hidden':model.hidden},a.output/'adapted.pt')
    frame(encode(model,x[0]),f['hosts']).to_csv(a.output/'adapted_source.csv',float_format='%.17g')
    (a.output/'adapted_history.json').write_text(json.dumps(history,indent=2)+'\n')
    f['candidates']['adapted']={'mode':'graph','dim':512,'source_sha256':sha(a.output/'adapted_source.csv'),'checkpoint_sha256':sha(a.output/'adapted.pt')}
    f['finetune_steps']=a.steps;f['code_sha256']=code_shas();f['parent_training_sha256']=sha(a.trained/'frozen_training.json')
    (a.output/'frozen_training.json').write_text(json.dumps(f,indent=2)+'\n')
    print('adaptation frozen',a.output,flush=True)


if __name__=='__main__':main()
