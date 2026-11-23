"""Полный запуск; benchmark включается отдельным флагом после выбора на dev."""
import argparse,json,subprocess,sys
from pathlib import Path
from .config import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark',action='store_true')
    a=p.parse_args();c=load(a.config)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);config=a.output/'config.json';config.write_text(json.dumps(c,indent=2)+'\n')
    def run(name,*args):subprocess.run([sys.executable,'-m','experiments.temporal_graph_mixer.'+name,*map(str,args)],check=True)
    run('data','--config',config,'--part','train')
    run('fit','--config',config,'--output',a.output/'pretrained')
    run('adapt','--trained',a.output/'pretrained','--output',a.output/'trained','--steps',c.get('finetune_steps',200))
    run('add_history','--trained',a.output/'trained')
    run('continue_identity','--trained',a.output/'trained','--output',a.output/'final_trained','--steps',c.get('global_gallery_steps',600))
    run('add_weights','--trained',a.output/'final_trained')
    run('evaluate','--config',config,'--trained',a.output/'final_trained','--output',a.output/'development','--part','development')
    if a.benchmark:
        c['selection']=str((a.output/'development/frozen_selection.json').resolve());config.write_text(json.dumps(c,indent=2)+'\n')
        run('evaluate','--config',config,'--trained',a.output/'final_trained','--output',a.output/'benchmark','--part','benchmark')


if __name__=='__main__':main()
