"""Run explicit rolling configurations; refuse a future-trained frozen base.

Every config owns its base checkpoint, train/dev windows, and five seeds. This
wrapper never truncates history around a base trained on later days. Later
independent closed holdouts must be supplied by the team; none is invented here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from experiments.similarity_graph.prepare import config_file
from experiments.similarity_graph.data import Window,check_schedule


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',nargs='+',required=True,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);records=[]
    for i,path in enumerate(args.configs):
        config=config_file(path)
        train=[Window(**w) for w in config['train']];dev=[Window(**w) for w in config['development']]
        check_schedule(train,dev,[Window(**w) for w in config.get('test',[])])
        if len(config.get('seeds',[]))!=5:raise ValueError('rolling comparison requires five predeclared seeds')
        folder=args.output/f'cutoff{i}';folder.mkdir()
        subprocess.run([sys.executable,'-m','experiments.decoupled_multiday.train','--config',str(path),
                        '--output',str(folder/'trained')],check=True)
        subprocess.run([sys.executable,'-m','experiments.decoupled_multiday.tune_partner','--config',str(path),
                        '--trained',str(folder/'trained'),'--output',str(folder/'head_selected')],check=True)
        subprocess.run([sys.executable,'-m','experiments.decoupled_multiday.run','--config',str(path),
                        '--trained',str(folder/'head_selected'),'--output',str(folder/'evaluation')],check=True)
        records.append({'config':str(path),'train_through_day':max(w.day for w in train),
                        'development_days':[w.day for w in dev],'output':str(folder)})
        (args.output/'rolling.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
