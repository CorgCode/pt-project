"""Переносимый bundle выбранной модели без потоков и train-тензоров."""
import argparse,json,shutil
from pathlib import Path
from .data import sha
from .config import versions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['trained','selection','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();selection=json.loads(a.selection.read_text());training=json.loads((a.trained/'frozen_training.json').read_text())
    if sha(a.trained/'frozen_training.json')!=selection['training_sha256']:raise ValueError('training changed')
    if a.output.exists():raise FileExistsError(a.output)
    label=selection['selected']['label'];files=['transforms.joblib','frozen_training.json']
    if label!='plain':
        if sha(a.trained/f'{label}.pt')!=training['candidates'][label]['checkpoint_sha256']:raise ValueError('checkpoint changed')
        files.append(f'{label}.pt')
    if sha(a.trained/'transforms.joblib')!=training['transforms_sha256']:raise ValueError('transforms changed')
    source=a.selection.parent/label/'source.csv'
    if sha(source)!=selection['source_sha256']:raise ValueError('source changed')
    a.output.mkdir(parents=True)
    for name in files:shutil.copyfile(a.trained/name,a.output/name)
    shutil.copyfile(a.selection,a.output/'frozen_selection.json');shutil.copyfile(source,a.output/'source.csv')
    (a.output/'SHARED_COORDINATES').touch()
    (a.output/'manifest.json').write_text(json.dumps({'selected':label,'versions':versions(),
        'files_sha256':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},
        'inference':'python -m experiments.temporal_graph_mixer.infer --trained BUNDLE --selection BUNDLE/frozen_selection.json --flows LATE.csv --cohort COHORT.csv --start START --end END --day DAY --output OUT.csv',
        'contains_raw_training_data':False},indent=2)+'\n')
    print('bundle ready',a.output,flush=True)


if __name__=='__main__':main()
