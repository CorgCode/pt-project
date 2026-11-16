"""Load the authenticated original base with its archived source and legacy names.

Requires the repository archive at --code-root (git commit86d709d), psutil,
pyarrow and the project ML dependencies. Never mutates the original pickle.
"""
import argparse,json,pickle,sys,os
from pathlib import Path

class LegacyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('experiments.fingerprint_'):
            module='experiments.multiday.'+module.removeprefix('experiments.fingerprint_')
        if module.startswith('experiments.multiday_gain'):
            module=module.replace('experiments.multiday_gain','experiments.multiday.gain')
        return super().find_class(module,name)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code-root',type=Path,default=Path('/private/tmp/pt-original-multiday'))
    args=ap.parse_args();root=Path.cwd();archive=args.code_root.resolve()
    import experiments
    experiments.__path__.insert(0,str(archive/'experiments'));sys.path.insert(0,str(archive))
    os.environ['PT_ML_ROOT']=str(root/'PT_PROJECT_ML')
    import numpy as np,pandas as pd
    with (root/'PT_PROJECT_ML/fingerprint_codebook_recent/final/model.pkl').open('rb') as f:
        model=LegacyUnpickler(f).load()
    csv=pd.read_csv(root/'PT_PROJECT_ML/late_test/vectors/multiday_codebook/embeddings.csv').set_index('host_id')
    delta=float(np.abs(model.source.loc[csv.index].to_numpy()-csv.to_numpy()).max())
    if delta>1e-7:raise ValueError('loaded source does not match saved vectors')
    out=root/'runs/original_codebook_recovery';out.mkdir(exist_ok=True)
    record={'source_shape':list(model.source.shape),'source_csv_max_abs_error':delta,'loaded_type':str(type(model)),
            'code_commit':'86d709d','plus_head_available':False}
    (out/'checkpoint_source.json').write_text(json.dumps(record,indent=2));print(record)
if __name__=='__main__':main()
