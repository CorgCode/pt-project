"""Full day13 evaluation after a frozen historical donor pilot."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .train_identity import sha
from experiments.masked_multiday.common.evaluate import geometry,with_identity


def main():
    p=argparse.ArgumentParser()
    for n in ('data','training','late','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    completion=json.loads((a.training/'completed.json').read_text())
    source_file=a.training/'source.csv'
    if (sha(a.training/'model.pt')!=completion['checkpoint_sha256'] or
        sha(source_file)!=completion['source_sha256'] or completion['protocol']['test_used']):
        raise ValueError('donor provenance mismatch')
    source=pd.read_csv(source_file,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    late=pd.read_csv(a.late,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    a.output.mkdir(parents=True,exist_ok=False)
    core=geometry(source,a.reference,a.data/'late_test/reference/validation',a.output/'cache')
    result=with_identity(source,late,a.reference,core)
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    summary={'score':result['score'],'metrics':result['metrics'],'source_sha256':sha(source_file),
             'late_sha256':sha(a.late),'benchmark_used':False,'day13_is_development':True}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
