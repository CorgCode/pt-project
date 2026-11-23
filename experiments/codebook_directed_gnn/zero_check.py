"""Контроль нулевой поправки на сохранённых векторах оригинальной основы."""
import argparse
import json
from pathlib import Path
import shutil
from .base import load_base
from experiments.temporal_graph_mixer.common.evaluate import geometry,with_identity
import pandas as pd
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','archive','source','late','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    model=load_base(a.data,a.archive)
    source=pd.read_csv(a.source,index_col=0,float_precision='round_trip')
    if not np.allclose(model.source.loc[source.index],source,atol=1e-7,rtol=0):
        raise ValueError('CSV не соответствует оригинальному codebook')
    a.output.mkdir(parents=True);shutil.copyfile(a.source,a.output/'source.csv')
    shutil.copyfile(a.late,a.output/'late.csv')
    source=pd.read_csv(a.output/'source.csv',index_col=0)
    late=pd.read_csv(a.output/'late.csv',index_col=0)
    result=with_identity(source,late,a.reference,geometry(source,a.reference,
        a.data/'late_test/reference/validation',a.output/'cache'))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result['score'])


if __name__=='__main__':main()
