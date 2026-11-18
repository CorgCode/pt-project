"""Verify zero residual preserves the stored baseline and score on day13."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import CodebookDirectedGNN
from experiments.masked_multiday.common.evaluate import geometry, with_identity


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source_path=a.data/'late_test/vectors/multiday_codebook/embeddings.csv'
    late_path=a.data/'fingerprint_codebook_recent/candidates/codebook1_e400/later.csv'
    source=pd.read_csv(source_path,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    late=pd.read_csv(late_path,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    model=CodebookDirectedGNN().eval();torch.manual_seed(17)
    with torch.no_grad():
        z=model.source(torch.from_numpy(source.to_numpy()),torch.randn(len(source),128)).numpy()
        y=model.late(torch.from_numpy(late.to_numpy()),torch.randn(len(late),128)).numpy()
    if not np.array_equal(z,source.to_numpy()) or not np.array_equal(y,late.to_numpy()):
        raise AssertionError('zero residual did not preserve original CSV values')
    # Round-trip both through the same format, then compare all values.
    source_out=a.output/'source.csv'; late_out=a.output/'late.csv'
    pd.DataFrame(z,index=source.index,columns=source.columns).to_csv(source_out)
    pd.DataFrame(y,index=late.index,columns=late.columns).to_csv(late_out)
    source2=pd.read_csv(source_out,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    late2=pd.read_csv(late_out,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    if not np.array_equal(source.to_numpy(),source2.to_numpy()) or not np.array_equal(late.to_numpy(),late2.to_numpy()):
        raise AssertionError('CSV roundtrip changed baseline numeric values')
    core=geometry(source2,a.reference,a.data/'late_test/reference/validation',a.output/'cache')
    result=with_identity(source2,late2,a.reference,core)
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    summary={'score':result['score'],'metrics':result['metrics'],
      'baseline_csv_source_sha256':sha(source_path),'baseline_csv_late_sha256':sha(late_path),
      'zero_output_equal_before_and_after_csv':True,'day':13,'benchmark_used':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
