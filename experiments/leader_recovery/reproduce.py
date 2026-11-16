"""Re-evaluate the existing authentic multiday_codebook vectors, not reconstructed plus."""
import json,subprocess,shutil
from pathlib import Path
import pandas as pd,numpy as np
from evaluation import scoreboard
from experiments.temporal_donor_identity.common.data import signature
from experiments.temporal_donor_identity.common.evaluate import geometry,with_identity,REQUIRED

def main():
 root=Path('PT_PROJECT_ML');out=Path('runs/original_codebook_recovery');out.mkdir(exist_ok=True)
 cache=out/'cache';cache.mkdir(exist_ok=True)
 params=json.loads(subprocess.check_output(['git','show','70347a3:scoreboard/submissions/multiday_codebook/params.json']))
 checkpoint=root/'fingerprint_codebook_recent/final/model.pkl'
 if signature(checkpoint)!=params['checkpoint_sha256']:raise ValueError('checkpoint is not original base')
 source=pd.read_csv(root/'late_test/vectors/multiday_codebook/embeddings.csv').set_index('host_id')
 expected=json.loads(subprocess.check_output(['git','show','70347a3:scoreboard/results/multiday_codebook.json']))
 results={}
 for window in ['test','test_day15','test_day16']:
  ref=root/'late_test/reference'/window;local=out/'references'/window
  if not local.exists():shutil.copytree(ref,local)
  metadata=json.loads((ref/'manifest.json').read_text())
  metadata['neighbor_ceiling']=scoreboard.NEIGHBORS_IDEAL[window]
  (local/'manifest.json').write_text(json.dumps(metadata,indent=2))
  core=geometry(source,local,root/'late_test/reference/validation',cache)
  late=pd.read_csv(root/f'late_test/vectors/multiday_codebook/{window}.csv').set_index('host_id')
  result=with_identity(source,late,local,core)
  results[window]=result
  (out/f'{window}.json').write_text(json.dumps(result,indent=2))
  print(window,result['score'],result['metrics'],flush=True)
 summary={'checkpoint_sha256':params['checkpoint_sha256'],'score_mean':float(np.mean([r['score'] for r in results.values()])),
 'by_window':{w:{'score':r['score'],'metrics':r['metrics']} for w,r in results.items()},
 'published_mean':float(scoreboard.overall_score([expected],REQUIRED).iloc[0]),'method':'original multiday_codebook, not plus'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
