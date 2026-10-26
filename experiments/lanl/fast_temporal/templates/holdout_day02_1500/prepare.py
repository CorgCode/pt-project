from datetime import datetime
import json
from pathlib import Path
import sys
import os
ROOT = Path(os.environ["PT_REPO_ROOT"])
WORK = Path(os.environ["PT_RUN_ROOT"])
sys.path.insert(0, str(ROOT))
import pipeline.prepare as prepare_module
from pipeline.windows import PipelineConfig
from pipeline.reference import build_reference
OUT = Path(__file__).resolve().parent
status = {'state':'running', 'pid':os.getpid(), 'started_at':datetime.now().astimezone().isoformat()}
def log(phase, **kwargs):
    status.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **kwargs)
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(datetime.now().isoformat(),phase,kwargs,flush=True)
original = prepare_module.read_canonical
def reader(*args, **kwargs):
    seen = 0
    for chunk in original(*args, **kwargs):
        seen += len(chunk)
        if seen % 5000000 == 0:
            log('prepare',rows_scanned=seen)
        yield chunk
prepare_module.read_canonical = reader
try:
    log('prepare', test_window='day02 15:00–15:30; fixed before model selection')
    if not (OUT/'processed/manifest.json').exists():
        prepare_module.prepare([Path(os.environ["PT_LANL_RAW"])],
                               PipelineConfig.from_json(OUT/'config.json'),OUT/'processed',
                               data_kind='lanl',source_note='Existing LANL day02 archive on SSD; untouched 15:00–15:30 evaluation window selected 2026-10-24.')
    log('reference')
    build_reference(OUT/'processed',OUT/'reference',pairs_per_anchor=10)
    log('completed',state='completed',finished_at=datetime.now().astimezone().isoformat())
except BaseException as e:
    log('failed',state='failed',error=repr(e))
    raise
