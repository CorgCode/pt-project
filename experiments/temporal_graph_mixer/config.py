"""Разворачивание конфига, версии и контроль хронологии."""
import json,os,platform
from pathlib import Path
from importlib.metadata import version


def load(path):
    c=json.loads(os.path.expandvars(Path(path).read_text()))
    windows=c['train'];dev=c['development'];tests=c.get('benchmark',[])
    if not windows or windows[0]['day']!=2:raise ValueError('source02 must be first')
    if any(w['end']<=w['start'] for w in [*windows,dev,*tests]):raise ValueError('invalid window bounds')
    ordered=sorted(windows,key=lambda w:w['start'])
    if windows!=ordered or any(a['end']>b['start'] for a,b in zip(windows,windows[1:])):raise ValueError('train must be ordered and disjoint')
    if max(w['end'] for w in windows)>dev['start']:raise ValueError('train overlaps development')
    if any(w['day']>12 for w in windows):raise ValueError('training after day12 requires a separate protocol')
    if tests and dev['end']>min(w['start'] for w in tests):raise ValueError('development overlaps benchmark')
    names=[w['name'] for w in [*windows,dev,*tests]]
    if len(set(names))!=len(names):raise ValueError('duplicate window names')
    return c


def versions():
    return {'python':platform.python_version(),**{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','joblib']}}
