"""Configuration and environment metadata; no training dependencies."""
import json
import os
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError


def config_file(path):
    def expand(v):
        if isinstance(v,str):return os.path.expandvars(v)
        if isinstance(v,list):return [expand(x) for x in v]
        if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
        return v
    return expand(json.loads(Path(path).read_text()))


def environment():
    result={}
    for package in ['numpy','pandas','scipy','scikit-learn','joblib','torch']:
        try:result[package]=version(package)
        except PackageNotFoundError:result[package]=None
    return result
