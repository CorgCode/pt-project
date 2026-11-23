"""Загрузка проверенной оригинальной основы с архивным кодом."""
import hashlib
import os
import pickle
import sys
from pathlib import Path

EXPECTED='a1c4bfa15ed0c227aefd981f59b87769aece146efb232da90617e4f561b10211'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('experiments.fingerprint_'):
            module='experiments.multiday.'+module.removeprefix('experiments.fingerprint_')
        if module.startswith('experiments.multiday_gain'):
            module=module.replace('experiments.multiday_gain','experiments.multiday.gain')
        return super().find_class(module,name)


def load_base(data,archive):
    import experiments
    archive=Path(archive).resolve()
    if not (archive/'experiments/multiday/codebook.py').is_file():
        raise FileNotFoundError('Нужен архив 86d709d: experiments/multiday, experiments/overnight, scoreboard')
    experiments.__path__.append(str(archive/'experiments'))
    sys.path.append(str(archive));os.environ['PT_ML_ROOT']=str(Path(data).resolve())
    checkpoint=Path(data)/'fingerprint_codebook_recent/final/model.pkl'
    if sha(checkpoint)!=EXPECTED:raise ValueError('Неверный оригинальный checkpoint')
    with checkpoint.open('rb') as stream:model=LegacyUnpickler(stream).load()
    # Оконные кэши старого объекта не переносятся между словарями и запусками.
    node=model
    while node is not None:
        for key in ['_encoded','_parts_cache','_evaluation_cache','_behavior_cache']:
            if hasattr(node,key):setattr(node,key,{})
        node=getattr(node,'base',None)
    return model
