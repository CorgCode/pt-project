"""След эксперимента: конфигурация, версии и хеши кода без чтения содержимого test."""
from pathlib import Path
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json


def record(output,config):
    versions={}
    for package in ('numpy','pandas','scipy','scikit-learn','joblib','torch'):
        try: versions[package]=version(package)
        except PackageNotFoundError: versions[package]=None
    root=Path(__file__).parent
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py')) if not p.name.startswith('test_')}
    files={}
    for part in ('train','development','test'):
        for row in config.get(part,[]):
            path=Path(row['flows']);stat=path.stat()
            files[row['name']]={'path':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                'content_read_for_protocol':False}
    result={'config':config,'versions':versions,'code_sha256':code,'files':files,
            'note':'OOF адаптера условен на уже замороженном source. Происхождение source должно проверяться отдельно.'}
    Path(output,'protocol.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
