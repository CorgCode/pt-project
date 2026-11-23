import copy
import csv
import pytest
from .scoreboard import render,REQUIRED


def payload():
    rows=[]
    for name in ['similarity_graph_recent_control','directed_bytes_gnn','temporal_graph_mixer_v2']:
        for window in ['test','test_day15','test_day16']:
            rows.append({'method':name,'window':window,'score':50.,'dim':512,'n_hosts':100,
                         'metrics':{k:.6 for k in REQUIRED}})
    dev=copy.deepcopy(rows[-1]);dev.update(window='development13',score=91.2345);rows.append(dev)
    return {'records':rows}


def test_development_is_not_averaged_into_same_methods_benchmark(tmp_path):
    render(payload(),tmp_path)
    benchmark,development=(tmp_path/'README.md').read_text().split('## Development дня 13')
    assert '91.2345' not in benchmark and '91.2345'in development
    with (tmp_path/'scoreboard.csv').open()as stream:rows=list(csv.DictReader(stream))
    assert rows[-1]['protocol']=='development13'


def test_local_table_rejects_missing_identity(tmp_path):
    data=payload();data['records'][0]['metrics'].pop('identity_top1')
    with pytest.raises(ValueError,match='пяти обязательных'):render(data,tmp_path)
