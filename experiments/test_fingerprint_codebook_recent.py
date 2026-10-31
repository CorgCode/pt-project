import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import experiments.fingerprint_codebook_recent as module


def fixture(tmp_path,monkeypatch,end=100):
    old=tmp_path/'old';old.mkdir();recent=tmp_path/'recent';recent.mkdir()
    source=tmp_path/'source';(source/'fingerprint_transfer').mkdir(parents=True)
    monkeypatch.setattr(module,'OLD',old);monkeypatch.setattr(module,'RECENT',recent)
    monkeypatch.setattr(module,'SOURCE',source)
    (old/'hosts.json').write_text(json.dumps(['a','b']))
    (old/'anchor.npy').touch()
    (source/'fingerprint_transfer/vocabulary.json').write_text(json.dumps(['peer']))
    for day in range(3,10):
        for name in [f'historical{day:02d}.npy',f'future{day:02d}.npz',f'future_active{day:02d}.npy']:(old/name).touch()
    updates=[]
    for day in [10,11,12]:
        updates.append({'day':day,'data':str(tmp_path/f'day{day}')})
        pd.DataFrame(np.ones((1,896)),index=['a']).to_csv(recent/f'observation{day}.csv')
    (recent/'history_ready.json').write_text(json.dumps({'updates':updates}))
    def data(path):
        windows={'validation':{'start':200}} if Path(path).name=='paired_validation' else {'test':{'start':50,'end':end}}
        return SimpleNamespace(root=path,manifest={'windows':windows})
    from pathlib import Path
    monkeypatch.setattr(module,'ProcessedData',data)
    counts={'events':np.array([350,100]),'counts':{'peers':pd.Series([30,10],
        index=pd.MultiIndex.from_tuples([('a','o:peer'),('a','i:unknown')],names=['h','t']))}}
    monkeypatch.setattr(module,'read_fingerprints',lambda *args:counts)
    return tmp_path/'work'


def test_recent_inputs_freeze_cohort_and_drop_unavailable_peer_labels(tmp_path,monkeypatch):
    work=fixture(tmp_path,monkeypatch);module.inputs(work)
    root=work/'history'
    assert np.load(root/'historical10.npy').shape==(2,896)
    np.testing.assert_array_equal(np.load(root/'future_active10.npy'),[True,False])
    np.testing.assert_array_equal(sp.load_npz(root/'future10.npz').toarray(),[[30],[0]])
    assert json.loads((root/'inputs_ready.json').read_text())['training_end_day']==12


def test_recent_inputs_reject_window_crossing_development(tmp_path,monkeypatch):
    work=fixture(tmp_path,monkeypatch,end=201)
    with pytest.raises(ValueError,match='пересекает'):module.inputs(work)
    assert not (work/'history/inputs_ready.json').exists()


def test_blind_test_is_not_opened_without_material_validation_gain(tmp_path,monkeypatch):
    (tmp_path/'selection_summary.json').write_text(json.dumps({'test_will_run':False,'validation_gain':0.2}))
    calls=[];monkeypatch.setattr(module,'run',lambda *args:calls.append(args))
    module.final_test(tmp_path)
    assert not calls
    assert json.loads((tmp_path/'state.json').read_text())['stage']=='complete'


def test_completed_blind_result_is_reused_without_another_evaluation(tmp_path,monkeypatch):
    (tmp_path/'final').mkdir()
    (tmp_path/'final/REPORT.md').write_text('done')
    (tmp_path/'final/summary.json').write_text(json.dumps({'scores':{'winner':80.}}))
    calls=[];monkeypatch.setattr(module,'run',lambda *args:calls.append(args))
    module.final_test(tmp_path)
    assert not calls
    assert json.loads((tmp_path/'state.json').read_text())['scores']['winner']==80.
