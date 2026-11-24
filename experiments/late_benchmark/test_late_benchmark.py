"""Protocol, extraction, byte-range and scoring regressions; no real data needed."""
from __future__ import annotations

import bz2
import copy
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import scoreboard, submit
from experiments.late_benchmark import download
from experiments.late_benchmark.prepare import extract_window, load_protocol, prepare
from experiments.late_benchmark.publish import validate
from pipeline.data import ProcessedData
from pipeline.lanl_netflow import RAW_COLUMNS, canonicalize
from pipeline.prepare import _sha256
from pipeline.windows import window_mask


def test_protocol_includes_last_day_and_preserves_source():
    protocol = load_protocol()
    assert protocol['windows']['train'] == [126000, 129600]
    assert protocol['windows']['validation'] == [129600, 131400]
    assert protocol['windows']['test_day90'] == [7734600, 7738200]
    assert set(protocol['test_days'].values()) == {88, 89, 90}
    assert protocol['cohort_policy'] == 'train_only'


class Response(io.BytesIO):
    status = 206

    def __init__(self, payload, start, end, total):
        super().__init__(payload)
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}'}


def test_downloader_resumes_at_exact_offset(monkeypatch, tmp_path):
    part = tmp_path / 'part'
    part.write_bytes(b'abc')
    calls = []
    def request(url, *, start, end):
        calls.append((start, end))
        return Response(b'def', start, end, 6)
    monkeypatch.setattr(download, 'request', request)
    assert download.fetch_part('unused', part, 0, 5, 6) == 6
    assert calls == [(3, 5)] and part.read_bytes() == b'abcdef'


@pytest.mark.parametrize('status,header', [(200, 'bytes 0-2/3'), (206, 'bytes 1-2/3'), (206, 'bytes 0-2/4')])
def test_wrong_http_ranges_never_append(monkeypatch, tmp_path, status, header):
    response = Response(b'xyz', 0, 2, 3)
    response.status, response.headers['Content-Range'] = status, header
    monkeypatch.setattr(download, 'request', lambda *a, **k: response)
    target = tmp_path / 'part'
    with pytest.raises(download.DownloadError):
        download.fetch_part('unused', target, 0, 2, 3)
    assert not target.exists()


def test_truncated_download_is_resumed_without_duplicate_bytes(monkeypatch, tmp_path):
    calls = []
    def request(url, *, start, end):
        calls.append(start)
        return Response(b'abc' if start == 0 else b'def', start, end, 6)
    monkeypatch.setattr(download, 'request', request)
    monkeypatch.setattr(download.time, 'sleep', lambda _: None)
    target = tmp_path / 'part'
    assert download.fetch_part('unused', target, 0, 5, 6, attempts=2) == 6
    assert calls == [0, 3] and target.read_bytes() == b'abcdef'


@pytest.fixture
def source_and_archives(tmp_path):
    spec = load_protocol()
    base, raw, output = tmp_path/'base', tmp_path/'raw', tmp_path/'processed'
    (base/'flows').mkdir(parents=True); (base/'graph').mkdir(); raw.mkdir()
    hosts = [f'h{i:02}' for i in range(30)]
    pd.DataFrame({'host_id': hosts, 'events_train': 300, 'events_validation': 300,
                  'evaluable_validation': True}).to_csv(base/'cohort.csv', index=False)
    windows = {}
    for name in ('train', 'validation'):
        start, end = spec['windows'][name]
        windows[name] = {'start':start, 'end':end, 'n_flows':30}
        pd.DataFrame({'src':hosts, 'dst':hosts[1:]+hosts[:1], 'dst_port':'443', 'proto':'6',
                      'bytes':10, 'packets':1, 'resp_bytes':5, 'resp_packets':1,
                      'duration':1, 'src_port':'123', 'ts':start+10}).to_csv(base/'flows'/f'{name}.csv', index=False)
        pd.DataFrame(columns=['host_a','host_b','flows','bytes','packets']).to_csv(base/'graph'/f'edges_{name}.csv', index=False)
    (base/'manifest.json').write_text(json.dumps({'data_kind':'synthetic_fixture','synthetic':True,
        'cohort':{'policy':'train_only'},'windows':windows}))
    for name, day in spec['test_days'].items():
        start, end = spec['windows'][name]
        rows = [f'{start+10},1,{host},{hosts[(i+1)%len(hosts)]},6,123,443,1,1,10,5'
                for i,host in enumerate(hosts) for _ in range(150)]
        rows += [f'{end-1},1,h00,h01,6,123,443,1,1,10,5',  # crosses the strict end boundary
                 f'{end},0,h00,h01,6,123,443,1,1,10,5',
                 f'{start+20},1,h00,h00,6,123,443,1,1,10,5']
        (raw/f'netflow_day-{day}.bz2').write_bytes(bz2.compress(('\n'.join(rows)+'\n').encode()))
    return base, raw, output


def test_late_preparation_keeps_exact_source_cohort_and_boundaries(source_and_archives):
    base, raw, output = source_and_archives
    original_hashes = {p.name:_sha256(p) for p in (base/'flows').glob('*.csv')}
    manifest = prepare(base,raw,output,chunksize=317)
    data = ProcessedData(output)
    assert len(data.cohort) == 30 and data.evaluable('test_day90') == data.cohort
    assert manifest['windows']['test_day90']['straddling_dropped'] == 1
    assert manifest['windows']['test_day90']['n_flows'] == 4501  # self-flow retained, excluded from event counts
    assert (output/'flows/train.csv').is_symlink()
    assert original_hashes == {p.name:_sha256(p) for p in (base/'flows').glob('*.csv')}
    cohort = pd.read_csv(output/'cohort.csv')
    assert (cohort.events_test_day90 == 300).all()
    assert prepare(base,raw,output)['cohort']['sha256'] == manifest['cohort']['sha256']
    with (output/'flows/test_day90.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='changed'):
        prepare(base,raw,output)


def test_extraction_matches_existing_pipeline_rules(source_and_archives, tmp_path):
    base, raw, _ = source_and_archives
    interval = load_protocol()['windows']['test_day90']
    raw_rows = pd.read_csv(raw/'netflow_day-90.bz2',header=None,names=RAW_COLUMNS,dtype=str)
    canonical, _ = canonicalize(raw_rows)
    expected = canonical.loc[window_mask(canonical.ts.to_numpy(),canonical.duration.to_numpy(),interval,True)]
    target = tmp_path/'window.csv'
    stats, events, graph = extract_window(raw/'netflow_day-90.bz2',target,interval,[f'h{i:02}' for i in range(30)],chunksize=83)
    actual = pd.read_csv(target,dtype={'src':str,'dst':str,'proto':str,'dst_port':str,'src_port':str})
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False)
    assert stats['n_flows'] == len(expected) and len(events) == 30 and len(graph) == 30


def row(ceiling=.1, value=.055, method='model'):
    return {'method':method,'n_hosts':30,'split':'test','benchmark_id':'late-v1',
        'score_calibration':{'neighbors_ideal':ceiling},'metrics':{scoreboard.NEIGHBORS:value},
        'summary':[{'metric':scoreboard.NEIGHBORS,'title':'Neighbors@10','group':'Behavior','value':value,'chance':.01}],
        'reference_info':{'files_sha256':{'neighbors.csv':'same'},'data_kind':'lanl_declared','synthetic':False}}


def test_late_scores_use_reference_calibration_instead_of_old_day14_constant():
    r = row()
    assert scoreboard.column_scores(r,[scoreboard.NEIGHBORS],window='test')[scoreboard.NEIGHBORS] == pytest.approx(50)
    a, b = row(), row(.2,.105)
    combined = submit.combine_windows({'test':a,'test_day90':b})
    assert 'score_calibration' not in combined
    assert combined['by_window']['test_day90']['score_calibration']['neighbors_ideal'] == .2
    assert scoreboard.overall_score([combined],[scoreboard.NEIGHBORS])['model'] == pytest.approx(50)
    assert '50.0' in scoreboard.window_scores_table([combined],{'test':'88','test_day90':'90'})


@pytest.mark.parametrize('ceiling',[0,-.1,float('nan'),1.1,True])
def test_invalid_score_calibration_fails(ceiling):
    with pytest.raises(ValueError,match='neighbors_ideal'):
        scoreboard.column_scores(row(ceiling),[scoreboard.NEIGHBORS])


def test_undefined_normalization_does_not_remove_behavior_task():
    r = row(.005)
    assert np.isnan(scoreboard.overall_score([r],[scoreboard.NEIGHBORS])['model'])
    del r['score_calibration']
    with pytest.raises(ValueError,match='нормировка'):
        scoreboard.overall_score([r],[scoreboard.NEIGHBORS])


def test_old_and_new_scores_or_calibrations_cannot_be_mixed():
    a,b=row(method='a'),row(method='b')
    scoreboard.check_compatible([a,b])
    b['score_calibration']={'neighbors_ideal':.2}
    with pytest.raises(ValueError,match='бенчмарка'):
        scoreboard.check_compatible([a,b])
    b=copy.deepcopy(a);b['method']='b';del b['benchmark_id'];del b['score_calibration']
    with pytest.raises(ValueError,match='бенчмарка'):
        scoreboard.check_compatible([a,b])


def test_publishing_refuses_missing_day90_or_synthetic_numbers(tmp_path):
    spec=load_protocol()
    with pytest.raises(ValueError,match='no late-window'):
        validate([],{},tmp_path,{},spec)
    with pytest.raises(ValueError,match='real-data'):
        validate([row()],{'data_kind':'synthetic_fixture','synthetic':True},tmp_path,{},spec)
    processed={'data_kind':'lanl_declared','synthetic':False,'windows':{'train':{},'validation':{},'test':{},'test_day89':{}}}
    with pytest.raises(ValueError,match='90'):
        validate([row()],processed,tmp_path,{},spec)


def test_official_evaluator_and_new_calibration_work_together(processed, tmp_path, monkeypatch):
    import shutil
    from pipeline.reference import build_reference
    from experiments.late_benchmark import evaluate as late_evaluate
    from methods.baselines.temporal import RandomBaseline

    data_root, work = tmp_path/'data', tmp_path/'work'
    root = data_root/'processed'
    shutil.copytree(processed, root)
    meta = json.loads((root/'manifest.json').read_text())
    spec = load_protocol()
    spec['bootstrap'] = 0
    spec['windows']['train'] = [meta['windows']['train']['start'],meta['windows']['train']['end']]
    spec['windows']['validation'] = [meta['windows']['validation']['start'],meta['windows']['validation']['end']]
    old = meta['windows']['test']
    old_flows = pd.read_csv(root/'flows/test.csv')
    cohort = pd.read_csv(root/'cohort.csv')
    for window in spec['test_days']:
        start,end = spec['windows'][window]
        flows = old_flows.copy()
        scale = (end-start)/(old['end']-old['start'])
        flows['ts'] = start+(flows.ts-old['start'])*scale
        flows['duration'] *= scale
        flows.to_csv(root/'flows'/f'{window}.csv',index=False)
        meta['windows'][window] = {**old,'start':start,'end':end,'min_time':float(flows.ts.min()),'max_time':float(flows.ts.max())}
        cohort[f'evaluable_{window}'] = cohort.evaluable_test
    meta['benchmark_id'] = spec['benchmark_id']
    meta['config']['windows'] = spec['windows']
    cohort.to_csv(root/'cohort.csv',index=False)
    meta['cohort']['sha256'] = _sha256(root/'cohort.csv')
    (root/'manifest.json').write_text(json.dumps(meta))
    references = data_root/'reference'
    build_reference(root,references,pairs_per_anchor=10)
    for window in spec['test_days']:
        path = references/window/'manifest.json'
        ref = json.loads(path.read_text())
        ref.update(benchmark_id=spec['benchmark_id'],score_calibration={'neighbors_ideal':.4})
        path.write_text(json.dumps(ref))
    (references/'complete.json').write_text('{}')
    data = ProcessedData(root)
    folder = work/'models/random';folder.mkdir(parents=True)
    model = RandomBaseline(dim=4,seed=0)
    model.embed(data,'train').to_csv(folder/'embeddings.csv')
    source_sha = _sha256(folder/'embeddings.csv')
    training = {'source_sha256':source_sha,'new_test_used_in_fit_or_selection':False}
    (folder/'training.json').write_text(json.dumps(training))
    for window in spec['test_days']:
        target = folder/f'{window}.csv'
        model.embed(data,window,data.evaluable(window)).to_csv(target)
        target.with_suffix('.json').write_text(json.dumps({'inputs':{'benchmark_id':spec['benchmark_id'],
            'processed_manifest':_sha256(root/'manifest.json'),'training':source_sha},'sha256':_sha256(target)}))
    monkeypatch.setattr(late_evaluate,'load_protocol',lambda:spec)
    for key in ('DATA','RESULTS','N_BOOT'):
        monkeypatch.setattr(submit,key,getattr(submit,key))  # restore mutations inside score_method at teardown
    result = late_evaluate.score_method('random',data_root,work)
    assert set(result['by_window']) == {'test','test_day89','test_day90'}
    assert all(set(late_evaluate.REQUIRED) <= set(w['metrics']) for w in result['by_window'].values())
    assert all(w['score_calibration']['neighbors_ideal']==.4 for w in result['by_window'].values())
    assert result['reference_info']['synthetic'] is True
    assert np.isfinite(scoreboard.overall_score([result],late_evaluate.REQUIRED)['random'])
    # A fully computed synthetic fixture still cannot replace real LANL results.
    with pytest.raises(ValueError,match='real-data'):
        validate([result],meta,references,{'random':{'status':'trained'}},spec)


def publication_fixture(tmp_path):
    from experiments.late_benchmark.evaluate import REQUIRED
    spec = load_protocol()
    data,work,board = tmp_path/'data',tmp_path/'work',tmp_path/'board'
    for folder in (data/'processed',work/'results',board/'results',board/'data/reference',board/'data/processed'):
        folder.mkdir(parents=True,exist_ok=True)
    windows={name:{'start':bounds[0],'end':bounds[1]} for name,bounds in spec['windows'].items()}
    processed={'benchmark_id':spec['benchmark_id'],'data_kind':'lanl_declared','synthetic':False,
        'windows':windows,'cohort':{'n_hosts':100,'evaluable':{name:100 for name in spec['test_days']}}}
    (data/'processed/manifest.json').write_text(json.dumps(processed))
    (data/'processed/cohort.csv').write_text('host_id\na\n')
    groups=['Поведение','Будущие связи','Будущие связи','Общие собеседники','Узнаваемость']
    values=[.2,.8,.2,.8,.5];chance=[.1,.5,.05,.5,.01]
    per={}
    for name in spec['test_days']:
        folder=data/'reference'/name;folder.mkdir(parents=True)
        (folder/'neighbors.csv').write_text('fixture '+name)
        info={'files_sha256':{'neighbors.csv':_sha256(folder/'neighbors.csv')},'data_kind':'lanl_declared','synthetic':False}
        calibration={'neighbors_ideal':.3}
        (folder/'manifest.json').write_text(json.dumps({'benchmark_id':spec['benchmark_id'],'windows':windows,
            'files_sha256':info['files_sha256'],'score_calibration':calibration}))
        per[name]={'method':'behavioral','n_hosts':100,'split':'test','benchmark_id':spec['benchmark_id'],
            'metrics':dict(zip(REQUIRED,values)),'metrics_ci':{},'reference_info':info,'score_calibration':calibration,
            'summary':[{'metric':key,'title':key,'group':group,'value':value,'chance':baseline}
                       for key,group,value,baseline in zip(REQUIRED,groups,values,chance)]}
    results=[]
    for method in ('behavioral','random'):
        result=submit.combine_windows(copy.deepcopy(per));result.update(method=method,dim=4,
            training_provenance={'new_test_used_in_fit_or_selection':False})
        results.append(result)
        (work/'results'/f'{method}.json').write_text(json.dumps(result))
        sub=board/'submissions'/method;sub.mkdir(parents=True)
        (sub/'README.md').write_text('# test\n'+''.join(f'## {title}\nFilled fixture.\n' for title in submit.CARD_SECTIONS))
    statuses={method:{'status':'trained'} for method in ('behavioral','random')}
    (work/'model_status.json').write_text(json.dumps(statuses))
    (board/'README.md').write_text('HISTORICAL TABLE')
    (board/'scoreboard.csv').write_text('method,score\nlegacy,70\n')
    (board/'results/legacy.json').write_text('{"method":"legacy"}')
    (board/'data/reference/legacy.txt').write_text('historical reference')
    (board/'data/processed/manifest.json').write_text('{}')
    return spec,data,work,board,processed,statuses,results


def test_publication_keeps_historical_numbers_and_promotes_complete_version(tmp_path,monkeypatch):
    from experiments.late_benchmark import publish as promotion
    spec,data,work,board,processed,statuses,results=publication_fixture(tmp_path)
    for key in ('BOARD','DATA','RESULTS','SUBMISSIONS'):
        monkeypatch.setattr(submit,key,getattr(submit,key))
    marker=promotion.publish(data,work,board)
    assert marker['benchmark_id']==spec['benchmark_id']
    assert (board/'history/days14-16/README.md').read_text()=='HISTORICAL TABLE'
    assert (board/'history/days14-16/results/legacy.json').exists()
    assert not (board/'results/legacy.json').exists()
    assert set(p.stem for p in (board/'results').glob('*.json'))=={'behavioral','random'}
    text=(board/'README.md').read_text()
    assert 'День 90' in text and spec['benchmark_id'] in text and 'историческими 70,0' in text
    assert json.loads((board/'benchmark.json').read_text())['benchmark_id']==spec['benchmark_id']
    assert (work/'scoreboard_before_publish/README.md').read_text()=='HISTORICAL TABLE'


def test_publication_rejects_changed_reference_before_touching_active_table(tmp_path):
    from experiments.late_benchmark import publish as promotion
    _,data,work,board,_,_,_=publication_fixture(tmp_path)
    (data/'reference/test_day90/neighbors.csv').write_text('tampered')
    with pytest.raises(ValueError,match='changed before publication'):
        promotion.publish(data,work,board)
    assert (board/'README.md').read_text()=='HISTORICAL TABLE'
    assert not (board/'history').exists()


def test_publication_rejects_wrong_clock_time_even_with_correct_day_label(tmp_path):
    spec,data,work,board,processed,statuses,results=publication_fixture(tmp_path)
    processed['windows']['test_day90']['start']=0
    with pytest.raises(ValueError,match='time windows'):
        validate(results,processed,data/'reference',statuses,spec)


def test_parallel_extraction_matches_serial(source_and_archives,tmp_path):
    base,raw,output=source_and_archives
    a=prepare(base,raw,output,chunksize=317,workers=1)
    other=tmp_path/'parallel'
    b=prepare(base,raw,other,chunksize=317,workers=2)
    assert a['windows']==b['windows'] and a['cohort']==b['cohort']
    for window in load_protocol()['test_days']:
        assert _sha256(output/'flows'/f'{window}.csv')==_sha256(other/'flows'/f'{window}.csv')
        assert _sha256(output/'graph'/f'edges_{window}.csv')==_sha256(other/'graph'/f'edges_{window}.csv')


def test_reused_validation_does_not_require_missing_appledouble_sidecars(tmp_path):
    from experiments.late_benchmark.reference import copy_validation
    source,target=tmp_path/'source',tmp_path/'copy'
    source.mkdir()
    (source/'edges.csv').write_text('unaltered classifier candidates')
    metadata={'files_sha256':{'edges.csv':_sha256(source/'edges.csv'),'._edges.csv':'legacy-metadata-only'}}
    (source/'manifest.json').write_text(json.dumps(metadata))
    original=_sha256(source/'manifest.json')
    copied=copy_validation(source,target)
    assert copied['files_sha256']=={'edges.csv':_sha256(source/'edges.csv')}
    assert copied['excluded_legacy_nonmetric_files']=={'._edges.csv':'legacy-metadata-only'}
    assert copied['source_validation_manifest_sha256']==original
    assert _sha256(source/'manifest.json')==original
    (target/'edges.csv').write_text('modified classifier candidates')
    with pytest.raises(ValueError,match='metric CSV changed'):
        copy_validation(source,target)
