import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize,StandardScaler

from experiments.fingerprint_aligned import (aligned_pairs,block_tensor,fused_frame,
    fused_tensor,history_input,passes_gate,gallery_labels)
from experiments.fingerprint_donors import DonorHead
from experiments.fingerprint_transfer import sparse_partner_scores


def test_training_fusion_matches_exported_geometry():
    rng=np.random.default_rng(2)
    values=rng.normal(size=(30,12)).astype(np.float32)
    scaler=StandardScaler().fit(normalize(values))
    head=DonorHead(scaler);head.net.eval()
    x=torch.from_numpy(normalize(scaler.transform(normalize(values))).astype(np.float32))
    with torch.no_grad(): branches=head.net(x)
    for weight in [0.5,1.0]:
        actual=fused_tensor(torch.from_numpy(values),branches,weight).numpy()
        exported=fused_frame(pd.DataFrame(values),head,weight).to_numpy()
        np.testing.assert_allclose(actual,normalize(exported),atol=1e-6)


def test_loss_backpropagates_through_the_full_fused_representation():
    torch.set_num_threads(1);torch.manual_seed(7)
    base=torch.randn(4,8)
    learned=torch.randn(4,4,requires_grad=True)
    z=fused_tensor(base,learned,1.0)
    scores=sparse_partner_scores(z,np.arange(4),sp.eye(4,format='csr',dtype=np.float32),np.arange(4))
    target=torch.tensor([1,0,3,2])
    loss=torch.nn.functional.cross_entropy(scores.clamp_min(1e-30).log(),target)
    loss.backward()
    assert learned.grad is not None and learned.grad.abs().sum()>0
    assert torch.isfinite(learned.grad).all()
    assert not base.requires_grad


def test_historical_input_uses_late_window_with_original_cohort(tmp_path):
    base,source,out=[tmp_path/name for name in ['base','day03','history']]
    for directory in [base,source]: (directory/'flows').mkdir(parents=True)
    (base/'flows/train.csv').write_text('src,dst\na,p\n')
    (source/'flows/test.csv').write_text('src,dst\nb,q\n')
    pd.DataFrame({'host_id':['a','b'],'events_train':[400,500]}).to_csv(base/'cohort.csv',index=False)
    pd.DataFrame({'host_id':['foreign']}).to_csv(source/'cohort.csv',index=False)
    (base/'manifest.json').write_text(json.dumps({'windows':{'train':{'start':0,'end':3600}},'config':{}}))
    (source/'manifest.json').write_text(json.dumps({'windows':{'test':{'start':90000,'end':93600}}}))
    data=history_input(base,source,out)
    assert data.cohort==['a','b']
    assert data.manifest['windows']['validation']=={'start':90000,'end':93600}
    assert (out/'flows/validation.csv').resolve()==(source/'flows/test.csv').resolve()


def test_contact_source_stays_day02_and_donor_population_matches_historical_target(tmp_path,monkeypatch):
    import experiments.fingerprint_aligned as module
    transfer=tmp_path/'transfer';transfer.mkdir()
    sp.save_npz(transfer/'source0.npz',sp.eye(4,format='csr',dtype=np.float32))
    monkeypatch.setattr(module,'TRANSFER',transfer)
    for day in range(3,10):
        future=sp.csr_matrix([[0,400,0,0],[400,0,0,0],[0,0,0,0],[0,0,0,10]],dtype=np.float32)
        sp.save_npz(tmp_path/f'future{day:02d}.npz',future)
        np.save(tmp_path/f'future_active{day:02d}.npy',np.array([True,True,False,False]))
    pairs=aligned_pairs(tmp_path,['a','b','c','d'],['p','q','r','s'])
    assert len(pairs)==7
    for pair in pairs:
        assert pair.source[0,0]==1 and pair.source[1,1]==1
        assert pair.source[2:].nnz==0
        np.testing.assert_array_equal(pair.active,[0,1])


def test_80_gate_checks_both_days_without_rounding():
    assert passes_gate({'day10':80.,'day12':85.})
    assert not passes_gate({'day10':90.,'day12':79.999999})
    assert not passes_gate({'day10':90.})
    assert not passes_gate({'day10':float('inf'),'day12':90.})
    assert not passes_gate({'day10':90.,'day12':float('nan')})


def test_identity_labels_follow_active_gallery_positions():
    gallery,labels=gallery_labels(np.array([4,0]),np.array([True,False,True,False,True]))
    np.testing.assert_array_equal(gallery,[0,2,4])
    np.testing.assert_array_equal(labels,[2,0])
