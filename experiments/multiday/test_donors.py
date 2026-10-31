import cloudpickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from experiments.multiday.donors import (DonorHead, donor_targets, peer_matrix,
                                           transfer_loss, train)


def test_donors_exclude_old_contacts_self_and_unseen_future_peers():
    # p,q,r,unseen. У a новый q: его знают b и d, но не c.
    source=sp.csr_matrix([[4,0,0,0],[0,2,0,0],[0,0,3,0],[0,7,0,0]],dtype=np.float32)
    future=sp.csr_matrix([[1,1,0,1],[0,1,1,0],[1,0,1,0],[0,1,0,0]],dtype=np.float32)
    targets=donor_targets(source,future).toarray()
    assert targets[0,1]>0 and targets[0,3]>0 and targets[0,2]==0
    np.testing.assert_array_equal(np.diag(targets),np.zeros(4))
    np.testing.assert_allclose(targets.sum(1),[1,1,1,0])
    assert targets[1,2]==1 and targets[2,0]==1


def test_peer_labels_combine_both_directions_and_align_by_host():
    index=pd.MultiIndex.from_tuples([('b','o:p'),('a','i:p'),('a','o:p')],names=['h','t'])
    counts={'counts':{'peers':pd.Series([3,2,5],index=index)}}
    matrix=peer_matrix(counts,['a','b'],['p']).toarray()
    np.testing.assert_array_equal(matrix,[[7],[3]])
    np.testing.assert_array_equal(peer_matrix(counts,['b','a'],['p']).toarray(),matrix[::-1])


def test_useful_donor_probability_is_learnable_and_self_is_never_a_target():
    torch.set_num_threads(1); torch.manual_seed(3)
    x=torch.eye(4)
    net=torch.nn.Linear(4,4)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.05)
    target=torch.tensor([[0.,1,0,0],[1.,0,0,0],[0,0,0,1.],[0,0,1.,0]])
    queries=torch.arange(4)
    def loss(): return transfer_loss(torch.nn.functional.normalize(net(x),dim=1),queries,target)
    before=float(loss().detach())
    for _ in range(20):
        value=loss(); optimizer.zero_grad(); value.backward(); optimizer.step()
    assert float(loss().detach())<before*0.25


def test_head_predictions_follow_features_after_serialization():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(30,6)).astype(np.float32)
    head=DonorHead(StandardScaler().fit(x))
    expected=head.encode(x)
    restored=cloudpickle.loads(cloudpickle.dumps(head))
    np.testing.assert_allclose(restored.encode(x[::-1]),expected[::-1],atol=1e-6)


def test_actual_small_training_writes_a_finite_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setenv('PT_TORCH_THREADS','1')
    rng=np.random.default_rng(7)
    np.save(tmp_path/'features.npy',rng.normal(size=(3,4,6)).astype(np.float32))
    target=sp.csr_matrix([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],dtype=np.float32)
    for i in range(2): sp.save_npz(tmp_path/f'targets{i}.npz',target)
    train(tmp_path,0.25,epochs=2,checkpoints=(2,))
    with (tmp_path/'donor025/epoch2.pkl').open('rb') as stream:
        head=cloudpickle.load(stream)
    assert np.isfinite(head.encode(rng.normal(size=(4,6)))).all()
    assert (tmp_path/'donor025/complete.json').exists()
