import numpy as np
import scipy.sparse as sp
import torch

from experiments.multiday.listwise import candidate_masks,listwise_loss
from experiments.multiday.transfer import PartnerPair


def test_masks_exclude_old_self_and_unavailable_peers():
    source=sp.csr_matrix([[1.,0,0,0],[0,2,3,0]])
    future=sp.csr_matrix([[0.,1,1,8],[1,0,0,0]])
    pair=PartnerPair(source,future,['b','a'],['a','b','c','d'])
    valid,positive=candidate_masks(pair,np.array([0]))
    assert valid.tolist()==[[False,False,True,False]]
    assert positive.tolist()==[[False,False,True,False]]


def test_full_list_loss_detects_an_unsampled_hard_negative():
    valid=torch.ones((1,4),dtype=torch.bool)
    positive=torch.tensor([[True,False,False,False]])
    easy=torch.tensor([[0.8,0.01,0.01,0.01]],requires_grad=True)
    hard=torch.tensor([[0.8,0.01,0.01,0.99]],requires_grad=True)
    assert listwise_loss(hard,valid,positive)>listwise_loss(easy,valid,positive)
    listwise_loss(hard,valid,positive).backward()
    assert hard.grad[0,0]<0 and hard.grad[0,3]>0


def test_any_true_peer_can_win_and_masked_scores_have_no_effect():
    scores=torch.tensor([[0.8,0.1,0.9,100.]],requires_grad=True)
    valid=torch.tensor([[True,True,True,False]])
    positive=torch.tensor([[True,False,True,False]])
    swapped=scores.detach()[:,[2,1,0,3]]
    torch.testing.assert_close(listwise_loss(scores,valid,positive),listwise_loss(swapped,valid,positive))
    listwise_loss(scores,valid,positive).backward()
    assert scores.grad[0,3]==0
