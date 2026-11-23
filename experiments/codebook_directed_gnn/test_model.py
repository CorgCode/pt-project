import numpy as np
import pandas as pd
import pytest
torch=pytest.importorskip('torch')
from .model import CodebookDirectedGNN
from .data import prepare_window
from .swaps import reciprocal_rank,swap_loss


def test_zero_residual_preserves_original_double_precision_vectors():
    m=CodebookDirectedGNN();base=torch.randn(7,896,dtype=torch.float64);g=torch.randn(7,128)
    torch.testing.assert_close(m.source(base,g),base,atol=0,rtol=0)
    torch.testing.assert_close(m.late(base,g),base,atol=0,rtol=0)


def test_six_bin_preparation_and_directional_bytes(tmp_path):
    p=tmp_path/'flows.csv'
    pd.DataFrame([['a','b','6','443',100,900,1,9,600,0]],
        columns=['src','dst','proto','dst_port','bytes','resp_bytes','packets','resp_packets','ts','duration']).to_csv(p,index=False)
    d=prepare_window(p,['a','b'],0,3600)
    assert d['totals'].shape==(2,6,8)
    np.testing.assert_array_equal(d['totals'][0,1,:4],[100,900,1,9])
    np.testing.assert_array_equal(d['totals'][1,1,:4],[900,100,9,1])
    assert len(d['graphs'])==6


def test_graph_encoder_updates_messages_and_handles_new_query_size():
    torch.manual_seed(17);m=CodebookDirectedGNN()
    x=torch.rand(5,6,8)*10
    g={'out_center':torch.tensor([0,1,2]),'out_neighbor':torch.tensor([1,2,3]),
       'out_attr':torch.rand(3,7),'in_center':torch.tensor([1,2,3]),
       'in_neighbor':torch.tensor([0,1,2]),'in_attr':torch.rand(3,7)}
    a=m.graph(x,[g]*6,n_output=5);b=m.graph(x,[g]*6,n_output=5,use_graph=False)
    assert not torch.allclose(a,b)
    a.square().sum().backward()
    assert m.layers[0].messages[0].weight.grad.abs().sum()>0
    assert a.shape==(5,128)


def test_swap_reward_midrank_and_learning_direction():
    assert reciprocal_rank(np.array([10.,10.,0.]),np.ones(3,bool),np.array([True,False,False]))==pytest.approx(2/3)
    z=torch.nn.Parameter(torch.tensor([[1.,0.],[.8,.6],[0.,1.]]))
    def gap():
        n=torch.nn.functional.normalize(z,dim=1)
        return (n[0]*n[2]).sum()-(n[0]*n[1]).sum()
    before=gap().item();loss=swap_loss(z,[(0,1,2,.5)]);loss.backward()
    with torch.no_grad():z-=.01*z.grad
    assert gap().item()>before
