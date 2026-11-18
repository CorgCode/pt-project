import numpy as np
import pandas as pd
import torch
from .data import prepare_window
from .model import CodebookDirectedGNN
from .swaps import reciprocal_rank,swap_loss


def test_bidirectional_bytes_and_external_host(tmp_path):
    frame=pd.DataFrame([['A','B','6','443',100,900,1,9,0,1],
                        ['A','X','6','443',7,5,1,1,601,1]],
       columns=['src','dst','proto','dst_port','bytes','resp_bytes','packets','resp_packets','ts','duration'])
    file=tmp_path/'flows.csv';frame.to_csv(file,index=False)
    result=prepare_window(file,['A','B'],0,3600,cap=8)
    assert len(result['graphs'])==6
    np.testing.assert_array_equal(result['totals'][0,:,:2].sum(0),[107,905])
    np.testing.assert_array_equal(result['totals'][1,:,:2].sum(0),[900,100])
    np.testing.assert_array_equal(result['hourly_bytes_int64'][0],[107,905])
    assert result['hourly_bytes_int64'].dtype==np.int64
    np.testing.assert_allclose(result['graphs'][0]['out_attr'][0,1:3],np.log1p([100,900]))
    np.testing.assert_allclose(result['graphs'][0]['in_attr'][0,1:3],np.log1p([900,100]))


def test_zero_residual_and_graph_gradients():
    torch.manual_seed(4)
    model=CodebookDirectedGNN(width=16)
    x=torch.rand(4,6,8)
    edge={'out_center':torch.tensor([0,1]),'out_neighbor':torch.tensor([1,2]),'out_attr':torch.rand(2,7),
          'in_center':torch.tensor([1,2]),'in_neighbor':torch.tensor([0,1]),'in_attr':torch.rand(2,7)}
    g=model.graph(x,[edge]*6,n_output=4)
    source=torch.randn(4,896);late=torch.randn(4,896)
    assert torch.equal(model.source(source,g),source)
    assert torch.equal(model.late(late,g),late)
    assert not torch.allclose(g,model.graph(x,[edge]*6,n_output=4,use_graph=False))
    with torch.no_grad():model.late_residual.weight.normal_(0,.01)
    model.late(late,g).square().sum().backward()
    assert model.layers[0].messages[0].weight.grad.abs().sum()>0


def test_swap_midrank_mask_and_gradient():
    valid=np.array([True,True,True,False]);positive=np.array([False,True,False,False])
    assert reciprocal_rank(np.array([2.,2.,1.,50.]),valid,positive)==2/3
    vectors=torch.tensor([[1.,0.],[.9,.1],[.1,.9]],requires_grad=True)
    loss=swap_loss(vectors,[(0,1,2,.5)])
    loss.backward()
    assert torch.isfinite(vectors.grad).all() and vectors.grad.norm()>0
