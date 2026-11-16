import numpy as np
import torch
from .run import soft_topk,aligned_logits


def test_topk_mass_and_gradient_against_finite_difference():
    x=torch.tensor([[.1,.2,.3,.4,.5]],dtype=torch.float64,requires_grad=True)
    w=soft_topk(x,k=2,temperature=.1)
    assert abs(float(w.sum().detach())-2)<1e-8
    coeff=torch.tensor([[1.,-2.,3.,4.,-1.]],dtype=torch.float64)
    analytic=torch.autograd.grad((w*coeff).sum(),x)[0]
    for j in range(5):
        shift=torch.zeros_like(x);shift[0,j]=1e-5
        numeric=((soft_topk(x+shift,2,.1)-soft_topk(x-shift,2,.1))*coeff).sum()/2e-5
        assert torch.allclose(analytic[0,j],numeric,atol=1e-6)
    assert abs(float(analytic.sum()))<1e-8


def test_procrustes_logits_and_gradients_are_finite():
    torch.manual_seed(17)
    a=torch.randn(40,8,requires_grad=True);b=torch.randn(40,8,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(aligned_logits(a,b),torch.arange(20))
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_topk_low_temperature_recovers_discrete_selection():
    x=torch.tensor([[.1,.3,.9,.6]],dtype=torch.float64)
    assert np.array_equal((soft_topk(x,2,.001)>.5).numpy(),[[False,False,True,True]])
