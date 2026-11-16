import numpy as np
import pytest
from evaluation.partners import _mid_ranks
from .run import rr,duplicate_bound,greedy,predicted_utility


def test_first_any_and_midrank_match_official_with_ties():
    x=np.array([4.,4.,4.,2.,0.,0.,10.]); valid=np.array([1,1,1,1,1,1,0],bool)
    targets=np.array([1,4]);masked=x.copy();masked[~valid]=-np.inf
    assert rr(x,valid,targets)==pytest.approx(1/_mid_ranks(masked,targets).min())
    assert rr(np.zeros(7),valid,targets)==pytest.approx(2/7)


def test_duplicate_bound_holds_for_arbitrary_nonnegative_weights():
    matrix=np.array([[1.,1.,3.,0.,0.],[0.,0.,2.,0.,0.],[2.,2.,0.,1.,0.]])
    groups=[[0,1],[0,1],[2],[3],[4]];valid=np.array([1,1,1,0,1],bool)
    for seed in range(100):
        w=np.random.default_rng(seed).random(3)
        for targets in (np.array([0]),np.array([0,4]),np.array([2])):
            assert rr(w@matrix,valid,targets)<=duplicate_bound(groups,valid,targets)+1e-12


def test_greedy_preserves_initial_when_no_better_swap():
    matrix=np.array([[1.,0.,0.],[0.,2.,0.],[0.,0.,3.],[1.,1.,0.]])
    cosine=np.array([.5,.6,.7,.8]);initial=np.array([0,1]);valid=np.ones(3,bool)
    obj=lambda s:rr(s,valid,np.array([1]))
    before=obj((matrix[initial]*cosine[initial,None]).sum(0))
    chosen,votes,_=greedy(matrix,cosine,valid,initial,np.arange(4),obj)
    assert len(set(chosen))==2
    assert obj(votes)>=before
    assert np.allclose(votes,(matrix[chosen]*cosine[chosen,None]).sum(0))


def test_history_utility_uses_candidate_ranks_not_truth():
    votes=np.array([3.,2.,2.,0.]);valid=np.ones(4,bool);pred=np.array([0,1]);prob=np.array([.5,.5])
    assert predicted_utility(votes,valid,pred,prob)==pytest.approx(.5+.5/2.5)
