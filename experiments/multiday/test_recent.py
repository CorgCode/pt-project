import numpy as np
import pandas as pd
import pytest

from experiments.multiday.recent import update_profiles,RecentEncoder


def source():
    return pd.DataFrame({'emb_0':[0.,100.,200.]},index=['a','b','c'])


def test_last_observation_uses_latest_past_date_and_keeps_missing_hosts():
    anchor=source();before=anchor.copy()
    observations=[(12,pd.DataFrame({'emb_0':[12.]},index=['b'])),
                  (10,pd.DataFrame({'emb_0':[10.]},index=['a'])),
                  (11,pd.DataFrame({'emb_0':[11.]},index=['a']))]
    result=update_profiles(anchor,observations,'last')
    np.testing.assert_array_equal(result.emb_0,[11,12,200])
    pd.testing.assert_frame_equal(anchor,before)


def test_exponential_weights_are_time_based_and_do_not_impute_missing_observations():
    anchor=source()
    observations=[(10,pd.DataFrame({'emb_0':[10.]},index=['a'])),
                  (12,pd.DataFrame({'emb_0':[20.]},index=['a']))]
    result=update_profiles(anchor,observations,'ema1')
    assert result.loc['a','emb_0']==pytest.approx((0*.125+10*.25+20)/(.125+.25+1))
    np.testing.assert_array_equal(result.loc[['b','c'],'emb_0'],[100,200])


def test_future_observations_new_hosts_and_repeated_dates_are_rejected():
    frame=pd.DataFrame({'emb_0':[1.]},index=['a'])
    with pytest.raises(ValueError,match='прошлого'):update_profiles(source(),[(13,frame)],'last')
    with pytest.raises(ValueError,match='расширять'):update_profiles(source(),[(12,frame.rename(index={'a':'new'}))],'last')
    with pytest.raises(ValueError,match='повторяется'):update_profiles(source(),[(12,frame),(12,frame)],'ema3')
    with pytest.raises(ValueError,match='прошлого'):update_profiles(source(),[(14,frame)],'last',asof=13)
    assert update_profiles(source(),[(13,frame)],'last',asof=13).loc['a','emb_0']==1


def test_query_uses_observed_features_even_for_known_training_host():
    class Base:
        def embed(self,data,window,hosts):return data.loc[hosts]
    model=RecentEncoder(Base(),source())
    observed=pd.DataFrame({'emb_0':[900.,901.,902.]},index=['a','b','c'])
    assert model.embed(observed,'validation',['b']).iloc[0,0]==901
    assert model.embed(observed,'train',['b']).iloc[0,0]==100
