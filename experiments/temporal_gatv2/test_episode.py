import json
import numpy as np
import pytest
pytest.importorskip('torch')
from scipy import sparse
from .episode import Episode


def test_reload_partner_order_never_reuses_old_indices(tmp_path):
    for name in ['fingerprint_aligned','fingerprint_transfer']:
        (tmp_path/name).mkdir()
    hosts=['h0','h1','h2'];vocab=['peer_a','h0','peer_b']
    source=np.array([[1,0,0],[0,1,1],[1,0,0]])
    target=np.array([[0,0,1],[1,0,0],[0,1,1]])
    (tmp_path/'fingerprint_aligned/hosts.json').write_text(json.dumps(hosts))
    def save(order):
        (tmp_path/'fingerprint_transfer/vocabulary.json').write_text(json.dumps([vocab[i]for i in order]))
        sparse.save_npz(tmp_path/'fingerprint_transfer/source0.npz',sparse.csr_matrix(source[:,order]))
        sparse.save_npz(tmp_path/'fingerprint_aligned/future03.npz',sparse.csr_matrix(target[:,order]))
    save([0,1,2]);old=Episode(tmp_path,3,np.ones(3,bool))
    save([2,1,0]);new=Episode(tmp_path,3,np.ones(3,bool))
    for ep in [old,new]:
        valid,positive=ep.masks(np.array([0,2]))
        names=[set(ep.vocabulary[row.numpy()])for row in positive]
        assert names==[{'peer_b'},{'h0','peer_b'}]
    assert old.vocabulary.tolist()!=new.vocabulary.tolist()
    assert old.order_sha256!=new.order_sha256
