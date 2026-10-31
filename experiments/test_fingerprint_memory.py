import numpy as np
import pandas as pd
import scipy.sparse as sp

from experiments.fingerprint_memory import FingerprintBank, fingerprint_counts


def observation():
    records = []
    for host in ["a", "b", "c"]:
        records.extend([{"src": host, "dst": "peer_" + host, "proto": "6", "dst_port": "443"}] * 30)
    return pd.DataFrame(records)


def test_self_flow_cannot_be_used_as_a_host_identifier():
    flows = pd.concat([observation(), pd.DataFrame([{"src": "a", "dst": "a", "proto": "6", "dst_port": "443"}])])
    counts = fingerprint_counts(flows, ["a", "b", "c"])
    assert ("a", "o:a") not in counts["counts"]["peers"].index
    assert ("a", "i:a") not in counts["counts"]["peers"].index
    np.testing.assert_array_equal(counts["events"], [30, 30, 30])


def test_memory_predictions_follow_traffic_not_query_row_or_host_id():
    counts = fingerprint_counts(observation(), ["a", "b", "c"])
    bank = FingerprintBank(latent_dim=4).fit([counts, counts], ["a", "b", "c"])
    x = bank.transform(counts)
    vectors, labels = bank.posterior(x)
    np.testing.assert_array_equal(labels, [0, 1, 2])
    order = [2, 0, 1]
    shuffled, shuffled_labels = bank.posterior(x[order])
    np.testing.assert_allclose(shuffled, vectors[order], atol=1e-6)
    np.testing.assert_array_equal(shuffled_labels, order)
    unknown, unknown_label = bank.posterior(sp.csr_matrix((1, x.shape[1]), dtype=np.float32))
    np.testing.assert_array_equal(unknown, np.zeros_like(unknown))
    np.testing.assert_array_equal(unknown_label, [-1])


def test_rare_training_tokens_survive_and_future_tokens_do_not_expand_vocabulary():
    counts = fingerprint_counts(observation(), ["a", "b", "c"])
    bank = FingerprintBank(latent_dim=4).fit([counts], ["a", "b", "c"])
    assert "o:peer_a" in bank.vocab["peers"]
    original = list(bank.vocab["peers"])
    unseen = observation().replace({"peer_a": "new_future_peer"})
    x = bank.transform(fingerprint_counts(unseen, ["a", "b", "c"]))
    assert list(bank.vocab["peers"]) == original
    assert x[0].nnz == 0


def test_temporal_stability_filters_one_day_noise_using_only_training_views():
    regular = observation()
    noisy = pd.concat([regular, pd.DataFrame([{"src": "a", "dst": "transient", "proto": "6", "dst_port": "443"}] * 30)])
    counts = [fingerprint_counts(f, ["a", "b", "c"]) for f in [noisy, regular]]
    bank = FingerprintBank(latent_dim=4).fit(counts, ["a", "b", "c"]).set_stability(2)
    column = bank.vocab["peers"].get_loc("o:transient")
    assert bank.memory[0, column] == 0
    _, labels = bank.posterior(bank.transform(counts[1]))
    np.testing.assert_array_equal(labels, [0, 1, 2])


def test_batched_link_predictions_equal_full_matrix_predictions():
    from sklearn.linear_model import LogisticRegression
    from evaluation.future_links import _predict_probability_batches
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.standard_normal((137, 7)))
    labels = (frame[0] + frame[1] > 0).astype(int)
    model = LogisticRegression().fit(frame, labels)
    actual = _predict_probability_batches(model, lambda f: f, frame, batch=17)
    np.testing.assert_allclose(actual, model.predict_proba(frame)[:, 1], atol=1e-12)
