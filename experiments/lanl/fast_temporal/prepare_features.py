"""Подготовить только признаки v5, без многочасового обучения node2vec/DeepWalk."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from pipeline.data import ProcessedData
from methods.baselines.temporal import BehavioralBaseline


def main():
    work = Path(os.environ['PT_RUN_ROOT'])
    spec = importlib.util.spec_from_file_location('feature_core', work / 'hybrid_v4/train.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    data = ProcessedData(Path(os.environ['PT_LANL_DIR']) / 'processed')
    core.log('prepare_behavioral')
    behavior = BehavioralBaseline().fit(data)
    target = work / 'base_embeddings/behavioral'
    target.mkdir(parents=True, exist_ok=True)
    for window in ('train', 'validation'):
        hosts = data.cohort if window == 'train' else data.evaluable(window)
        behavior.embed(data, window, hosts).to_csv(target / f'{window}.csv')
    joblib.dump(behavior, work / 'behavioral_model.joblib')
    counts, vocabs = core.aggregate('train', data.cohort)
    models = {}
    for kind in ('peers', 'services'):
        core.log('fit_svd', kind=kind, dim=128)
        tfidf = TfidfTransformer().fit(counts[kind])
        dim = min(128, counts[kind].shape[0]-1, counts[kind].shape[1]-1)
        if dim != 128:
            raise ValueError('This fixed day02 experiment requires 128-dimensional blocks')
        svd = TruncatedSVD(dim, random_state=0).fit(normalize(tfidf.transform(counts[kind])))
        models[kind, 128] = (tfidf, svd)
    joblib.dump({'models': models, 'vocabs': vocabs, 'hosts': data.cohort}, work / 'hybrid_v4/models.joblib')
    core.log('features_completed', state='completed')


if __name__ == '__main__':
    main()
