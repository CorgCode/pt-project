"""Train graph and matched ablations using source02 and days10–12 only."""
import argparse
import json
from pathlib import Path
import numpy as np
from .model import load_observation, train, infer, save_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    names = ['day02', 'day10', 'day11', 'day12']
    expected = {'day02': (126000, 129600), 'day10': (822600, 826200),
                'day11': (909000, 912600), 'day12': (995400, 999000)}
    metadata = {name: json.loads((args.windows / name / 'manifest.json').read_text()) for name in names}
    if len({m['hosts_sha256'] for m in metadata.values()}) != 1:
        raise ValueError('training windows do not have same host order')
    for name, span in expected.items():
        if (metadata[name]['start'], metadata[name]['end']) != span:
            raise ValueError('unexpected training window')
    (args.output / 'protocol.json').write_text(json.dumps({'train': names,
        'forecast': ['10→11', '10→12', '11→12'], 'development': 'day13',
        'benchmark': ['day14', 'day15', 'day16'], 'test_labels_in_training': False,
        'base': 'Similarity Graph Recent: source and adapter already fitted through day12',
        'same_hosts_are_supervision_not_encoder_features': True,
        'windows': metadata}, indent=2))
    for mode in ['gnn', 'no_graph', 'no_bytes']:
        obs = {name: load_observation(args.windows, name, include_bytes=mode != 'no_bytes') for name in names}
        state = train(obs, mode=mode, steps=args.steps, seed=args.seed)
        save_state(state, args.output / f'{mode}.pt')
        np.save(args.output / f'{mode}_source.npy', infer(state, obs['day02']))
        # day13 is input to frozen inference only, not part of normalization or training.
        dev = load_observation(args.windows, 'day13', include_bytes=mode != 'no_bytes')
        np.save(args.output / f'{mode}_day13.npy', infer(state, dev))
        print('saved', mode, flush=True)


if __name__ == '__main__':
    main()
