"""Minimal entry point for the staged CAF forecasting experiment."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from exp.exp_ReTSF import Exp_ReTSF
from data_provider.caf_data import SUPPORTED_DATASETS, canonical_dataset, configure_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', type=canonical_dataset,
                        default=['ETTh1', 'ETTm1'], choices=SUPPORTED_DATASETS)
    parser.add_argument('--log_every_batches', type=int, default=0)
    parser.add_argument('--horizons', nargs='+', type=int, default=[96, 192, 336, 720])
    parser.add_argument('--dataset_root', default='./dataset', help='Root containing dataset subdirectories')
    parser.add_argument('--root_path', default=None, help='Override direct CSV folder')
    parser.add_argument('--data_path', default=None, help='Override filename for a single dataset')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--target_metric', type=str.upper, choices=['MSE', 'MAE'], default='MSE',
                        help='Validation metric for checkpoint selection; training losses remain MSE.')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--caf_patch_len', type=int, default=48)
    parser.add_argument('--caf_hidden', type=int, default=64)
    parser.add_argument('--caf_budget', type=int, default=-1)
    parser.add_argument('--caf_warmup_epochs', type=int, default=3)
    parser.add_argument('--caf_gain_epochs', type=int, default=8)
    parser.add_argument('--caf_gain_lr', type=float, default=0.001)
    parser.add_argument('--caf_rank_weight', type=float, default=0.1)
    parser.add_argument('--caf_rank_margin', type=float, default=0.01)
    parser.add_argument('--caf_rank_min_gap', type=float, default=0.005)
    parser.add_argument('--caf_stop_threshold', type=float, default=0.0)
    parser.add_argument('--caf_train_fraction', type=float, default=0.8)
    parser.add_argument('--caf_candidates', type=int, default=3)
    parser.add_argument('--reuse_forecaster_from', default=None,
                        help='Reuse only saved forecaster/refiner weights; retrain controller.')
    parser.add_argument('--output_dir', default='./results/CAF_utility')
    parser.add_argument('--run_tag', default=None)
    args = parser.parse_args()
    if args.data_path and len(args.datasets) != 1:
        parser.error('--data_path overrides require exactly one dataset.')
    args.run_tag = args.run_tag or f'utility_rank_seed{args.seed}'
    torch.set_num_threads(args.threads)
    args.model, args.features = 'CAF', 'M'
    args.use_amp = args.use_multi_gpu = False
    args.label_len = 0
    args.checkpoints = str(Path(args.output_dir) / 'checkpoints')
    results = []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        configure_dataset(args, dataset, args.dataset_root)
        for horizon in args.horizons:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            args.pred_len = horizon
            setting = f'{dataset}_L{args.seq_len}_H{horizon}_{args.run_tag}'
            print(f'\n=== {setting} ===', flush=True)
            experiment = Exp_ReTSF(args)
            experiment.train(setting)
            results.append(experiment.test(setting))
            (out / f'summary_{args.run_tag}.json').write_text(
                json.dumps(results, indent=2), encoding='utf-8')
            write_summary(out / f'summary_{args.run_tag}.csv', results)
            del experiment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



def write_summary(path, results):
    import csv

    fields = [
        'dataset', 'horizon', 'seed', 'mse', 'mae',
        'mean_steps', 'budget', 'params'
    ]

    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction='ignore'
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


if __name__ == '__main__':
    main()
