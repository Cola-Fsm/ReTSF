"""Train a point forecaster, then learn signed local refinement utility and ranking."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data_provider.caf_data import make_dataset
from models.ReTSF import Model


def controller_loss(scores, labels, rank_weight=0.1, rank_margin=0.01,
                    rank_min_gap=0.005):
    """Calibrated utility regression plus within-state, non-tied pair ranking.

    Both tensors are [batch, candidates]. All quantities use raw local-MSE
    units. Regression anchors zero for stopping; ranking is offset invariant.
    Labels never carry gradients. One candidate/no valid pair gives zero rank loss.
    """
    labels = labels.detach()
    regression = F.mse_loss(scores, labels)
    gap = labels[:, :, None] - labels[:, None, :]
    difference = scores[:, :, None] - scores[:, None, :]
    # Each strict preference appears once (the reverse has a negative gap).
    valid = gap > rank_min_gap
    penalties = F.relu(rank_margin - difference)
    ranking = (penalties * valid).sum() / valid.sum().clamp_min(1)
    return regression + rank_weight * ranking, regression, ranking


class Exp_ReTSF:
    def __init__(self, args):
        self.args = args
        if args.features != 'M' or args.use_amp or args.use_multi_gpu:
            raise ValueError('ReTSF currently supports M forecasting, full precision, one device.')
        if min(args.train_epochs, args.caf_gain_epochs, args.caf_candidates,
               args.batch_size, args.patience) < 1 or args.caf_warmup_epochs < 0:
            raise ValueError('Epochs/candidates/batch size/patience must be positive; warmup may be zero.')
        if min(args.caf_rank_weight, args.caf_rank_margin, args.caf_rank_min_gap) < 0:
            raise ValueError('Ranking weight/margin/minimum gap must be nonnegative.')
        self.device = torch.device(args.device)
        self.model = Model(args).float().to(self.device)
        self.output_root = Path(getattr(args, 'output_dir', './results/CAF_utility'))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.seed = getattr(args, 'seed', 2025)
        self.history = []

    def _dataset(self, flag):
        return make_dataset(self.args, flag)

    def _loader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                          num_workers=self.args.num_workers, drop_last=False,
                          pin_memory=self.device.type == 'cuda')

    def _batch(self, batch):
        return (batch[0].float().to(self.device, non_blocking=True),
                batch[1][:, -self.args.pred_len:, :].float().to(
                    self.device, non_blocking=True))

    @torch.no_grad()
    def _evaluate(self, loader, policy='learned', budget=None, stop=True):
        self.model.eval()
        squared = absolute = steps = elements = samples = 0
        for batch in loader:
            x, y = self._batch(batch)
            state = self.model.initialize(x)
            pred, state, _ = self.model.forecast_state(
                state, budget=budget, policy=policy, stop=stop)
            error = pred - y
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            elements += error.numel()
            samples += x.size(0)
            steps += state['counts'].sum().item()
        return {'mse': squared / elements, 'mae': absolute / elements,
                'mean_steps': steps / samples}

    def _record(self, stage, epoch, loss, val, elapsed, **diagnostics):
        row = dict(stage=stage, epoch=epoch, train_loss=loss,
                   val_mse=val, elapsed_seconds=elapsed, **diagnostics)
        self.history.append(row)
        mae_text = f'val_mae={diagnostics["val_mae"]:.6f} ' if diagnostics.get('val_mae') is not None else ''
        print(f'{stage} epoch {epoch}: train={loss:.6f} val_mse={val:.6f} {mae_text}'
              f'time={elapsed:.1f}s', flush=True)
        (self.folder / 'training.json').write_text(
            json.dumps(self.history, indent=2), encoding='utf-8')
        callback = getattr(self, 'epoch_callback', None)
        if callback is not None:
            callback(stage, epoch, val)

    def _log_batch(self, stage, epoch, batch_index, batches, start):
        interval = getattr(self.args, 'log_every_batches', 0)
        if interval > 0 and (batch_index % interval == 0 or batch_index == batches):
            print(f'{stage} epoch {epoch}: batch {batch_index}/{batches} '
                  f'({100 * batch_index / batches:.1f}%), '
                  f'elapsed={time.perf_counter() - start:.1f}s', flush=True)

    def _save(self, name):
        torch.save({'state_dict': self.model.state_dict(),
                    'controller_target': 'signed_local'}, self.folder / name)

    def _load(self, name):
        checkpoint = torch.load(self.folder / name, map_location=self.device, weights_only=True)
        if checkpoint.get('controller_target') != 'signed_local':
            raise ValueError('Legacy global-gain checkpoint cannot be used as a local-utility controller. '
                             'Use --reuse_forecaster_from to import only its forecaster.')
        self.model.load_state_dict(checkpoint['state_dict'])

    def _reuse_forecaster(self, source_root):
        source = Path(source_root)
        if not (source / 'config.json').exists():
            source = source / (f'{self.args.data}_L{self.args.seq_len}_'
                               f'H{self.args.pred_len}_seed{self.seed}')
        config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        for key in ('data', 'data_path', 'seq_len', 'pred_len', 'enc_in', 'caf_patch_len',
                    'caf_hidden', 'caf_train_fraction'):
            if config.get(key) != getattr(self.args, key):
                raise ValueError(f'Incompatible source forecaster configuration: {key}')
        path = source / 'refiner.pth'
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        weights = checkpoint.get('state_dict', checkpoint)
        current = self.model.state_dict()
        required = set(current) - {'controller_target_version'}
        if not required.issubset(weights):
            raise ValueError('Source checkpoint is missing required forecasting weights.')
        # Explicit legacy import: retain the new format marker and freshly initialized
        # controller. Never reinterpret an old 100*global-gain head as local utility.
        for key, value in weights.items():
            if key.startswith('controller.') or key == 'controller_target_version':
                continue
            if key not in current or value.shape != current[key].shape:
                raise ValueError(f'Incompatible source forecasting tensor: {key}')
            current[key] = value
        self.model.load_state_dict(current)
        provenance = {'source_refiner': str(path.resolve()),
                      'source_refiner_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'source_config': str((source / 'config.json').resolve()),
                      'source_metrics': str((source / 'metrics.json').resolve()),
                      'imported_controller': False}
        if (source / 'metrics.json').exists():
            baseline = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
            (self.folder / 'baseline_metrics.json').write_text(
                json.dumps(baseline, indent=2), encoding='utf-8')
        (self.folder / 'forecaster_source.json').write_text(
            json.dumps(provenance, indent=2), encoding='utf-8')
        self._save('refiner.pth')
        print(f'Reused forecaster only: {path}', flush=True)

    def _local_mse(self, forecast, target, actions):
        offsets = torch.arange(self.model.patch_len, device=self.device)
        indices = actions[:, None] * self.model.patch_len + offsets[None, :]
        valid = indices < self.args.pred_len
        index = indices.clamp(max=self.args.pred_len - 1)
        index = index[:, :, None].expand(-1, -1, forecast.size(-1))
        error = forecast.gather(1, index) - target.gather(1, index)
        return (error.square() * valid[:, :, None]).sum() / (
            valid.sum() * forecast.size(-1))

    def _local_utility(self, before, after, target, actions):
        """Per-sample signed improvement per valid selected time point/channel.

        Compute squared-error differences directly inside the selected segment;
        no horizon averaging, label scaling, or subtraction of two global means.
        """
        offsets = torch.arange(self.model.patch_len, device=before.device)
        indices = actions[:, None] * self.model.patch_len + offsets[None, :]
        valid = indices < before.size(1)
        index = indices.clamp(max=before.size(1) - 1)
        index = index[:, :, None].expand(-1, -1, before.size(-1))
        truth = target.gather(1, index)
        differences = ((before.gather(1, index) - truth).square()
                       - (after.gather(1, index) - truth).square())
        return (differences * valid[:, :, None]).sum(dim=(1, 2)) / (
            valid.sum(dim=1) * before.size(-1))

    def _train_forecaster(self, loader, val_loader, refine):
        optimizer = torch.optim.Adam(self.model.forecast_parameters(),
                                     lr=self.args.learning_rate)
        stage = 'refiner' if refine else 'base'
        epochs = self.args.train_epochs if refine else self.args.caf_warmup_epochs
        best, stale = float('inf'), 0
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            self.model.train()
            total = count = 0
            for batch_index, batch in enumerate(loader, 1):
                x, y = self._batch(batch)
                optimizer.zero_grad(set_to_none=True)
                state = self.model.initialize(x)
                loss = F.mse_loss(state['forecast'], y)
                if refine:
                    accumulated = 0
                    # Random schedules with replacement train repeated visits.
                    for _ in range(self.model.budget):
                        actions = torch.randint(self.model.num_segments, (x.size(0),),
                                                device=self.device)
                        state = self.model.refine(state, actions)
                        accumulated = accumulated + F.mse_loss(state['forecast'], y)
                        accumulated = accumulated + self._local_mse(
                            state['forecast'], y, actions)
                    loss = loss + accumulated / max(self.model.budget, 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.forecast_parameters()), 1.0)
                optimizer.step()
                total += loss.item() * x.size(0)
                count += x.size(0)
                self._log_batch(stage, epoch, batch_index, len(loader), start)
            metrics = self._evaluate(val_loader, policy='uniform',
                                     budget=self.model.budget if refine else 0)
            val = metrics['mse']
            self._record(stage, epoch, total / count, val, time.perf_counter() - start,
                         val_mae=metrics.get('mae'))
            selection = metrics[getattr(self.args, 'target_metric', 'MSE').lower()]
            if selection < best:
                best, stale = selection, 0
                self._save(stage + '.pth')
            else:
                stale += 1
                if stale >= self.args.patience:
                    break
        if epochs:
            self._load(stage + '.pth')

    def _train_controller(self, loader, val_loader):
        self.model.freeze_forecaster()
        optimizer = torch.optim.Adam(self.model.controller_parameters(),
                                     lr=self.args.caf_gain_lr)
        best, stale = float('inf'), 0
        for epoch in range(1, self.args.caf_gain_epochs + 1):
            start = time.perf_counter()
            self.model.eval()  # Frozen forecaster must be deterministic for gain labels.
            total = regression_total = ranking_total = count = 0
            for batch_index, batch in enumerate(loader, 1):
                x, y = self._batch(batch)
                with torch.no_grad():
                    state = self.model.initialize(x)
                    depth = random.randrange(self.model.budget + 1)
                    policy = 'random' if epoch == 1 or random.random() < 0.5 else 'learned'
                    _, state, _ = self.model.forecast_state(
                        state, budget=depth, policy=policy, stop=False)
                    n = min(self.args.caf_candidates, self.model.num_segments)
                    # All candidates share exactly the same computation state.
                    candidates = torch.rand(x.size(0), self.model.num_segments,
                                            device=self.device).argsort(dim=1)[:, :n]
                    labels = []
                    for i in range(n):
                        updated = self.model.refine(state, candidates[:, i])
                        labels.append(self._local_utility(
                            state['forecast'], updated['forecast'], y, candidates[:, i]))
                    labels = torch.stack(labels, dim=1)
                optimizer.zero_grad(set_to_none=True)
                scores = self.model.utility_scores(state).gather(1, candidates)
                loss, regression, ranking = controller_loss(
                    scores, labels, self.args.caf_rank_weight,
                    self.args.caf_rank_margin, self.args.caf_rank_min_gap)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.controller_parameters()), 1.0)
                optimizer.step()
                total += loss.item() * x.size(0)
                regression_total += regression.item() * x.size(0)
                ranking_total += ranking.item() * x.size(0)
                count += x.size(0)
                self._log_batch('controller', epoch, batch_index, len(loader), start)
            # Validation targets select checkpoints; they never create training labels.
            metrics = self._evaluate(val_loader)
            val = metrics['mse']
            self._record('controller', epoch, total / count, val,
                         time.perf_counter() - start,
                         val_mae=metrics.get('mae'),
                         utility_regression_loss=regression_total / count,
                         utility_ranking_loss=ranking_total / count)
            selection = metrics[getattr(self.args, 'target_metric', 'MSE').lower()]
            if selection < best:
                best, stale = selection, 0
                self._save('checkpoint.pth')
            else:
                stale += 1
                if stale >= self.args.patience:
                    break
        self._load('checkpoint.pth')

    def train(self, setting):
        self.folder = self.output_root / setting
        self.folder.mkdir(parents=True, exist_ok=True)
        train = self._dataset('train')
        val_loader = self._loader(self._dataset('val'))
        fraction = self.args.caf_train_fraction
        if not 0 < fraction < 1 or self.model.budget < 1:
            raise ValueError('Use a train fraction in (0,1) and at least one refinement.')
        cutoff = int(len(train.data_x) * fraction)
        # No future target timestamps overlap between forecaster and gain-label subsets.
        forecast_end = cutoff - self.args.seq_len - self.args.pred_len + 1
        controller_start = cutoff - self.args.seq_len
        if forecast_end <= 0 or controller_start >= len(train):
            raise ValueError('Insufficient train data for the chronological internal split.')
        forecast_loader = self._loader(Subset(train, range(forecast_end)), shuffle=True)
        gain_loader = self._loader(Subset(train, range(controller_start, len(train))), shuffle=True)
        metadata = {key: str(value) if isinstance(value, (Path, torch.device)) else value
                    for key, value in vars(self.args).items()}
        metadata.update(forecaster_windows=forecast_end,
                        controller_windows=len(train) - controller_start,
                        internal_target_boundary=cutoff,
                        train_total_timesteps=len(train.data_x),
                        refinement_budget=self.model.budget,
                        controller_target='signed_local',
                        controller_output_units='MSE improvement per selected valid time point/channel',
                        device_name=torch.cuda.get_device_name(self.device)
                        if self.device.type == 'cuda' else 'CPU')
        (self.folder / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'forecaster windows={forecast_end}, controller windows='
              f'{len(train) - controller_start}, segments={self.model.num_segments}, '
              f'budget={self.model.budget}', flush=True)
        source = getattr(self.args, 'reuse_forecaster_from', None)
        if source:
            self._reuse_forecaster(source)
        else:
            self._train_forecaster(forecast_loader, val_loader, refine=False)
            self._train_forecaster(forecast_loader, val_loader, refine=True)
        self._train_controller(gain_loader, val_loader)
        # Profile every completed experiment/trial using the selected checkpoint.
        # Validation inputs suffice; test data is still reserved for final reporting.
        self.profile_model(val_loader, split='validation')
        return self.model

    @torch.no_grad()
    def test(self, setting, test=0):
        self.folder = self.output_root / setting
        self._load('checkpoint.pth')
        self.model.eval()
        # Instantiate the test split only after training and checkpoint selection.
        loader = self._loader(self._dataset('test'))

        # Keep profiling for model size, GPU memory and inference timing.
        profile = self.profile_model(loader, split='test')

        # Evaluate ONLY the actual ReTSF policy (adaptive, learned selection).
        # Do not execute or report base, uniform or learned-fixed comparisons.
        metrics = self._evaluate(loader)
        result = dict(
            dataset=self.args.data,
            horizon=self.args.pred_len,
            seed=self.seed,
            controller_target='signed_local',
            rank_weight=self.args.caf_rank_weight,
            **metrics,   # mse, mae, mean_steps
            budget=self.model.budget,
            params=sum(p.numel() for p in self.model.parameters()),
            profile=profile,
        )

        # No separate latency_ms_per_batch, base/uniform/fixed metrics,
        # comparison runtimes, or unused trace rollout is saved.
        (self.folder / 'metrics.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
        print('TEST ' + json.dumps(result), flush=True)
        return result

    @torch.no_grad()
    def profile_model(self, loader, split='test'):
        self.model.eval()
        # Training has ended; discard gradient buffers before profiling inference.
        self.model.zero_grad(set_to_none=True)
        batch = next(iter(loader))
        x = batch[0].float().to(self.device, non_blocking=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        # Includes initialization, utility selection, refinement and early stopping.
        # Match the existing latency metric: 5 warmup runs, then 20 timed runs.
        inference_time = self._latency(x) / 1000
        cuda = self.device.type == 'cuda'
        profile = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'inference_time_s': inference_time,
            'gpu_memory_mb': torch.cuda.memory_allocated(self.device) / 1024 ** 2 if cuda else None,
            'peak_memory_mb': torch.cuda.max_memory_allocated(self.device) / 1024 ** 2 if cuda else None,
            'batch_size': x.size(0), 'device': str(self.device), 'split': split,
            'warmup_runs': 5, 'timed_runs': 20,
        }
        print('=' * 80, flush=True)
        print(f'Model Profiling Summary ({split})', flush=True)
        print(f'{"Total Params":<25}: {profile["total_params"]:,}', flush=True)
        print(f'{"Trainable Params (now)":<25}: {profile["trainable_params"]:,}', flush=True)
        print(f'{"Batch Size":<25}: {profile["batch_size"]}', flush=True)
        print(f'{"Inference Time (s)":<25}: {inference_time:.6f}', flush=True)
        for label, key in (('GPU Mem Footprint (MB)', 'gpu_memory_mb'),
                           ('Peak Mem (MB)', 'peak_memory_mb')):
            value = profile[key]
            print(f'{label:<25}: {value:.2f}' if value is not None else f'{label:<25}: N/A (CPU)',
                  flush=True)
        print('=' * 80, flush=True)
        (self.folder / f'profile_{split}.json').write_text(
            json.dumps(profile, indent=2), encoding='utf-8')
        return profile

    @torch.no_grad()
    def _latency(self, x, policy='learned', budget=None, stop=True):
        def predict():
            return self.model.forecast_state(self.model.initialize(x),
                                             policy=policy, budget=budget, stop=stop)
        for _ in range(5):
            predict()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for _ in range(20):
            predict()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - start) * 1000 / 20
