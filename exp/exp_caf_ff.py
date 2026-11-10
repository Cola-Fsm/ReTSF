"""The original three-stage CAF experiment for forecaster + refinement models."""
from importlib import import_module
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data_provider.caf_data import make_dataset
from exp.exp_ReTSF import Exp_ReTSF, controller_loss
from models.ReTSF_FF import ForecastInput


class CycleDataset(Dataset):
    """Preserve original window indices, including when wrapped in Subset.

    Official EMAformer uses the phase of the first forecast timestamp, i.e.
    ``global_start + window_index + seq_len``, rather than the history's phase.
    """
    def __init__(self, dataset, global_start, cycle, clock_offset=0):
        self.dataset = dataset
        self.global_start = int(global_start)
        self.cycle = int(cycle)
        self.clock_offset = int(clock_offset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        phase = (self.global_start + index + self.dataset.seq_len + self.clock_offset) % self.cycle
        return (*item, phase)

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)


class Exp_CAF_FF(Exp_CAF):
    def __init__(self, args):
        # Match Exp_CAF's validation without constructing and discarding an
        # additional randomly initialized CAF model before the requested FF.
        self.args = args
        if args.features != 'M' or args.use_amp or args.use_multi_gpu:
            raise ValueError('CAF FF supports M forecasting, full precision, one device.')
        if min(args.train_epochs, args.caf_gain_epochs, args.caf_candidates,
               args.batch_size, args.patience) < 1 or args.caf_warmup_epochs < 0:
            raise ValueError('Epochs/candidates/batch size/patience must be positive; warmup may be zero.')
        if min(args.caf_rank_weight, args.caf_rank_margin, args.caf_rank_min_gap) < 0:
            raise ValueError('Ranking weight/margin/minimum gap must be nonnegative.')
        self.backbone_name = getattr(args, 'ff_backbone', None) or args.model.removesuffix('FF')
        if self.backbone_name not in ('EMAformer', 'iTransformer', 'PatchMLP', 'TimeMixer'):
            raise ValueError('Choose EMAformerFF, iTransformerFF, PatchMLPFF or TimeMixerFF.')
        self.model_name = self.backbone_name + 'FF'
        self.device = torch.device(args.device)
        model_class = import_module('models.' + self.model_name).Model
        self.model = model_class(args).float().to(self.device)
        self.output_root = Path(getattr(args, 'output_dir', './results/CAF_FF'))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.seed = getattr(args, 'seed', 2025)
        self.history = []
        self._split_lengths = {}
        self.micro_batch = int(getattr(args, 'ff_micro_batch', 0))
        if self.micro_batch < 0:
            raise ValueError('ff_micro_batch must be nonnegative; zero retains the full batch.')

    def _chunks(self, inputs):
        batch = inputs.size(0)
        width = self.micro_batch or batch
        for start in range(0, batch, width):
            end = min(start + width, batch)
            marks = inputs.x_mark_enc
            piece = ForecastInput(inputs.values[start:end],
                                  None if marks is None else marks[start:end],
                                  None if inputs.cycle_index is None else inputs.cycle_index[start:end])
            yield start, end, piece

    @torch.no_grad()
    def _evaluate(self, loader, policy='learned', budget=None, stop=True):
        if not self.micro_batch or self.micro_batch >= self.args.batch_size:
            return super()._evaluate(loader, policy=policy, budget=budget, stop=stop)
        self.model.eval()
        squared = absolute = steps = elements = samples = 0
        for batch in loader:
            inputs, target = self._batch(batch)
            for start, end, piece in self._chunks(inputs):
                prediction, state, _ = self.model.forecast_state(
                    self.model.initialize(piece), policy=policy, budget=budget, stop=stop)
                error = prediction - target[start:end]
                squared += error.square().sum().item()
                absolute += error.abs().sum().item()
                elements += error.numel()
                steps += state['counts'].sum().item()
            samples += inputs.size(0)
        return {'mse': squared / elements, 'mae': absolute / elements,
                'mean_steps': steps / samples}

    def _train_forecaster(self, loader, val_loader, refine):
        if not self.micro_batch or self.micro_batch >= self.args.batch_size:
            return super()._train_forecaster(loader, val_loader, refine)
        optimizer = torch.optim.Adam(self.model.forecast_parameters(), lr=self.args.learning_rate)
        stage = 'refiner' if refine else 'base'
        epochs = self.args.train_epochs if refine else self.args.caf_warmup_epochs
        best, stale = float('inf'), 0
        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            self.model.train()
            total = count = 0
            for batch_index, batch in enumerate(loader, 1):
                inputs, target = self._batch(batch)
                batch_size = inputs.size(0)
                optimizer.zero_grad(set_to_none=True)
                # A schedule belongs to the original complete optimizer batch.
                # Draw it before slicing, including variable-length tail actions.
                schedules = [torch.randint(self.model.num_segments, (batch_size,), device=self.device)
                             for _ in range(self.model.budget)] if refine else []
                valid_totals = [self.model.segment_lengths[actions].sum() for actions in schedules]
                batch_loss = 0.0
                for start, end, piece in self._chunks(inputs):
                    truth = target[start:end]
                    state = self.model.initialize(piece)
                    fraction = (end - start) / batch_size
                    loss = fraction * F.mse_loss(state['forecast'], truth)
                    if refine:
                        for step, all_actions in enumerate(schedules):
                            actions = all_actions[start:end]
                            state = self.model.refine(state, actions)
                            # The local term is normalized by valid selected
                            # points across the original batch, not chunk size.
                            local_fraction = (self.model.segment_lengths[actions].sum().to(truth.dtype)
                                              / valid_totals[step].to(truth.dtype))
                            loss = loss + (fraction * F.mse_loss(state['forecast'], truth)
                                           + local_fraction * self._local_mse(
                                               state['forecast'], truth, actions)) / self.model.budget
                    loss.backward()
                    batch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(list(self.model.forecast_parameters()), 1.0)
                optimizer.step()  # Exactly one step per original loader batch.
                total += batch_loss * batch_size
                count += batch_size
                self._log_batch(stage, epoch, batch_index, len(loader), start_time)
            metrics = self._evaluate(val_loader, policy='uniform',
                                     budget=self.model.budget if refine else 0)
            self._record(stage, epoch, total / count, metrics['mse'],
                         time.perf_counter() - start_time, val_mae=metrics['mae'])
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
        if not self.micro_batch or self.micro_batch >= self.args.batch_size:
            return super()._train_controller(loader, val_loader)
        self.model.freeze_forecaster()
        optimizer = torch.optim.Adam(self.model.controller_parameters(), lr=self.args.caf_gain_lr)
        best, stale = float('inf'), 0
        for epoch in range(1, self.args.caf_gain_epochs + 1):
            start_time = time.perf_counter()
            self.model.eval()
            total = regression_total = ranking_total = count = 0
            for batch_index, batch in enumerate(loader, 1):
                inputs, target = self._batch(batch)
                batch_size = inputs.size(0)
                depth = random.randrange(self.model.budget + 1)
                policy = 'random' if epoch == 1 or random.random() < 0.5 else 'learned'
                random_schedules = [torch.randint(self.model.num_segments, (batch_size,), device=self.device)
                                    for _ in range(depth)] if policy == 'random' else []
                n = min(self.args.caf_candidates, self.model.num_segments)
                candidates = torch.rand(batch_size, self.model.num_segments, device=self.device).argsort(dim=1)[:, :n]
                optimizer.zero_grad(set_to_none=True)
                all_labels, all_scores = [], []
                for start, end, piece in self._chunks(inputs):
                    chosen = candidates[start:end]
                    with torch.no_grad():
                        state = self.model.initialize(piece)
                        if policy == 'random':
                            for actions in random_schedules:
                                state = self.model.refine(state, actions[start:end])
                        else:
                            _, state, _ = self.model.forecast_state(state, budget=depth,
                                                                   policy=policy, stop=False)
                        labels = []
                        for i in range(n):
                            updated = self.model.refine(state, chosen[:, i])
                            labels.append(self._local_utility(state['forecast'], updated['forecast'],
                                                              target[start:end], chosen[:, i]))
                        all_labels.append(torch.stack(labels, dim=1))
                    # Only the small controller graphs span chunks. The frozen
                    # backbone/refiner graphs never remain in GPU memory.
                    all_scores.append(self.model.utility_scores(state).gather(1, chosen))
                # Original batch normalization, including the single global
                # count of non-tied ranking pairs, remains exactly unchanged.
                loss, regression, ranking = controller_loss(
                    torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0),
                    self.args.caf_rank_weight, self.args.caf_rank_margin, self.args.caf_rank_min_gap)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.controller_parameters()), 1.0)
                optimizer.step()
                total += loss.item() * batch_size
                regression_total += regression.item() * batch_size
                ranking_total += ranking.item() * batch_size
                count += batch_size
                self._log_batch('controller', epoch, batch_index, len(loader), start_time)
            metrics = self._evaluate(val_loader)
            self._record('controller', epoch, total / count, metrics['mse'],
                         time.perf_counter() - start_time, val_mae=metrics['mae'],
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

    @torch.no_grad()
    def _latency(self, inputs, policy='learned', budget=None, stop=True):
        if not self.micro_batch or self.micro_batch >= inputs.size(0):
            return super()._latency(inputs, policy=policy, budget=budget, stop=stop)

        def predict():
            for _, _, piece in self._chunks(inputs):
                self.model.forecast_state(self.model.initialize(piece), policy=policy,
                                          budget=budget, stop=stop)

        for _ in range(5):
            predict()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        for _ in range(20):
            predict()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - start_time) * 1000 / 20

    def _dataset(self, flag):
        dataset = make_dataset(self.args, flag)
        self._split_lengths[flag] = len(dataset.data_x)
        if self.backbone_name != 'EMAformer':
            return dataset
        # Recover the loader's exact border1 without changing it or rereading
        # raw timestamps. Validation/test each contain seq_len history overlap.
        for needed in (() if flag == 'train' else ('train',) if flag == 'val' else ('train', 'val')):
            if needed not in self._split_lengths:
                self._split_lengths[needed] = len(make_dataset(self.args, needed).data_x)
        global_start = 0
        if flag == 'val':
            global_start = self._split_lengths['train'] - self.args.seq_len
        elif flag == 'test':
            global_start = (self._split_lengths['train'] + self._split_lengths['val']
                            - 2 * self.args.seq_len)
        return CycleDataset(dataset, global_start, self.args.cycle,
                            getattr(self.args, 'ema_cycle_offset', 0))

    def _batch(self, batch):
        x, y = super()._batch(batch)
        marks = None if self.args.data == 'Solar' else batch[2].float().to(
            self.device, non_blocking=True)
        phase = None
        if self.backbone_name == 'EMAformer':
            if len(batch) != 5:
                raise ValueError('EMAformerFF requires CycleDataset to supply the exact window phase.')
            phase = batch[4].to(self.device, dtype=torch.long, non_blocking=True)
        return ForecastInput(x, marks, phase), y

    @torch.no_grad()
    def profile_model(self, loader, split='test'):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        inputs, _ = self._batch(next(iter(loader)))
        cuda = self.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        elapsed = self._latency(inputs) / 1000
        profile = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'inference_time_s': elapsed,
            'gpu_memory_mb': torch.cuda.memory_allocated(self.device) / 1024 ** 2 if cuda else None,
            'peak_memory_mb': torch.cuda.max_memory_allocated(self.device) / 1024 ** 2 if cuda else None,
            'batch_size': inputs.size(0), 'device': str(self.device), 'split': split,
            'warmup_runs': 5, 'timed_runs': 20, 'model': self.model_name,
            'effective_batch_size': inputs.size(0),
            'micro_batch_size': min(self.micro_batch or inputs.size(0), inputs.size(0)),
        }
        print('=' * 80, flush=True)
        print(f'Model Profiling Summary ({split}; {self.model_name})', flush=True)
        for label, key in (('Total Params', 'total_params'), ('Trainable Params (now)', 'trainable_params'),
                           ('Batch Size', 'batch_size')):
            print(f'{label:<25}: {profile[key]:,}', flush=True)
        print(f'{"Inference Time (s)":<25}: {elapsed:.6f}', flush=True)
        print(f'{"Microbatch Size":<25}: {profile["micro_batch_size"]}', flush=True)
        for label, key in (('GPU Mem Footprint (MB)', 'gpu_memory_mb'), ('Peak Mem (MB)', 'peak_memory_mb')):
            value = profile[key]
            print(f'{label:<25}: {value:.2f}' if value is not None else f'{label:<25}: N/A (CPU)', flush=True)
        print('=' * 80, flush=True)
        (self.folder / f'profile_{split}.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
        return profile

    @torch.no_grad()
    def test(self, setting, test=0):
        self.folder = self.output_root / setting
        self._load('checkpoint.pth')
        self.model.eval()
        loader = self._loader(self._dataset('test'))
        profile = self.profile_model(loader, split='test')
        # Only the full refined model is evaluated on the complete test split.
        metrics = self._evaluate(loader)
        result = dict(dataset=self.args.data, horizon=self.args.pred_len, seed=self.seed,
                      model=self.model_name, backbone=self.backbone_name,
                      configuration_sha256=getattr(self.args, 'configuration_sha256', None),
                      controller_target='signed_local_mse_utility_v2',
                      rank_weight=self.args.caf_rank_weight, metric_split='test',
                      **metrics, budget=self.model.budget,
                      params=sum(p.numel() for p in self.model.parameters()),
                      latency_ms_per_batch=profile['inference_time_s'] * 1000,
                      latency_batch_size=profile['batch_size'], profile=profile)
        inputs, target = self._batch(next(iter(loader)))
        # A few cases suffice for later inspection; the metrics above still
        # aggregate every test point. Avoid storing GB-sized traffic batches.
        trace_count = min(8, inputs.size(0))
        inputs = ForecastInput(inputs.values[:trace_count],
                               None if inputs.x_mark_enc is None else inputs.x_mark_enc[:trace_count],
                               None if inputs.cycle_index is None else inputs.cycle_index[:trace_count])
        target = target[:trace_count]
        predictions, initials, counts = [], [], []
        traces = {}
        for _, _, piece in self._chunks(inputs):
            state = self.model.initialize(piece)
            prediction, final, trace = self.model.forecast_state(state, return_trace=True)
            predictions.append(prediction.cpu())
            initials.append(state['forecast'].cpu())
            counts.append(final['counts'].cpu())
            for key, value in trace.items():
                traces.setdefault(key, []).append(value.cpu())
        # torch.save({'history': inputs.values.cpu(), 'target': target.cpu(),
        #             'prediction': torch.cat(predictions), 'base_prediction': torch.cat(initials),
        #             'counts': torch.cat(counts),
        #             'trace': {k: torch.cat(values) for k, values in traces.items()}},
        #            self.folder / 'sample_trace.pt')
        (self.folder / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('TEST ' + json.dumps(result), flush=True)
        return result
