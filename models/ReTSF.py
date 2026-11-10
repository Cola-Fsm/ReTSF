"""Computation Allocation Forecasting (CAF).

Only the observed target history is used. A complete cheap point forecast is
followed by budgeted, revisitable local residual updates. A separate small
controller predicts signed *local MSE reduction per valid point/channel*
without executing candidates. Utility labels and regression/ranking training
are handled by the experiment.
"""

from itertools import chain
import math

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.patch_len = int(getattr(configs, "caf_patch_len", 48))
        self.hidden = int(getattr(configs, "caf_hidden", getattr(configs, "caf_d_model", 64)))
        if min(self.seq_len, self.pred_len, self.channels, self.patch_len) < 1:
            raise ValueError("CAF lengths and channel count must be positive")
        self.num_segments = math.ceil(self.pred_len / self.patch_len)
        self.budget = int(getattr(configs, "caf_budget", self.num_segments))
        if self.budget == -1:
            self.budget = self.num_segments
        self.stop_threshold = float(getattr(configs, "caf_stop_threshold", 0.0))
        if getattr(configs, "caf_gain_scale", None) is not None:
            raise ValueError(
                "caf_gain_scale has been removed: CAF predicts unscaled local "
                "refinement utility in local-MSE units."
            )
        if self.budget < 0:
            raise ValueError("Budget must be nonnegative")
        # Full checkpoints must identify the controller target semantics.
        # Legacy global-gain checkpoints lack this key and fail strict load.
        self.register_buffer("controller_target_version", torch.tensor(2, dtype=torch.long))

        # Shared across target channels; computed exactly once per forecast.
        self.history_encoder = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.hidden), nn.GELU(),
        )
        self.base_linear = nn.Linear(self.seq_len, self.pred_len)
        self.base_head = nn.Linear(self.hidden, self.pred_len)
        nn.init.constant_(self.base_linear.weight, 1.0 / self.seq_len)
        nn.init.zeros_(self.base_linear.bias)
        nn.init.normal_(self.base_head.weight, std=0.01)
        nn.init.zeros_(self.base_head.bias)

        # Local/current + left/right patches, cached local/global history,
        # four positional features, and log(1 + number of previous visits).
        refiner_inputs = 2 * self.hidden + 3 * self.patch_len + 5
        self.refiner = nn.Sequential(
            nn.Linear(refiner_inputs, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.patch_len),
        )
        nn.init.normal_(self.refiner[-1].weight, std=0.01)
        nn.init.zeros_(self.refiner[-1].bias)

        # Cheap statistics only. The last layer is unconstrained: local
        # refinement utilities can be positive, zero, or negative. Its output
        # is directly in local-MSE units, with no target/output scaling.
        self.controller = nn.Sequential(
            nn.Linear(self.hidden + 18, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.hidden // 2), nn.GELU(),
            nn.Linear(self.hidden // 2, 1),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

        starts = torch.arange(self.num_segments) * self.patch_len
        lengths = (self.pred_len - starts).clamp(max=self.patch_len)
        positions = (starts + lengths / 2.0) / self.pred_len
        position_features = torch.stack((
            starts / self.pred_len,
            (starts + lengths) / self.pred_len,
            torch.sin(2 * math.pi * positions),
            torch.cos(2 * math.pi * positions),
        ), dim=-1)
        valid = torch.arange(self.patch_len)[None, :] < lengths[:, None]
        self.register_buffer("segment_lengths", lengths.long())
        self.register_buffer("segment_positions", position_features.float())
        self.register_buffer("segment_valid", valid)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version_key = prefix + "controller_target_version"
        if version_key in state_dict:
            incoming = state_dict[version_key]
            if not torch.is_tensor(incoming) or incoming.numel() != 1 or incoming.item() != 2:
                error_msgs.append(
                    f"{version_key}: CAF requires controller target version 2 "
                    "(unscaled local refinement utility); incompatible target "
                    "semantics must not reuse controller weights."
                )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forecast_parameters(self):
        return chain(self.history_encoder.parameters(), self.base_linear.parameters(),
                     self.base_head.parameters(), self.refiner.parameters())

    def controller_parameters(self):
        return self.controller.parameters()

    def freeze_forecaster(self):
        for parameter in self.forecast_parameters():
            parameter.requires_grad_(False)
        for parameter in self.controller_parameters():
            parameter.requires_grad_(True)
        return self

    def initialize(self, x):
        """Encode history once and return a complete forecast in input units.

        `forecast` always uses the dataset's input scale (typically its train
        standardization), never CAF's internal per-window normalization.
        State can be passed through arbitrary repeated refine calls.
        """
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(
                f"Expected [B,{self.seq_len},{self.channels}], got {tuple(x.shape)}"
            )
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        normalized = ((x - mean) / std).transpose(1, 2)
        history = self.history_encoder(normalized)
        base = (self.base_linear(normalized) + self.base_head(history)).transpose(1, 2)
        return {
            "forecast": base * std + mean,
            "counts": torch.zeros(x.size(0), self.num_segments,
                                  device=x.device, dtype=torch.long),
            "history": history,
            "mean": mean,
            "std": std,
        }

    def _patches(self, forecast):
        # Replicate the last point for a partial tail; the mask excludes this
        # padding from statistics, and refine never writes beyond pred_len.
        padding = self.num_segments * self.patch_len - self.pred_len
        if padding:
            forecast = torch.cat((forecast,
                                  forecast[:, -1:, :].expand(-1, padding, -1)), dim=1)
        return forecast.reshape(forecast.size(0), self.num_segments,
                                self.patch_len, self.channels)

    def refine(self, state, actions):
        """Run the shared refiner only for one selected segment per active row.

        actions is [B], with -1 meaning inactive. No candidate enumeration or
        history re-encoding occurs here. Returned state does not mutate input.
        """
        forecast = state["forecast"]
        actions = torch.as_tensor(actions, device=forecast.device, dtype=torch.long)
        if actions.shape != (forecast.size(0),):
            raise ValueError("actions must have shape [B]")
        if torch.any((actions < -1) | (actions >= self.num_segments)):
            raise ValueError("action is outside [-1, num_segments)")
        rows = (actions >= 0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return state
        chosen = actions[rows]
        patches = self._patches((forecast - state["mean"]) / state["std"])
        current = patches[rows, chosen]
        left = patches[rows, (chosen - 1).clamp_min(0)]
        right = patches[rows, (chosen + 1).clamp_max(self.num_segments - 1)]
        history = state["history"][rows]
        global_history = history.mean(dim=1, keepdim=True).expand_as(history)
        positions = self.segment_positions[chosen].to(forecast.dtype)
        visits = torch.log1p(state["counts"][rows, chosen].to(forecast.dtype))[:, None]
        metadata = torch.cat((positions, visits), dim=-1)
        features = torch.cat((
            history, global_history, current.transpose(1, 2),
            left.transpose(1, 2), right.transpose(1, 2),
            metadata[:, None, :].expand(-1, self.channels, -1),
        ), dim=-1)
        residual = self.refiner(features).transpose(1, 2)
        updated = (current + residual) * state["std"][rows] + state["mean"][rows]
        times = chosen[:, None] * self.patch_len + torch.arange(
            self.patch_len, device=forecast.device)[None, :]
        valid = times < self.pred_len
        batch_rows = rows[:, None].expand_as(times)
        new_forecast = forecast.clone()
        new_forecast[batch_rows[valid], times[valid], :] = updated[valid]
        new_counts = state["counts"].clone()
        new_counts[rows, chosen] += 1
        return {**state, "forecast": new_forecast, "counts": new_counts}

    def utility_scores(self, state):
        """Predict [B,J] signed local-MSE improvement per valid point/channel.

        Utilities have no gain_scale or horizon-length scaling. For partial
        tail segments, priority remains per-point local improvement, without
        length weighting; this differs from global-MSE-gain priority. Equal
        length segments give the same ideal action ordering under both units.

        This reads cached history and inexpensive forecast statistics only.
        It never calls refine, never reads targets, and is not an uncertainty
        or future-error estimator.
        """
        forecast = state["forecast"]
        patches = self._patches((forecast - state["mean"]) / state["std"])
        raw = self._patches(forecast)
        valid = self.segment_valid[None, :, :, None].to(forecast.dtype)
        denom = (self.segment_lengths * self.channels).to(forecast.dtype)[None, :]

        def statistics(values):
            average = (values * valid).sum(dim=(2, 3)) / denom
            variance = ((values - average[:, :, None, None]).square() * valid).sum(
                dim=(2, 3)) / denom
            return average, (variance + 1e-6).sqrt()

        average, deviation = statistics(patches)
        absolute = (patches.abs() * valid).sum(dim=(2, 3)) / denom
        last_index = (self.segment_lengths - 1)[None, :, None, None].expand(
            forecast.size(0), -1, 1, self.channels)
        last = patches.gather(2, last_index).squeeze(2)
        first = patches[:, :, 0, :]
        slope = (last - first).mean(dim=-1)
        diff_valid = self.segment_valid[:, 1:][None, :, :, None].to(forecast.dtype)
        diff_denom = ((self.segment_lengths - 1).clamp_min(1) * self.channels).to(
            forecast.dtype)[None, :]
        variation = ((patches[:, :, 1:, :] - patches[:, :, :-1, :]).abs()
                     * diff_valid).sum(dim=(2, 3)) / diff_denom
        left_last = torch.cat((first[:, :1, :], last[:, :-1, :]), dim=1)
        right_first = torch.cat((first[:, 1:, :], last[:, -1:, :]), dim=1)
        left_gap = (first - left_last).mean(dim=-1)
        right_gap = (right_first - last).mean(dim=-1)
        raw_average, raw_deviation = statistics(raw)
        counts = torch.log1p(state["counts"].to(forecast.dtype))
        total_visits = torch.log1p(state["counts"].to(forecast.dtype).mean(dim=1))
        scale = state["std"].mean(dim=(1, 2))
        global_mean = forecast.mean(dim=(1, 2))
        global_std = (forecast.var(dim=(1, 2), unbiased=False) + 1e-6).sqrt()
        scalar_features = torch.stack((
            average, deviation, absolute, slope, variation, left_gap, right_gap,
            counts, total_visits[:, None].expand_as(counts), raw_average,
            raw_deviation, scale[:, None].expand_as(counts),
            global_mean[:, None].expand_as(counts),
            global_std[:, None].expand_as(counts),
        ), dim=-1)
        positions = self.segment_positions.to(forecast.dtype)[None, :, :].expand(
            forecast.size(0), -1, -1)
        history = state["history"].mean(dim=1)[:, None, :].expand(
            -1, self.num_segments, -1)
        features = torch.cat((history, scalar_features, positions), dim=-1)
        return self.controller(features).squeeze(-1)

    def forecast_state(self, state, budget=None, policy="learned",
                       return_trace=False, stop=True):
        """Refine up to budget times; budget counts *additional* actions.

        stop_threshold is in local-MSE utility units. Only learned scheduling
        uses utility-based stopping. Uniform balances visit counts; random
        draws segments independently and permits revisits. Trace utilities
        are raw local-MSE improvements, NaN for unscored or inactive actions.
        """
        budget = self.budget if budget is None else int(budget)
        if budget < 0 or policy not in ("learned", "uniform", "random"):
            raise ValueError("Invalid budget or policy")
        current = state
        batch = state["forecast"].size(0)
        device = state["forecast"].device
        dtype = state["forecast"].dtype
        active = torch.ones(batch, device=device, dtype=torch.bool)
        steps = torch.zeros(batch, device=device, dtype=torch.long)
        if return_trace:
            action_trace = torch.full((batch, budget), -1, device=device, dtype=torch.long)
            utility_trace = torch.full((batch, budget), float("nan"), device=device, dtype=dtype)
        for step in range(budget):
            utilities = torch.full((batch,), float("nan"), device=device, dtype=dtype)
            if policy == "learned":
                scores = self.utility_scores(current)
                selected_score, actions = scores.max(dim=-1)
                utilities = selected_score
                if stop:
                    active = active & (utilities > self.stop_threshold)
            elif policy == "uniform":
                actions = current["counts"].argmin(dim=-1)
            else:
                actions = torch.randint(self.num_segments, (batch,), device=device)
            actions = actions.masked_fill(~active, -1)
            if return_trace:
                action_trace[:, step] = actions
                utility_trace[:, step] = utilities.masked_fill(~active, float("nan"))
            if not active.any():
                break
            current = self.refine(current, actions)
            steps += active.long()
        trace = None
        if return_trace:
            trace = {"actions": action_trace, "predicted_utilities": utility_trace,
                     "counts": current["counts"].clone(), "steps": steps}
        return current["forecast"], current, trace

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Timestamp features / decoder inputs / masks are deliberately ignored.
        # Future observations are never part of this model's inference inputs.
        prediction, _, _ = self.forecast_state(self.initialize(x_enc))
        return prediction

