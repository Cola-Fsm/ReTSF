"""Existing point forecasters with the unchanged CAF refinement algorithm."""
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from itertools import chain
from typing import Optional

import torch

from models.ReTSF import Model as ReTSFModel


@dataclass(frozen=True)
class ForecastInput:
    """One batch's observations and known calendar/phase inputs.

    Keeping these inputs together avoids shared, mutable timestamp state when
    the inherited experiment alternates training, validation and profiling.
    ``size`` is the only tensor method needed by its training loops.
    """
    values: torch.Tensor
    x_mark_enc: Optional[torch.Tensor] = None
    cycle_index: Optional[torch.Tensor] = None

    def size(self, dim=None):
        return self.values.size() if dim is None else self.values.size(dim)


class Model(CAFModel):
    def __init__(self, configs, backbone_name=None):
        super().__init__(configs)
        self.backbone_name = backbone_name or getattr(configs, 'ff_backbone', None)
        if self.backbone_name not in ('iTransformer', 'PatchMLP', 'TimeMixer'):
            raise ValueError('FF requires iTransformer, PatchMLP or TimeMixer.')
        # The original CAF prediction heads are replaced, not added to the
        # backbone forecast. All update and scheduling modules are inherited.
        del self.base_linear
        del self.base_head
        backbone_config = deepcopy(configs)
        # These local loaders return five continuous time features for minutes;
        # the original embedding classes call that frequency "t".
        frequency = str(getattr(backbone_config, 'freq', 'h')).lower()
        if 'min' in frequency:
            backbone_config.freq = 't'
        self.backbone = import_module('models.' + self.backbone_name).Model(backbone_config)

    def forecast_parameters(self):
        return chain(self.backbone.parameters(), self.history_encoder.parameters(),
                     self.refiner.parameters())

    def _initial_prediction(self, inputs):
        x = inputs.values
        decoder = x.new_zeros(x.size(0), self.pred_len, self.channels)

        forecast = self.backbone(x, inputs.x_mark_enc, decoder, None)
        if isinstance(forecast, tuple):
            forecast = forecast[0]
        if not torch.is_tensor(forecast) or forecast.shape != (x.size(0), self.pred_len, self.channels):
            shape = tuple(forecast.shape) if torch.is_tensor(forecast) else type(forecast).__name__
            raise ValueError(f'{self.backbone_name} must return [B,{self.pred_len},'
                             f'{self.channels}], got {shape}.')
        return forecast

    def initialize(self, x):
        inputs = x if isinstance(x, ForecastInput) else ForecastInput(x)
        values = inputs.values
        if values.ndim != 3 or values.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f'Expected [B,{self.seq_len},{self.channels}], '
                             f'got {tuple(values.shape)}.')
        mean = values.mean(dim=1, keepdim=True).detach()
        std = (values.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        history = self.history_encoder(((values - mean) / std).transpose(1, 2))
        # The original backbone handles its own normalization. Feed its original
        # dataset-scale input and retain its dataset-scale output unchanged.
        prediction = self._initial_prediction(inputs)
        return {'forecast': prediction,
                'counts': torch.zeros(values.size(0), self.num_segments,
                                      device=values.device, dtype=torch.long),
                'history': history, 'mean': mean, 'std': std}

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, cycle_index=None):
        inputs = (x_enc if isinstance(x_enc, ForecastInput)
                  else ForecastInput(x_enc, x_mark_enc, cycle_index))
        prediction, _, _ = self.forecast_state(self.initialize(inputs))
        return prediction
