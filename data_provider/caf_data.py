"""Shared configuration and standard splits for CAF forecasting datasets."""
import csv
from pathlib import Path

from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
)


SUPPORTED_DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELC', 'Weather', 'Traffic')
_ALIASES = {name.lower(): name for name in SUPPORTED_DATASETS}
_ALIASES.update(electricity='ELC', electric='ELC')
_CUSTOM = {
    'ELC': ('electricity', 'electricity.csv', 'h'),
    'Weather': ('weather', 'weather.csv', '10min'),
    'Traffic': ('traffic', 'traffic.csv', 'h'),
}


def canonical_dataset(name):
    """Return a supported canonical name, accepting case and electricity aliases."""
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f'Unsupported CAF dataset {name!r}; choose {SUPPORTED_DATASETS}.')
    return _ALIASES[key]


def _header(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        columns = next(csv.reader(handle), [])
    if len(columns) < 2 or columns[0] != 'date' or len(set(columns)) != len(columns):
        raise ValueError(f'{path}: expected unique columns with date first and at least one channel.')
    return columns





def configure_dataset(args, dataset, dataset_root='./dataset'):
    """Resolve one dataset and mutate args, returning the same namespace.

    dataset_root contains ETT-small (or ETT), electricity, weather, traffic.
    An initial args.root_path overrides the direct data folder for every dataset;
    an initial args.data_path overrides its filename. Explicit overrides are
    captured once, so this namespace can be reused across datasets and trials
    without treating a previously resolved folder as the next override.
    """
    dataset = canonical_dataset(dataset)
    if not hasattr(args, '_caf_explicit_root_path'):
        args._caf_explicit_root_path = getattr(args, 'root_path', None)
        args._caf_explicit_data_path = getattr(args, 'data_path', None)
    root_override = args._caf_explicit_root_path
    filename_override = args._caf_explicit_data_path
    dataset_root = Path(dataset_root)
    if dataset.startswith('ETT'):
        filename = dataset + '.csv'
        folder = dataset_root / 'ETT-small'
        if not (folder / filename).is_file() and (dataset_root / 'ETT' / filename).is_file():
            folder = dataset_root / 'ETT'
        freq = '15min' if dataset.startswith('ETTm') else 'h'
    else:
        subdirectory, filename, freq = _CUSTOM[dataset]
        folder = dataset_root / subdirectory
    if root_override:
        folder = Path(root_override)
    if filename_override:
        filename = str(filename_override)
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f'{dataset} data not found: {path}; set --dataset_root or --root_path.')
    columns = _header(path)
    args.data = dataset
    args.root_path = str(folder)
    args.data_path = filename
    args.enc_in = len(columns) - 1
    args.dec_in = args.c_out = args.enc_in
    args.features = 'M'
    # Custom loaders move this column to the end even under M forecasting.
    args.target = 'OT' if columns is None or 'OT' in columns else columns[-1]
    args.freq = freq
    return args


def make_dataset(args, flag):
    """Load standard ETT 12/4/4-month or Custom 70/10/20% splits.

    Existing loaders fit StandardScaler only on the official training split. Calendar
    features are returned for compatibility; CAF never reads them.
    This also accepts already configured legacy args.root_path/data_path.
    """
    dataset = canonical_dataset(args.data)
    if flag not in ('train', 'val', 'test'):
        raise ValueError('Dataset flag must be train, val, or test.')
    if getattr(args, 'features', 'M') != 'M':
        raise ValueError('CAF datasets currently support multivariate M forecasting.')
    path = Path(args.root_path) / args.data_path
    columns =  _header(path)
    channels = len(columns) - 1
    if getattr(args, 'enc_in', channels) != channels:
        raise ValueError(f'{dataset} has {channels} channels, but enc_in={args.enc_in}.')
    elif dataset.startswith('ETTh'):
        cls, freq = Dataset_ETT_hour, 'h'
    elif dataset.startswith('ETTm'):
        cls, freq = Dataset_ETT_minute, '15min'
    else:
        cls, freq = Dataset_Custom, _CUSTOM[dataset][2]
    target = getattr(args, 'target', 'OT')
    if columns is not None and target not in columns:
        target = 'OT' if 'OT' in columns else columns[-1]
    return cls(str(args.root_path), data_path=str(args.data_path), flag=flag,
               size=[args.seq_len, 0, args.pred_len], features='M', target=target,
               timeenc=1, freq=freq)
