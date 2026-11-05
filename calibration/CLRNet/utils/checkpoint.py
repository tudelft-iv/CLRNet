"""Model settings stored inside checkpoints.

Every checkpoint stores the settings it was trained with under its 'config' key. The evaluation scripts
take the model and preprocessing settings from there, so checkpoint files can be named freely.
Older checkpoints without a stored config fall back to the settings encoded in their filename.
"""
import os
import re

import torch

# Settings that define how a checkpoint has to be run.
MODEL_KEYS = ('input_size', 'spherical_projection', 'use_reflectance_LIDAR', 'use_reflectance_RADAR',
              'use_velocity_RADAR', 'use_time_RADAR', 'use_depth', 'use_depth_input', 'multiple_input',
              'max_depth', 'max_r', 'max_t', 'dropout')

# Settings that select the test data (miscalibration range, depth input); taken from the first
# checkpoint of a cascade only, because the test set is built once.
DATASET_KEYS = ('max_r', 'max_t', 'use_depth_input')


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def read_checkpoint_config(path):
    """Settings stored in a checkpoint, read without loading the weights into memory."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError:  # files in the legacy (non-zip) format cannot be memory-mapped
        ckpt = load_checkpoint(path)
    config = dict(ckpt.get('config') or {})
    return config if config else _config_from_legacy_name(path)


def apply_checkpoint_config(config, ckpt_config, stage=False):
    """Copy the model settings of a checkpoint into `config`.

    With stage=True (a later model of a cascade) the dataset-level settings are left unchanged.
    Settings under `eval_overrides` in `config` are applied last, for evaluations that deliberately
    run a model differently from how it was trained (e.g. the cross-domain experiment).
    """
    for key in MODEL_KEYS:
        if stage and key in DATASET_KEYS:
            continue
        if key in ckpt_config:
            config[key] = ckpt_config[key]
    config.update(config.get('eval_overrides') or {})
    return config


def mode_from_config(ckpt_config, default):
    """Evaluation mode for a checkpoint that names its model and dataset."""
    model, dataset = ckpt_config.get('model'), ckpt_config.get('dataset')
    if model == 'CLRNet':
        return 'CLR'
    if model == 'CRNet':
        return 'CRDual' if dataset == 'Dual-Radar' else 'CR'
    if model == 'CLNet':
        return 'CL'
    return default


def _config_from_legacy_name(path):
    name = os.path.basename(path)
    config = {}
    m = re.search(r'_r(?P<r>[\d.]+)_t(?P<t>[\d.]+)_e', name)
    if m:
        config['max_r'], config['max_t'] = float(m.group('r')), float(m.group('t'))
    else:
        m = re.search(r'_r(?P<r>[\d.]+)\.(?:pth|tar)$', name)
        if m:
            config['max_r'] = float(m.group('r'))
    m = re.search(r'_mi(?P<mi>\d+)', name)
    if m:
        config['multiple_input'] = int(m.group('mi'))
    m = re.search(r'Rchannels(?P<R>True|False)_depth(?P<depth>True|False)_sph(?P<sph>True|False)', name)
    if m:
        channels = m.group('R') == 'True'
        config.update(use_reflectance_RADAR=channels, use_velocity_RADAR=channels, use_time_RADAR=channels,
                      use_depth=m.group('depth') == 'True', spherical_projection=m.group('sph') == 'True')
    return config
