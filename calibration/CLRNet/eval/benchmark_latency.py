"""Per-frame latency benchmark (CUDA-synchronised) for the configurations reported in the paper.

Operating points reported:

    1. backbone only        -- depth branch disabled            (the real-time configuration)
    2. backbone + ViT-S     -- DepthAnythingV2 small
    3. backbone + ViT-L     -- DepthAnythingV2 large            (the published configuration)

Usage (needs an idle GPU -- do not run beside a training job):

    docker exec -e CLR_MODE=CR clr_calibration python3 \
        /home/appuser/src/calibration/CLRNet/eval/benchmark_latency.py

Environment:
    CLR_CONFIG   config to size the model from  (default: VOD_CR_config.yaml)
    CLR_MODE     CR (default) or CLR
    CLR_ITERS    timed iterations per operating point (default 200)
    CLR_WARMUP   warm-up iterations, excluded from statistics (default 50)
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from CLRNet.utils.utils import read_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2')))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, iters, warmup, label):
    """Time `fn` with explicit synchronisation on both sides of every call."""
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(samples)
    # Report the median as well: the mean of a latency distribution is dragged by occasional
    # scheduler stalls, and the median is the more honest "typical frame" figure.
    print(f'{label:<34} mean {a.mean():7.1f} ms   median {np.median(a):7.1f} ms   '
          f'std {a.std():6.1f} ms   -> {1000.0 / a.mean():5.1f} fps (mean)')
    return a.mean()


def main():
    cfg_path = os.environ.get(
        'CLR_CONFIG', '/home/appuser/src/calibration/CLRNet/configs/VOD_CR_config.yaml')
    mode = os.environ.get('CLR_MODE', 'CR').upper()
    iters = int(os.environ.get('CLR_ITERS', '200'))
    warmup = int(os.environ.get('CLR_WARMUP', '50'))

    _config = read_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: no CUDA device; these numbers are meaningless for the paper.')

    H, W = _config['input_size']
    print(f'config={cfg_path}\nmode={mode}  input={H}x{W}  iters={iters} (+{warmup} warm-up)  '
          f'device={torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}\n')

    # ---- backbone -------------------------------------------------------------------------
    from models.CLRNet import CLRNet, CRNet

    common = dict(input_dims=(H, W), use_feat_from=1, md=4,
                  use_reflectance_RADAR=_config['use_reflectance_RADAR'],
                  use_velocity_RADAR=_config['use_velocity_RADAR'],
                  use_time_RADAR=_config['use_time_RADAR'],
                  use_depth=_config['use_depth'], dropout=_config['dropout'], res_num=18)

    # The encoders are built with a channel count derived from the feature flags, so the dummy
    # inputs must match exactly or the first conv will reject them.
    use_refl_lidar = _config.get('use_reflectance_LIDAR', True)
    radar_ch = (1 + _config['use_reflectance_RADAR'] + _config['use_velocity_RADAR']
                + _config['use_time_RADAR'])
    lidar_ch = 2 if use_refl_lidar else 1

    # The forward indexes its inputs as rgb_list[:, i, ...], i.e. (B, multiple_input, C, H, W) --
    # the encoders consume a stack of frames, not a single image.
    mi = _config['multiple_input']
    rgb = torch.randn(1, mi, 3, H, W, device=device)
    depth_in = torch.randn(1, mi, 1, H, W, device=device)
    radar = torch.randn(1, mi, radar_ch, H, W, device=device)
    lidar = torch.randn(1, mi, lidar_ch, H, W, device=device)

    if mode == 'CLR':
        model = CLRNet(use_reflectance_LIDAR=use_refl_lidar, **common).to(device).eval()
        fwd = lambda: model(rgb, lidar, radar, depth_in)
    else:
        model = CRNet(**common).to(device).eval()
        fwd = lambda: model(rgb, radar, depth_in)

    with torch.no_grad():
        backbone_ms = timeit(fwd, iters, warmup, f'{mode}Net backbone (no depth)')

        # ---- depth backbones --------------------------------------------------------------
        from depth_anything_v2.dpt import DepthAnythingV2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        # DepthAnythingV2 runs on the full-resolution camera image, not the projected tensor.
        img = np.random.randint(0, 255, (1200, 1920, 3), dtype=np.uint8)

        for enc in ('vits', 'vitl'):
            ckpt = next((p for p in (f'/home/appuser/data/checkpoints/depth_anything_v2_{enc}.pth',
                                     f'/home/appuser/models/depth_anything_v2_{enc}.pth') if os.path.exists(p)),
                        f'/home/appuser/data/checkpoints/depth_anything_v2_{enc}.pth')
            if not os.path.exists(ckpt):
                print(f'{enc}: checkpoint missing ({ckpt}) -- skipped')
                continue
            dmodel = DepthAnythingV2(**model_configs[enc])
            dmodel.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            dmodel = dmodel.to(device).eval()

            depth_ms = timeit(lambda: dmodel.infer_image(img), iters, warmup,
                              f'  DepthAnythingV2 {enc} only')
            total = backbone_ms + depth_ms
            print(f'  => {mode}Net + {enc} total{"":<15} {total:7.1f} ms'
                  f'{"":>27}-> {1000.0 / total:5.1f} fps\n')

            del dmodel
            torch.cuda.empty_cache()

    print('Quote the medians for typical-frame latency and state the GPU. '
          'These supersede the eval scripts\' "average run time", which is unsynchronised.')


if __name__ == '__main__':
    main()
