"""Per-frame explanatory factors for the cross-domain worst-case analysis (paper: Discussion).

Reads the Dual-Radar test frames directly from disk (no GPU, no network) and records, per frame,
the quantities the paper's Discussion names as suspected causes of the domain gap:

  n_points      radar returns in the frame, the density factor (Arbe Phoenix against ZF FRGen21)
  mean_range    mean radial distance of the returns, a proxy for how far the structure sits
  frac_near     fraction of returns within 20 m, where the projected geometry is best conditioned
  brightness    mean image luminance, a proxy for the day/night distribution shift
  contrast      standard deviation of luminance, low for both night frames and washed-out ones

Join it against per_frame_errors.csv (written by eval_pairwise.py with CLR_DUMP_PERFRAME=1) to rank
the factors by how much they actually separate the worst frames from the rest.

    python3 crossdomain_factors.py            # writes factors.csv next to the Dual-Radar data
"""
import os
import numpy as np
from PIL import Image

ROOT = os.environ.get('CLR_DUAL_ROOT', '/home/appuser/data/DualRadar')
SPLIT = os.path.join(ROOT, 'lidar', 'ImageSets', 'test.txt')
OUT = os.environ.get('CLR_FACTORS_OUT', '/home/appuser/data/output/CRDual/factors.csv')


def main():
    with open(SPLIT) as fh:
        frames = [ln.strip() for ln in fh if ln.strip()]
    print('%d test frames' % len(frames))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as out:
        out.write('frame,n_points,mean_range,frac_near,brightness,contrast\n')
        for k, f in enumerate(frames):
            radar_path = os.path.join(ROOT, 'radar', 'training', 'velodyne', f + '.bin')
            img_path = os.path.join(ROOT, 'lidar', 'training', 'image_2', f + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(ROOT, 'lidar', 'training', 'image_2', f + '.png')

            pc = np.fromfile(radar_path, dtype=np.float32).reshape((-1, 5))
            xyz = pc[:, :3]
            rng = np.linalg.norm(xyz, axis=1)
            n_points = len(pc)
            mean_range = float(rng.mean()) if n_points else 0.0
            frac_near = float((rng < 20.0).mean()) if n_points else 0.0

            img = np.asarray(Image.open(img_path).convert('L'), dtype=np.float32)
            brightness, contrast = float(img.mean()), float(img.std())

            out.write('%s,%d,%.4f,%.4f,%.4f,%.4f\n'
                      % (f, n_points, mean_range, frac_near, brightness, contrast))
            if (k + 1) % 500 == 0:
                print('  %d/%d' % (k + 1, len(frames)))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
