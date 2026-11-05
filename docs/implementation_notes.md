# Implementation notes

Details of the code that the paper describes only briefly.

- **Projection.** Point clouds are transformed into the camera frame (x right, y down, z forward) and projected
  there with an equirectangular projection. Points beyond 35 m are dropped. The range channel is min–max
  normalised per image, inverted (near = bright), scaled to 0–255 and divided by `max_depth` (80 by default).
- **Radar preprocessing.** For View-of-Delft radar, z is divided by cos(azimuth). Points with |x| ≤ 2 m and
  |y| ≤ 2 m are removed.
- **Miscalibration sampling.** Six values are drawn uniformly within ±`max_r` and ±`max_t` and inverted as a
  pose; the resulting rotation is stored as a quaternion. Test miscalibrations are stored as
  `val_RT_*.csv` next to the data and replayed, so runs are comparable. The files used in the paper are in
  `calibration/CLRNet/datasets/miscalibrations/`.
- **Loop-closure weight.** `lambda_loop: 0.0625` in the configs is the weight that was effectively applied in
  the paper's runs.
- **CRNet depth input.** `use_depth_input` selects whether the CRNet depth branch receives the Depth-Anything-V2
  map or an all-zero image. The released View-of-Delft CRNet models use the depth map; the Dual-Radar models and
  the cross-domain runs use zeros.
- **Dual-Radar channels.** Radar column 3 is fed as the velocity channel and column 4 as the reflectance channel;
  the time channel is zero.
- **Correlation.** With 512×1024 inputs the correlation layer operates on 16×32 feature maps.
- **Checkpoint settings.** Every checkpoint stores its training settings under `config` (input channels,
  projection, `max_depth`, `use_depth_input`, miscalibration range, ...); the scripts apply them per cascade
  stage. `clrnet_vod_20cm_1deg.pth` was trained with `max_depth: 40`, all other models with 80.
