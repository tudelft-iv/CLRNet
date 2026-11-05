import random
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import SETTINGS
from tqdm import tqdm
import time

from models.CLRNet import LCCNet, CRNet, CLNet, CRSchollerNet
from models.FourDRCOC import TestNet
from CLRNet.datasets.DatasetCLR import DatasetCRDual
from CLRNet.datasets.DatasetCLR import DatasetCLVoD, DatasetCRVoD

from CLRNet.utils.utils import (mat2xyzrpy, merge_inputs, quat2mat,
                   quaternion_from_matrix,
                   tvector2mat, quaternion_distance, write_results_to_txt, read_config)

from CLRNet.utils.checkpoint import load_checkpoint, read_checkpoint_config, apply_checkpoint_config, mode_from_config
from CLRNet.utils.preprocess import preprocess_RTs_pairwise, preprocess_sample_CL, preprocess_sample_CR, preprocess_sample_CL_LCC, preprocess_sample_CRScholler, preprocess_sample_CR_Dual

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))

from CLRNet.visualization.inference_and_projection import save_imgs

# import matplotlib
# matplotlib.rc("font",family='AR PL UMing CN')
plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font',family='DejaVu Sans')
font_EN = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 16}
plt_size = 10.5

ex = Experiment("CLRNet-evaluate-iterative")
ex.captured_out_filter = apply_backspaces_and_linefeeds
SETTINGS.CAPTURE_MODE = 'no'

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CLR_GPU', '0')

save_images = os.environ.get('CLR_SAVE_IMAGES', '0') == '1'
# CLR_SAVE_ONLY_IDX: comma-separated batch indices to render. The split cannot be subset instead,
# because the val_RT CSV is replayed by row position and subsetting would change the miscalibrations.
save_only = {int(x) for x in os.environ['CLR_SAVE_ONLY_IDX'].split(',')} if os.environ.get('CLR_SAVE_ONLY_IDX') else None
mode = os.environ.get('CLR_MODE', 'CR')

CKPT = '/home/appuser/data/checkpoints/'
DEFAULT_WEIGHTS = {
    'CR': [CKPT + 'crnet_vod_100cm_20deg.pth', CKPT + 'crnet_vod_50cm_5deg.pth',
           CKPT + 'crnet_vod_20cm_1deg.pth', CKPT + 'crnet_vod_5cm_0.5deg.pth'],
    'CRDual': [CKPT + 'crnet_dualradar_150cm_20deg.pth', CKPT + 'crnet_dualradar_100cm_10deg.pth',
               CKPT + 'crnet_dualradar_50cm_5deg.pth', CKPT + 'crnet_dualradar_20cm_2deg.pth',
               CKPT + 'crnet_dualradar_10cm_1deg.pth'],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1

def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@ex.automain
def main(_config, seed):
    global EPOCH, mode
    weights = os.environ['CLR_WEIGHTS'].split(',') if os.environ.get('CLR_WEIGHTS') else DEFAULT_WEIGHTS[mode]
    ckpt_config = read_checkpoint_config(weights[0])
    if not os.environ.get('CLR_MODE'):
        mode = mode_from_config(ckpt_config, mode)
    config_name = 'Dual_CR_config.yaml' if mode == 'CRDual' else 'VOD_' + mode + '_config.yaml'
    _config = read_config(os.environ.get('CLR_CONFIG', '/home/appuser/src/calibration/CLRNet/configs/' + config_name))
    apply_checkpoint_config(_config, ckpt_config)
    output = os.environ.get('CLR_OUTPUT', os.path.join('/home/appuser/data/output', mode,
                                                       os.path.splitext(os.path.basename(weights[0]))[0]))
    print('Mode:', mode, ' Output folder:', output)


    if mode == 'CRDual':
        dataset_class = DatasetCRDual
    elif mode == 'CR' or mode == 'CRScholler' or mode == '4DRCOC':
        dataset_class = DatasetCRVoD
    elif mode == 'CL' or mode == 'LCC':
        dataset_class = DatasetCLVoD

    # use_depth_input: false feeds the depth branch an all-zero image instead of the Depth-Anything-V2 map.
    if mode == 'CR' or mode == 'CRDual':
        use_depth_input = _config.get('use_depth_input', _config['use_depth'])
        print('Using Depth INPUT (real depth maps, not zeros): {}'.format(use_depth_input))
        if mode == 'CRDual':
            # downsample_points lets the cross-domain run match the radar density the source model was
            # trained on; 0/None keeps every point (see parse_point_cloud).
            dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=_config['multiple_input'], rgb_depth_input=use_depth_input, downsample_points=_config.get('downsample_points', 0))
        else:
            dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=_config['multiple_input'], rgb_depth_input=use_depth_input, radar_frames=_config.get('radar_frames', 'radar_5frames'))
    else:
        dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=_config['multiple_input'], radar_frames=_config.get('radar_frames', 'radar_5frames'))

    # Optional: evaluate with a different stored miscalibration CSV.
    miscalib_csv = os.environ.get('CLR_MISCALIB_CSV')
    if miscalib_csv:
        if mode != 'LCC':
            raise ValueError('CLR_MISCALIB_CSV is currently supported only for LCC evaluation')
        miscalibrations = np.loadtxt(miscalib_csv, delimiter=',', skiprows=1)
        if len(miscalibrations) != len(dataset_val):
            raise ValueError('Miscalibration CSV has {} rows; expected {}'.format(
                len(miscalibrations), len(dataset_val)))
        dataset_val.val_RT_CR = miscalibrations.tolist()
        print('Using explicit LCC miscalibration CSV: {}'.format(miscalib_csv))

    def init_fn(x):
        return _init_fn(x, seed)

    num_worker = _config['num_worker']
    batch_size = _config['batch_size']

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    # Initialize error lists
    errors_r, errors_t, errors_t2, errors_rpy, mis_calib_list = [], [], [], [], []
    RTs = []
    predicted_RT = []
    total_time = 0

    # Pre-create error lists based on the number of weights
    for i in range(len(weights) + 1):
        errors_r.append([])
        errors_t.append([])
        errors_t2.append([])
        errors_rpy.append([])

        RTs.append([])

        predicted_RT.append([])

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
            
        RTs_sample = preprocess_RTs_pairwise(sample)
        RTs[0].append(RTs_sample)

        predicted_RT_temp = []

        for i in range(len(sample['initial_RT'])):
            mis_calib = sample['initial_RT'][i][1:]
            mis_calib_list.append(mis_calib)

            T_composed = RTs_sample[i][:3, 3]
            R_composed = quaternion_from_matrix(RTs_sample[i])
            errors_t[0].append(T_composed.norm().item())
            errors_t2[0].append(T_composed)
            errors_r[0].append(quaternion_distance(R_composed.unsqueeze(0),
                                                    torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                    R_composed.device))
            rpy_error = mat2xyzrpy(RTs_sample[i])[3:]
            rpy_error *= (180.0 / 3.141592)
            errors_rpy[0].append(rpy_error)

            predicted_RT_temp.append(torch.eye(4, device=RTs_sample[i].device))

        predicted_RT[0].append(predicted_RT_temp)

    # Iterate over the model weights, load the model, and process the test set for each model
    for iteration, weight in enumerate(weights):
        print("Loading weight number:", iteration + 1, "named: ", weight)
        checkpoint = load_checkpoint(weight)
        apply_checkpoint_config(_config, checkpoint.get('config') or read_checkpoint_config(weight), stage=True)
        if mode == 'CR' or mode == 'CRDual':
            model = CRNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                       use_reflectance_RADAR=_config['use_reflectance_RADAR'], use_velocity_RADAR = _config['use_velocity_RADAR'], use_time_RADAR=_config['use_time_RADAR'], use_depth=_config['use_depth'],
                       dropout=_config['dropout'], res_num=18)
            
        if mode == 'CL':
            model = CLNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                       use_reflectance_LIDAR=_config['use_reflectance_LIDAR'], use_depth=_config['use_depth'],
                       dropout=_config['dropout'], res_num=18)
            
        if mode == 'LCC':
            model = LCCNet((256,512))

        if mode == 'CRScholler':
            model = CRSchollerNet()

        if mode == '4DRCOC':
            model = TestNet(image_size=(_config['input_size'][0], _config['input_size'][1]),
                            use_feat_from=1, md=4, attention=False,
                            dropout=_config['dropout'], res_num=18)

        
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        total_time = 0

        # Iterate over the test set 
        for batch_idx, sample in enumerate(tqdm(TestImgLoader)):

            if mode == 'CL':
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CL(sample, _config, RTs[iteration][batch_idx])
            elif mode == 'CR':
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR(sample, _config, RTs[iteration][batch_idx])
            elif mode == 'CRDual':
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR_Dual(sample, _config, RTs[iteration][batch_idx])
            elif mode == 'LCC':
                rgb_input, radar_input, _ = preprocess_sample_CL_LCC(sample, _config, RTs[iteration][batch_idx])
            elif mode == 'CRScholler':
                rgb_input, radar_input = preprocess_sample_CRScholler(sample, _config, RTs[iteration][batch_idx])
            elif mode == '4DRCOC':
                rgb_input, radar_input, _ = preprocess_sample_CR(sample, _config, RTs[iteration][batch_idx])

            start_time = time.time()

            with torch.no_grad():
                RTs_temp = []
                predicted_RT_temp = []
                if mode == 'LCC':
                    T_predicted, R_predicted = model(rgb_input, radar_input)
                if mode == 'CRScholler':
                    R_predicted = model(rgb_input, radar_input)
                if mode == '4DRCOC':
                    T_predicted, R_predicted = model(rgb_input, radar_input, True)
                if mode == 'CL' or mode == 'CR' or mode == 'CRDual':    
                    T_predicted, R_predicted = model(rgb_input, radar_input, rgb_depth_input)
                inference_time = time.time() - start_time
                total_time += inference_time

                for i in range(len(sample['initial_RT'])):
                    if mode == 'CRScholler':
                        RT_predicted = quat2mat(R_predicted[i])
                    else:
                        RT_predicted = torch.mm(tvector2mat(T_predicted[i]), quat2mat(R_predicted[i]))
                    RTs_temp.append(torch.mm(RTs[iteration][batch_idx][i], RT_predicted.inverse()))
                    predicted_RT_temp.append(torch.mm(RT_predicted, predicted_RT[iteration][batch_idx][i]))

                    T_composed = RTs_temp[i][:3, 3]
                    R_composed = quaternion_from_matrix(RTs_temp[i])
                    errors_t[iteration + 1].append(T_composed.norm().item())
                    errors_t2[iteration + 1].append(T_composed)
                    errors_r[iteration + 1].append(quaternion_distance(R_composed.unsqueeze(0),
                                                                        torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                                        R_composed.device))
                    rpy_error = mat2xyzrpy(RTs_temp[i])[3:]
                    rpy_error *= (180.0 / 3.141592)
                    errors_rpy[iteration + 1].append(rpy_error)

                    root_folder = output + '/projections' + '/iter' + str(iteration + 1)

                    if iteration == len(weights) - 1 and save_images and (save_only is None or batch_idx in save_only):
                        # the camera--radar modes carry the cloud under 'radar'; CL/LCC under 'lidar'
                        cloud_key = 'radar' if mode in ('CR', 'CRDual', 'CRScholler') else 'lidar'
                        for i in range(len(sample['initial_RT'])):
                            sample[cloud_key][i] = sample[cloud_key][i].cuda()
                            lidar = sample[cloud_key][i]
                            lidar = lidar[:3, :]
                            tr_error = sample['tr_error'][i]
                            rot_error = sample['rot_error'][i]
                            # Radar panels are rendered at point_size=7 to match the qualitative radar figure produced by
                            # eval_CLR.py; lidar stays at 3. Using one size for both made the radar figures
                            # unreadable against the dense Arbe cloud.
                            ps = 7 if cloud_key == 'radar' else 3
                            save_imgs(os.path.join(root_folder, cloud_key), sample['orig_img'][i], sample['calib'][i], torch.mm(RT_predicted, predicted_RT[iteration][batch_idx][i]), lidar, batch_idx, tr_error, rot_error, point_size=ps, colormap='viridis')

            RTs[iteration + 1].append(RTs_temp)
            predicted_RT[iteration + 1].append(predicted_RT_temp)

        del model
        torch.cuda.empty_cache()

    for i in range(len(weights) + 1):
        errors_r[i] = torch.tensor(errors_r[i]).abs() * (180.0 / 3.141592)
        errors_t[i] = torch.tensor(errors_t[i]).abs() * 100

        for k in range(len(errors_rpy[i])):
            errors_rpy[i][k] = errors_rpy[i][k].clone().detach().abs()
            errors_t2[i][k] = errors_t2[i][k].clone().detach().abs() * 100

        write_results_to_txt(output, errors_t, errors_r, errors_t2, errors_rpy, i, mode)

    # Per-frame dump for the worst-K domain-transfer analysis (paper: Discussion). The test loader runs with
    # shuffle=False, so errors_t/errors_r are in dataset order and line up with dataset_val.indexes.
    if os.environ.get('CLR_DUMP_PERFRAME') == '1':
        frame_ids = getattr(dataset_val, 'indexes', None)
        csv_path = os.path.join(output, 'per_frame_errors.csv')
        with open(csv_path, 'w') as fh:
            fh.write('iteration,index,frame,transl_cm,rot_deg\n')
            for it in range(len(weights) + 1):
                for n in range(len(errors_t[it])):
                    fid = frame_ids[n] if frame_ids is not None and n < len(frame_ids) else ''
                    fh.write('%d,%d,%s,%.6f,%.6f\n' % (it, n, fid, errors_t[it][n].item(), errors_r[it][n].item()))
        print('Wrote per-frame errors to', csv_path)

    avg_time = total_time / len(TestImgLoader)
    print("average run time on {} iteration: {} s".format(len(weights), avg_time))
    print("End!")
