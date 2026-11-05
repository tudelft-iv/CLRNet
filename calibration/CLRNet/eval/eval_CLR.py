import csv
import gc
import random
import open3d as o3

import cv2

import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# Sacred buffers every byte written to stdout for the run report. tqdm redraws its bar ~12x/second and this
# script makes 5 passes over 1334 batches, so that buffer (plus apply_backspaces_and_linefeeds over it) grew to
# well over 10 GB and OOM-killed the run. Nothing here depends on the captured output - it all goes to the log
# file anyway - so turn the capture off.
SETTINGS.CAPTURE_MODE = 'no'
from skimage import io
from tqdm import tqdm, trange
import time

from CLRNet.models.CLRNet import CLRNet, CLRHayoun
from CLRNet.datasets.DatasetCLR import DatasetCLRVoD

from CLRNet.utils.utils import (mat2xyzrpy, merge_inputs, quat2mat,
                   quaternion_from_matrix,
                   tvector2mat, quaternion_distance, write_results_to_txt, read_config)

from CLRNet.utils.preprocess import preprocess_sample_CLR, preprocess_RTs_CLR

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))

from CLRNet.visualization.inference_and_projection import save_imgs
from CLRNet.utils.checkpoint import load_checkpoint, read_checkpoint_config, apply_checkpoint_config

# import matplotlib
# matplotlib.rc("font",family='AR PL UMing CN')
plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font',family='DejaVu Sans')
font_EN = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 16}
plt_size = 10.5

ex = Experiment("CLRNet-evaluate-iterative")
ex.captured_out_filter = apply_backspaces_and_linefeeds

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CLR_GPU', '0')

save_images = os.environ.get('CLR_SAVE_IMAGES', '0') == '1'
mode = os.environ.get('CLR_MODE', 'CLR')

CKPT = '/home/appuser/data/checkpoints/'
DEFAULT_WEIGHTS = [CKPT + 'clrnet_vod_100cm_20deg.pth', CKPT + 'clrnet_vod_50cm_5deg.pth',
                   CKPT + 'clrnet_vod_20cm_1deg.pth', CKPT + 'clrnet_vod_5cm_0.5deg.pth']
weights = os.environ['CLR_WEIGHTS'].split(',') if os.environ.get('CLR_WEIGHTS') else DEFAULT_WEIGHTS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This script keeps one 4x4 transform per sensor pair, per sample, per refinement stage. On the 1334-frame test
# split with a 4-model cascade that is ~100k tensors; leaving them on the GPU grew host RSS to ~20 GB during the
# initial pass alone (CUDA caching-allocator bookkeeping, not the 1.3 MB the matrices actually occupy) and
# OOM-killed the run. The matrices are tiny, so park them on the CPU and page each batch back in on demand.
# _keep() also clones, because entries like T_composed = RT[:3, 3] are *views* that would otherwise pin their
# 4x4 parent alive for the whole run.
def _keep(tensors):
    """Detach a list of tensors to CPU for long-term accumulation."""
    return [t.detach().to('cpu', copy=True) for t in tensors]


def _use(tensors):
    """Move an accumulated batch back to the compute device."""
    return [t.to(device) for t in tensors]


EPOCH = 1

def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    # print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@ex.automain
def main(_config, seed):
    global EPOCH, weights
    _config = read_config(os.environ.get('CLR_CONFIG', '/home/appuser/src/calibration/CLRNet/configs/VOD_CLR_config.yaml'))

    apply_checkpoint_config(_config, read_checkpoint_config(weights[0]))
    output = os.environ.get('CLR_OUTPUT', os.path.join('/home/appuser/data/output', mode,
                                                       os.path.splitext(os.path.basename(weights[0]))[0]))
    print("Output folder: ", output)

    dataset_class = DatasetCLRVoD

    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=_config['multiple_input'])

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
    errors_r_CL, errors_t_CL, errors_t2_CL, errors_rpy_CL, mis_calib_list_CL = [], [], [], [], []
    errors_r_CR, errors_t_CR, errors_t2_CR, errors_rpy_CR, mis_calib_list_CR = [], [], [], [], []
    errors_r_LR, errors_t_LR, errors_t2_LR, errors_rpy_LR, mis_calib_list_LR = [], [], [], [], []
    RTs_CL, RTs_CR, RTs_LR = [], [], []
    predicted_RT_CL, predicted_RT_CR, predicted_RT_LR = [], [], []


    total_time = 0

    # Pre-create error lists based on the number of weights
    for i in range(len(weights) + 1):
        errors_r_CL.append([])
        errors_t_CL.append([])
        errors_t2_CL.append([])
        errors_rpy_CL.append([])

        errors_r_CR.append([])
        errors_t_CR.append([])
        errors_t2_CR.append([])
        errors_rpy_CR.append([])

        errors_r_LR.append([])
        errors_t_LR.append([])
        errors_t2_LR.append([])
        errors_rpy_LR.append([])

        RTs_CL.append([])
        RTs_CR.append([])
        RTs_LR.append([])

        predicted_RT_CL.append([])
        predicted_RT_CR.append([])
        predicted_RT_LR.append([])

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
            
        RTs_CL_sample, RTs_CR_sample, RTs_LR_sample = preprocess_RTs_CLR(sample)
        RTs_CL[0].append(_keep(RTs_CL_sample))
        RTs_CR[0].append(_keep(RTs_CR_sample))
        RTs_LR[0].append(_keep(RTs_LR_sample))

        predicted_RT_CL_temp = []
        predicted_RT_CR_temp = []
        predicted_RT_LR_temp = []

        for i in range(len(sample['initial_RT_CL'])):
            mis_calib_CL = sample['initial_RT_CL'][i][1:]
            mis_calib_list_CL.append(mis_calib_CL)

            T_composed_CL = RTs_CL_sample[i][:3, 3]
            R_composed_CL = quaternion_from_matrix(RTs_CL_sample[i])
            errors_t_CL[0].append(T_composed_CL.norm().item())
            errors_t2_CL[0].append(T_composed_CL.detach().cpu().clone())
            errors_r_CL[0].append(quaternion_distance(R_composed_CL.unsqueeze(0),
                                                    torch.tensor([1., 0., 0., 0.], device=R_composed_CL.device).unsqueeze(0),
                                                    R_composed_CL.device).detach().cpu())
            rpy_error_CL = mat2xyzrpy(RTs_CL_sample[i])[3:]
            rpy_error_CL *= (180.0 / 3.141592)
            errors_rpy_CL[0].append(rpy_error_CL.detach().cpu().clone())

            mis_calib_CR = sample['initial_RT_CR'][i][1:]
            mis_calib_list_CR.append(mis_calib_CR)

            T_composed_CR = RTs_CR_sample[i][:3, 3]
            R_composed_CR = quaternion_from_matrix(RTs_CR_sample[i])
            errors_t_CR[0].append(T_composed_CR.norm().item())
            errors_t2_CR[0].append(T_composed_CR.detach().cpu().clone())
            errors_r_CR[0].append(quaternion_distance(R_composed_CR.unsqueeze(0),
                                                    torch.tensor([1., 0., 0., 0.], device=R_composed_CR.device).unsqueeze(0),
                                                    R_composed_CR.device).detach().cpu())
            rpy_error_CR = mat2xyzrpy(RTs_CR_sample[i])[3:]
            rpy_error_CR *= (180.0 / 3.141592)
            errors_rpy_CR[0].append(rpy_error_CR.detach().cpu().clone())

            mis_calib_LR = sample['initial_RT_LR'][i][1:]
            mis_calib_list_LR.append(mis_calib_LR)

            T_composed_LR = RTs_LR_sample[i][:3, 3]
            R_composed_LR = quaternion_from_matrix(RTs_LR_sample[i])
            errors_t_LR[0].append(T_composed_LR.norm().item())
            errors_t2_LR[0].append(T_composed_LR.detach().cpu().clone())
            errors_r_LR[0].append(quaternion_distance(R_composed_LR.unsqueeze(0),
                                                    torch.tensor([1., 0., 0., 0.], device=R_composed_LR.device).unsqueeze(0),
                                                    R_composed_LR.device).detach().cpu())
            rpy_error_LR = mat2xyzrpy(RTs_LR_sample[i])[3:]
            rpy_error_LR *= (180.0 / 3.141592)
            errors_rpy_LR[0].append(rpy_error_LR.detach().cpu().clone())

            predicted_RT_CL_temp.append(torch.eye(4, device=RTs_CL_sample[i].device))
            predicted_RT_CR_temp.append(torch.eye(4, device=RTs_CL_sample[i].device))
            predicted_RT_LR_temp.append(torch.eye(4, device=RTs_CL_sample[i].device))

        predicted_RT_CL[0].append(_keep(predicted_RT_CL_temp))
        predicted_RT_CR[0].append(_keep(predicted_RT_CR_temp))
        predicted_RT_LR[0].append(_keep(predicted_RT_LR_temp))   

    # Iterate over the model weights, load the model, and process the test set for each model
    for iteration, weight in enumerate(tqdm(weights, desc='Loading weights')):
        checkpoint = load_checkpoint(weight)
        apply_checkpoint_config(_config, checkpoint.get('config') or read_checkpoint_config(weight), stage=True)
        if mode == 'Hayoun':
            model = CLRHayoun()
        else:
            model = CLRNet(input_dims=(_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                        use_reflectance_LIDAR=_config['use_reflectance_LIDAR'], use_reflectance_RADAR=_config['use_reflectance_RADAR'],
                        use_velocity_RADAR=_config['use_velocity_RADAR'], use_time_RADAR=_config['use_time_RADAR'],
                        use_depth=_config['use_depth'], dropout=_config['dropout'], res_num=18)
        
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

        # Free the CPU copy of the (large) checkpoint before the evaluation pass.
        del checkpoint, saved_state_dict
        gc.collect()
        model = model.to(device)
        model.eval()

        total_time = 0
        num_batches = 0

        # Iterate over the test set 
        for batch_idx, sample in enumerate(tqdm(TestImgLoader)):

            start_time = time.time()

            # The RTs_*/predicted_RT_* accumulators are held on the CPU (see _keep/_use below); move just this
            # batch's 4x4 matrices back to the GPU for the forward pass and the compositions that follow.
            rts_cl_dev = _use(RTs_CL[iteration][batch_idx])
            rts_cr_dev = _use(RTs_CR[iteration][batch_idx])
            rts_lr_dev = _use(RTs_LR[iteration][batch_idx])
            prt_cl_dev = _use(predicted_RT_CL[iteration][batch_idx])
            prt_cr_dev = _use(predicted_RT_CR[iteration][batch_idx])
            prt_lr_dev = _use(predicted_RT_LR[iteration][batch_idx])

            rgb_input, lidar_input, radar_input, rgb_depth_input = preprocess_sample_CLR(sample, _config, rts_cl_dev, rts_cr_dev)

            with torch.no_grad():
                RTs_CL_temp, RTs_CR_temp, RTs_LR_temp = [], [], []
                predicted_RT_CL_temp, predicted_RT_CR_temp, predicted_RT_LR_temp = [], [], []
                # RTs_CL_compensated, RTs_CR_compensated, RTs_LR_compensated = [], [], []
                t1 = time.time()
                T_predicted_CL, R_predicted_CL, T_predicted_LR, R_predicted_LR, T_predicted_CR, R_predicted_CR = model(rgb_input, lidar_input, radar_input, rgb_depth_input)
                inference_time = time.time() - start_time
                total_time += inference_time
                num_batches += 1

                for i in range(len(sample['initial_RT_CL'])):
                    RT_predicted_CL = torch.mm(tvector2mat(T_predicted_CL[i]), quat2mat(R_predicted_CL[i]))
                    # RTs_CL_temp.append(torch.mm(RTs_CL[iteration][batch_idx][i], RT_predicted_CL))
                    RTs_CL_temp.append(torch.mm(RT_predicted_CL.inverse(), rts_cl_dev[i]))
                    predicted_RT_CL_temp.append(torch.mm(RT_predicted_CL, prt_cl_dev[i]))

                    RT_predicted_CR = torch.mm(tvector2mat(T_predicted_CR[i]), quat2mat(R_predicted_CR[i]))
                    # RTs_CR_temp.append(torch.mm(RTs_CR[iteration][batch_idx][i], RT_predicted_CR))
                    RTs_CR_temp.append(torch.mm(RT_predicted_CR.inverse(), rts_cr_dev[i]))
                    predicted_RT_CR_temp.append(torch.mm(RT_predicted_CR, prt_cr_dev[i]))

                    RT_predicted_LR = torch.mm(tvector2mat(T_predicted_LR[i]), quat2mat(R_predicted_LR[i]))
                    # RTs_LR_temp.append(torch.mm(RTs_LR[iteration][batch_idx][i], RT_predicted_LR))
                    RTs_LR_temp.append(torch.mm(RT_predicted_LR.inverse(), rts_lr_dev[i]))
                    predicted_RT_LR_temp.append(torch.mm(RT_predicted_LR, prt_lr_dev[i]))

                    T_composed_CL = RTs_CL_temp[i][:3, 3]
                    R_composed_CL = quaternion_from_matrix(RTs_CL_temp[i])
                    errors_t_CL[iteration + 1].append(T_composed_CL.norm().item())
                    errors_t2_CL[iteration + 1].append(T_composed_CL.detach().cpu().clone())
                    errors_r_CL[iteration + 1].append(quaternion_distance(R_composed_CL.unsqueeze(0),
                                                                        torch.tensor([1., 0., 0., 0.], device=R_composed_CL.device).unsqueeze(0),
                                                                        R_composed_CL.device).detach().cpu())
                    rpy_error_CL = mat2xyzrpy(RTs_CL_temp[i])[3:]
                    rpy_error_CL *= (180.0 / 3.141592)
                    errors_rpy_CL[iteration + 1].append(rpy_error_CL.detach().cpu().clone())

                    T_composed_CR = RTs_CR_temp[i][:3, 3]
                    R_composed_CR = quaternion_from_matrix(RTs_CR_temp[i])
                    errors_t_CR[iteration + 1].append(T_composed_CR.norm().item())
                    errors_t2_CR[iteration + 1].append(T_composed_CR.detach().cpu().clone())
                    errors_r_CR[iteration + 1].append(quaternion_distance(R_composed_CR.unsqueeze(0),
                                                                        torch.tensor([1., 0., 0., 0.], device=R_composed_CR.device).unsqueeze(0),
                                                                        R_composed_CR.device).detach().cpu())
                    rpy_error_CR = mat2xyzrpy(RTs_CR_temp[i])[3:]
                    rpy_error_CR *= (180.0 / 3.141592)
                    errors_rpy_CR[iteration + 1].append(rpy_error_CR.detach().cpu().clone())

                    T_composed_LR = RTs_LR_temp[i][:3, 3]
                    R_composed_LR = quaternion_from_matrix(RTs_LR_temp[i])
                    errors_t_LR[iteration + 1].append(T_composed_LR.norm().item())
                    errors_t2_LR[iteration + 1].append(T_composed_LR.detach().cpu().clone())
                    errors_r_LR[iteration + 1].append(quaternion_distance(R_composed_LR.unsqueeze(0),
                                                                        torch.tensor([1., 0., 0., 0.], device=R_composed_LR.device).unsqueeze(0),
                                                                        R_composed_LR.device).detach().cpu())
                    rpy_error_LR = mat2xyzrpy(RTs_LR_temp[i])[3:]
                    rpy_error_LR *= (180.0 / 3.141592)
                    errors_rpy_LR[iteration + 1].append(rpy_error_LR.detach().cpu().clone())

                    root_folder = '/home/appuser/data/output/' + 'Hayoun' + '/projections' + '/iter' + str(iteration + 1)

                    if iteration == len(weights) - 1 and save_images:
                        for i in range(len(sample['initial_RT_CL'])):
                            sample['radar'][i] = sample['radar'][i].cuda()
                            radar = sample['radar'][i]
                            radar = radar[:3, :]
                            tr_error = sample['tr_error_CR'][i]
                            rot_error = sample['rot_error_CR'][i]
                            save_imgs(os.path.join(root_folder, 'radar'), sample['orig_img'][i], sample['calib'][i], torch.mm(RT_predicted_CR, prt_cr_dev[i]), radar, batch_idx, tr_error, rot_error, point_size=7, colormap='viridis')

                            sample['lidar'][i] = sample['lidar'][i].cuda()
                            lidar = sample['lidar'][i]
                            lidar = lidar[:3, :]
                            tr_error = sample['tr_error_CL'][i]
                            rot_error = sample['rot_error_CL'][i]
                            save_imgs(os.path.join(root_folder, 'lidar'), sample['orig_img'][i], sample['calib'][i], torch.mm(RT_predicted_CL, prt_cl_dev[i]), lidar, batch_idx, tr_error, rot_error, point_size=3, colormap='viridis')

            RTs_CL[iteration + 1].append(_keep(RTs_CL_temp))
            RTs_CR[iteration + 1].append(_keep(RTs_CR_temp))
            RTs_LR[iteration + 1].append(_keep(RTs_LR_temp))

            predicted_RT_CL[iteration + 1].append(_keep(predicted_RT_CL_temp))
            predicted_RT_CR[iteration + 1].append(_keep(predicted_RT_CR_temp))
            predicted_RT_LR[iteration + 1].append(_keep(predicted_RT_LR_temp))

            # Host memory grew by ~17 MB per batch here - almost exactly one collated sample (orig_img ~7 MB,
            # depth ~2.4 MB, plus both point clouds) - which exhausted the 20 GB cgroup and got the run
            # SIGKILLed partway through the first refinement stage. Drop this batch's tensors explicitly and
            # break any reference cycles periodically; the accumulators above already hold CPU copies.
            del rgb_input, lidar_input, radar_input, rgb_depth_input
            del rts_cl_dev, rts_cr_dev, rts_lr_dev, prt_cl_dev, prt_cr_dev, prt_lr_dev
            del RTs_CL_temp, RTs_CR_temp, RTs_LR_temp
            del predicted_RT_CL_temp, predicted_RT_CR_temp, predicted_RT_LR_temp
            if batch_idx % 200 == 0:
                gc.collect()

        # Clear the GPU memory for the current model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # After processing all models, stack mis-calibration inputs
    mis_calib_input_CL = torch.stack(mis_calib_list_CL)
    mis_calib_input_CR = torch.stack(mis_calib_list_CR)
    mis_calib_input_LR = torch.stack(mis_calib_list_LR)

    for i in trange(len(weights) + 1, desc='Iterative refinement'):
        errors_r_CL[i] = torch.tensor(errors_r_CL[i]).abs() * (180.0 / 3.141592)
        errors_t_CL[i] = torch.tensor(errors_t_CL[i]).abs() * 100

        errors_r_CR[i] = torch.tensor(errors_r_CR[i]).abs() * (180.0 / 3.141592)
        errors_t_CR[i] = torch.tensor(errors_t_CR[i]).abs() * 100

        errors_r_LR[i] = torch.tensor(errors_r_LR[i]).abs() * (180.0 / 3.141592)
        errors_t_LR[i] = torch.tensor(errors_t_LR[i]).abs() * 100

        for k in range(len(errors_rpy_CL[i])):
            errors_rpy_CL[i][k] = errors_rpy_CL[i][k].clone().detach().abs()
            errors_t2_CL[i][k] = errors_t2_CL[i][k].clone().detach().abs() * 100

            errors_rpy_CR[i][k] = errors_rpy_CR[i][k].clone().detach().abs()
            errors_t2_CR[i][k] = errors_t2_CR[i][k].clone().detach().abs() * 100

            errors_rpy_LR[i][k] = errors_rpy_LR[i][k].clone().detach().abs()
            errors_t2_LR[i][k] = errors_t2_LR[i][k].clone().detach().abs() * 100

        write_results_to_txt(output, errors_t_CL, errors_r_CL, errors_t2_CL, errors_rpy_CL, i, "CL")
        write_results_to_txt(output, errors_t_CR, errors_r_CR, errors_t2_CR, errors_rpy_CR, i, "CR")
        write_results_to_txt(output, errors_t_LR, errors_r_LR, errors_t2_LR, errors_rpy_LR, i, "LR")

        # Per-frame dump, so that statistics the summary file cannot express -- a bootstrap interval on
        # the median, for instance -- can be computed afterwards without re-running the network.
        if os.environ.get('CLR_DUMP_PERFRAME') == '1':
            csv_path = os.path.join(output, 'per_frame_errors_iter%d.csv' % i)
            with open(csv_path, 'w') as fh:
                fh.write('pair,index,transl_cm,rot_deg\n')
                for pair, et, er in (('CL', errors_t_CL, errors_r_CL), ('CR', errors_t_CR, errors_r_CR),
                                     ('LR', errors_t_LR, errors_r_LR)):
                    for n in range(len(et[i])):
                        fh.write('%s,%d,%.6f,%.6f\n' % (pair, n, et[i][n].item(), er[i][n].item()))
            print('Wrote per-frame errors to', csv_path)

    # plot_figures(errors_r_CL, errors_t_CL, errors_t2_CL, errors_rpy_CL, mis_calib_input_CL, os.path.join(output, 'CL'), N=100)
    # plot_figures(errors_r_CR, errors_t_CR, errors_t2_CR, errors_rpy_CR, mis_calib_input_CR, os.path.join(output, 'CR'), N=100)
    # plot_figures(errors_r_LR, errors_t_LR, errors_t2_LR, errors_rpy_LR, mis_calib_input_LR, os.path.join(output, 'LR'), N=100)

    avg_time = total_time / len(TestImgLoader)
    average_inference_time = total_time / num_batches
    print(f"Average inference time per batch: {average_inference_time:.4f} seconds")
    print("average runing time on {} iteration: {} s".format(len(weights), avg_time))
    print("End!")
