import argparse
import os
import random
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from models.CLRNet import CRNet, CLNet, CLRNet, CRSchollerNet
from CLRNet.datasets.DatasetCLR import DatasetCRVoD, DatasetCLVoD, DatasetCLRVoD
from CLRNet.utils.utils import (merge_inputs, quat2mat, rotate_forward, rotate_back,
                   tvector2mat, lidar_project_depth, read_config)
from CLRNet.utils.checkpoint import read_checkpoint_config, apply_checkpoint_config, mode_from_config

from CLRNet.utils.preprocess import preprocess_sample_CR, preprocess_sample_CL, preprocess_sample_CLR, preprocess_sample_CRScholler

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration evaluation.")
    parser.add_argument("--weights", nargs="+", required=True, help="Paths to model weights.")
    parser.add_argument("--config", default=None, help="Path to the configuration file.")
    parser.add_argument("--show", action="store_true", help="Show the projected images.")
    parser.add_argument("--save", action="store_true", help="Save the projected images.")
    parser.add_argument("--rigid", action="store_true", help="Assume rigid sensor platform.")
    parser.add_argument("--mode", default=None, help="Calibration mode (CR, CL, CLR); read from the checkpoint if omitted.")
    return parser.parse_args()


def apply_colormap(image, depth_map, point_size=5, colormap='viridis'):
    """Applies a colormap to an image based on the depth map."""
    assert image.shape[:2] == depth_map.shape, "Image and depth map must have the same shape"
    depth_map = depth_map.astype(np.float32)
    norm = plt.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
    cmap = cm.get_cmap(colormap)
    colors = cmap(norm(depth_map))
    output_image = image.astype(np.uint8).copy()

    # Draw farthest first, so nearer returns end up on top. With VoD's few hundred radar points the
    # circles barely overlap and this is indistinguishable from the raster order used previously; with
    # Dual-Radar's 10k+ point clouds raster order meant whichever point sat lowest in the image won,
    # which made the depth colouring incoherent regardless of the actual geometry.
    ys, xs = np.nonzero(depth_map)
    order = np.argsort(-depth_map[ys, xs])
    for i, j in zip(ys[order], xs[order]):
        color = colors[i, j, :3] * 255
        cv2.circle(output_image, (int(j), int(i)), point_size,
                   (int(color[0]), int(color[1]), int(color[2])), -1)
    return output_image


def project_points(lidar, calib, img_shape, img, point_size=5, colormap='viridis'):
    # zbuffer=True: figures show the nearest return per pixel. Network paths keep the default.
    depth_map, _ = lidar_project_depth(lidar, calib, img_shape, zbuffer=True)
    depth_map = depth_map.squeeze().cpu().numpy()
    img = apply_colormap(img, depth_map, point_size=point_size, colormap=colormap)
    return img


def save_imgs(root_folder, orig_img, calib, extrinsics, lidar, batch_idx, tr_error, rot_error,
              point_size=5, colormap='viridis', show=False, save=True):
    
    original_img = np.uint8(orig_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    gt_lidar = project_points(lidar, calib, original_img.shape[:2], original_img, point_size, colormap)
    initial_RT = torch.mm(tvector2mat(tr_error), quat2mat(rot_error))
    initial_lidar = rotate_back(lidar, initial_RT)
    initial_depth = project_points(initial_lidar, calib, original_img.shape[:2], original_img, point_size, colormap)
    predicted_lidar = rotate_forward(initial_lidar, extrinsics)
    predicted_depth = project_points(predicted_lidar, calib, original_img.shape[:2], original_img, point_size, colormap)

    if save:
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        cv2.imwrite(f'{root_folder}/original_img_{batch_idx}.png', original_img)
        cv2.imwrite(f'{root_folder}/gt_img_{batch_idx}.png', gt_lidar)
        cv2.imwrite(f'{root_folder}/initial_img_{batch_idx}.png', initial_depth)
        cv2.imwrite(f'{root_folder}/predicted_img_{batch_idx}.png', predicted_depth)

    if show:
        height, width = original_img.shape[:2]
        gt_lidar_resized = cv2.resize(gt_lidar, (width, height))
        initial_depth_resized = cv2.resize(initial_depth, (width, height))
        predicted_depth_resized = cv2.resize(predicted_depth, (width, height))

        # Concatenate images horizontally
        combined_image = cv2.hconcat([original_img, gt_lidar_resized, initial_depth_resized, predicted_depth_resized])

        # Resize the combined image to fit the screen if needed
        screen_width = 1920  # Adjust as per your screen resolution
        screen_height = 1080
        combined_height, combined_width = combined_image.shape[:2]

        if combined_width > screen_width or combined_height > screen_height:
            scaling_factor = min(screen_width / combined_width, screen_height / combined_height)
            new_width = int(combined_width * scaling_factor)
            new_height = int(combined_height * scaling_factor)
            combined_image = cv2.resize(combined_image, (new_width, new_height))

        # Display the combined image
        cv2.imshow("Demo Projections(Original image input, GT params, mis-calibrated params, predicted params)", combined_image)
        cv2.waitKey(20)

def run_eval(_config, seed, models=None, mode='CLR', show=False, save=False, rigid=False):
    if mode == 'CR':
        dataset_class = DatasetCRVoD
    if mode == 'CL':
        dataset_class = DatasetCLVoD
    if mode == 'CLR':
        dataset_class = DatasetCLRVoD
    if mode == 'CRScholler':
        dataset_class = DatasetCRVoD

    if rigid:
        test_same_miscalib_number = 0
    else:
        test_same_miscalib_number = 1
    extra = {'rgb_depth_input': _config.get('use_depth_input', _config['use_depth'])} if dataset_class is DatasetCRVoD else {}
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='demo', test_same_miscalib_number=test_same_miscalib_number, **extra)
    
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    batch_size = 1

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=None,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)
    
    all_RTs_radar = []
    all_RTs_lidar = []

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        # preprocess the sample
        RTs_radar = []
        RTs_lidar = []

        if mode == 'CR':
            rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR(sample, _config)

        if mode == 'CL':
            rgb_input, lidar_input, rgb_depth_input = preprocess_sample_CL(sample, _config)

        if mode == 'CLR':
            rgb_input, lidar_input, radar_input, rgb_depth_input = preprocess_sample_CLR(sample, _config)

        if mode == 'CRScholler':
            rgb_input, radar_input = preprocess_sample_CRScholler(sample, _config)

        # Run model
        with torch.no_grad():
            for iteration in range(len(models)):
                RTs_temp_lidar = []
                RTs_temp_radar = []
                if mode == 'CR':
                    T_predicted, R_predicted = models[iteration](rgb_input, radar_input, rgb_depth_input)
                    for i in range(len(sample['initial_RT'])):    
                        RT_predicted = torch.mm(tvector2mat(T_predicted[i]), quat2mat(R_predicted[i]))
                        RTs_temp_radar.append(RT_predicted)
                
                    RTs_radar.append(RTs_temp_radar)
                
                if mode == 'CL':
                    T_predicted, R_predicted = models[iteration](rgb_input, lidar_input, rgb_depth_input)
                    for i in range(len(sample['initial_RT'])):
                        RT_predicted = torch.mm(tvector2mat(T_predicted[i]), quat2mat(R_predicted[i]))
                        RTs_temp_lidar.append(RT_predicted)
                    
                    RTs_lidar.append(RTs_temp_lidar)

                if mode == 'CLR':
                    T_predicted_CL, R_predicted_CL, _,_, T_predicted_CR, R_predicted_CR = models[iteration](rgb_input, lidar_input, radar_input, rgb_depth_input)
                    for i in range(len(sample['initial_RT_CL'])):
                        RT_predicted_CL = torch.mm(tvector2mat(T_predicted_CL[i]), quat2mat(R_predicted_CL[i]))
                        RT_predicted_CR = torch.mm(tvector2mat(T_predicted_CR[i]), quat2mat(R_predicted_CR[i]))
                        RTs_temp_lidar.append(RT_predicted_CL)
                        RTs_temp_radar.append(RT_predicted_CR)
                    
                    RTs_lidar.append(RTs_temp_lidar)
                    RTs_radar.append(RTs_temp_radar)

                if mode == 'CRScholler':
                    R_predicted = models[iteration](rgb_input, radar_input)
                    for i in range(len(sample['initial_RT'])):
                        RT_predicted = quat2mat(R_predicted[i])
                        RTs_temp_radar.append(RT_predicted)

                    RTs_radar.append(RTs_temp_radar)

        root_folder = '/home/appuser/data/output/' + mode + '/projections'

        average_matrix_lidar = None
        average_matrix_radar = None

        if mode == 'CR' or mode == 'CRScholler':
            for RTs_radar in RTs_radar[-1]:
                all_RTs_radar.append(RTs_radar)
            if rigid:
                average_matrix_radar = torch.mean(torch.stack(all_RTs_radar), axis=0)
            else:
                average_matrix_radar = all_RTs_radar[-1]
            sample['radar'][0] = sample['radar'][0].cuda()
            radar = sample['radar'][0]
            radar = radar[:3, :]
            tr_error = sample['tr_error_CR'][0]
            rot_error = sample['rot_error_CR'][0]
            save_imgs(os.path.join(root_folder, 'radar'), sample['orig_img'][0], sample['calib'][0], average_matrix_radar, radar, batch_idx, tr_error, rot_error, show=show, save=save)

        if mode == 'CL':
            for RTs_lidar in RTs_lidar[-1]:
                all_RTs_lidar.append(RTs_lidar)
            if rigid:
                average_matrix_lidar = torch.mean(torch.stack(all_RTs_lidar), axis=0)
            else:
                average_matrix_lidar = all_RTs_lidar[-1]
            sample['lidar'][0] = sample['lidar'][0].cuda()
            lidar = sample['lidar'][0]
            lidar = lidar[:3, :]
            tr_error = sample['tr_error_CL'][0]
            rot_error = sample['rot_error_CL'][0]
            save_imgs(os.path.join(root_folder, 'lidar'), sample['orig_img'][0], sample['calib'][0], average_matrix_lidar, lidar, batch_idx, tr_error, rot_error, show=show, save=save)

        if mode == 'CLR':
            for RTs_radar in RTs_radar[-1]:
                all_RTs_radar.append(RTs_radar)
            if rigid:
                average_matrix_radar = torch.mean(torch.stack(all_RTs_radar), axis=0)
            else:
                average_matrix_radar = all_RTs_radar[-1]

            for RTs_lidar in RTs_lidar[-1]:
                all_RTs_lidar.append(RTs_lidar)
            if rigid:
                average_matrix_lidar = torch.mean(torch.stack(all_RTs_lidar), axis=0)
            else:
                average_matrix_lidar = all_RTs_lidar[-1]

            sample['radar'][0] = sample['radar'][0].cuda()
            radar = sample['radar'][0]
            radar = radar[:3, :]
            tr_error = sample['tr_error_CR'][0]
            rot_error = sample['rot_error_CR'][0]
            save_imgs(os.path.join(root_folder, 'radar'), sample['orig_img'][0], sample['calib'][0], average_matrix_radar, radar, batch_idx, tr_error, rot_error, show=show, save=save)

            sample['lidar'][0] = sample['lidar'][0].cuda()
            lidar = sample['lidar'][0]
            lidar = lidar[:3, :]
            tr_error = sample['tr_error_CL'][0]
            rot_error = sample['rot_error_CL'][0]
            save_imgs(os.path.join(root_folder, 'lidar'), sample['orig_img'][0], sample['calib'][0], average_matrix_lidar, lidar, batch_idx, tr_error, rot_error, show=show, save=save)
            
    return average_matrix_radar, average_matrix_lidar

def main(seed):
    args = parse_args()
    ckpt_config = read_checkpoint_config(args.weights[0])
    mode = args.mode or mode_from_config(ckpt_config, 'CLR')
    if args.config is not None:
        _config = read_config(args.config)
    else:
        _config = read_config('/home/appuser/src/calibration/CLRNet/configs/VOD_' + mode + '_config.yaml')

    apply_checkpoint_config(_config, ckpt_config)

    models = []
    for weight in tqdm(args.weights, desc='Loading weights'):
        checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
        apply_checkpoint_config(_config, checkpoint.get('config') or read_checkpoint_config(weight), stage=True)
        if mode == 'CR':
            model = CRNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                       use_reflectance_RADAR=_config['use_reflectance_RADAR'], use_velocity_RADAR = _config['use_velocity_RADAR'], use_time_RADAR=_config['use_time_RADAR'], use_depth=_config['use_depth'],
                       dropout=_config['dropout'], res_num=18)
            
        if mode == 'CL':
            model = CLNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                       use_reflectance_LIDAR=_config['use_reflectance_LIDAR'], use_depth=_config['use_depth'],
                       dropout=_config['dropout'], res_num=18)

        if mode == 'CLR':
            model = CLRNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                       use_reflectance_LIDAR=_config['use_reflectance_LIDAR'], use_depth=_config['use_depth'],
                       use_reflectance_RADAR=_config['use_reflectance_RADAR'], use_velocity_RADAR = _config['use_velocity_RADAR'], use_time_RADAR=_config['use_time_RADAR'],
                       dropout=_config['dropout'], res_num=18)
            
        if mode == "CRScholler":
            model = CRSchollerNet()

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)

    average_RT_radar, average_RT_lidar = run_eval(_config, seed, models=models, mode=mode, show=args.show, save=args.save)

if __name__ == '__main__':
    main(0)