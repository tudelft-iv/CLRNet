import math

import numpy as np
import torch
from matplotlib import cm
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import os
import matplotlib.pyplot as plt
import yaml

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def read_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
            return config_data
        except yaml.YAMLError as exc:
            print(exc)

def preprocess_RTs(sample):
    RTs = []
    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()
    for idx in range(len(sample['rot_error'])):
        RT = torch.mm(tvector2mat(sample['tr_error'][idx]), quat2mat(sample['rot_error'][idx]))
        RTs.append(RT.clone().inverse())
    return RTs

def quatmultiply(q, r, device = 'cuda'):
    """
    Batch quaternion multiplication
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]
        r (torch.Tensor/np.ndarray): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = torch.zeros(q.shape[0], 4, device=device)
    elif isinstance(q, np.ndarray):
        t = np.zeros(q.shape[0], 4)
    else:
        raise TypeError("Type not supported")
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t

def quatinv(q):
    """
    Batch quaternion inversion
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]

    Returns:
        torch.Tensor/np.ndarray: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError("Type not supported")
    t *= -1
    t[:, 0] *= -1
    return t

def quaternion_distance(q, r, device):
    """
    Batch quaternion distances, used as loss
    Args:
        q (torch.Tensor): shape=[Nx4]
        r (torch.Tensor): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[N]
    """
    t = quatmultiply(q, quatinv(r), device)
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))

def print_results(RTs, results_path):

    plt_size = 10.5

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    errors_r = []
    errors_t = []
    errors_rpy = []
    errors_t2 = []
    for i in range(len(RTs)):
        T_composed = RTs[i][:3, 3]
        R_composed = quaternion_from_matrix(RTs[i])
        rpy_composed = mat2xyzrpy(RTs[i])[3:]
        t2_composed = RTs[i][:3, 3]

        errors_t.append(T_composed.norm().item())
        errors_r.append(quaternion_distance(R_composed.unsqueeze(0),
                                                            torch.tensor([1., 0., 0., 0.], device=R_composed.device).unsqueeze(0),
                                                            R_composed.device))
        errors_rpy.append(rpy_composed)
        errors_t2.append(t2_composed)


        if not isinstance(errors_t[i], torch.Tensor):
            errors_t[i] = torch.tensor(errors_t[i])

        if not isinstance(errors_r[i], torch.Tensor):
            errors_r[i] = torch.tensor(errors_r[i])

        if not isinstance(errors_rpy[i], torch.Tensor):
            errors_rpy[i] = torch.tensor(errors_rpy[i])
        
        if not isinstance(errors_t2[i], torch.Tensor):
            errors_t2[i] = torch.tensor(errors_t2[i])
        
        errors_t[i] = errors_t[i].clone().detach().abs() * 100
        errors_r[i] = errors_r[i].clone().detach().abs() * (180.0 / 3.141592)
        errors_rpy[i] = errors_rpy[i].clone().detach().abs() * (180.0 / 3.141592)
        errors_t2[i] = errors_t2[i].clone().detach().abs() * 100

    # Convert lists to tensors
    errors_r = torch.stack(errors_r)
    errors_t = torch.stack(errors_t)
    errors_rpy = torch.stack(errors_rpy)
    errors_t2 = torch.stack(errors_t2)

    # Shape (N, 3): one row per sample, columns X/Y/Z (resp. roll/pitch/yaw).
    t2_x, t2_y, t2_z = errors_t2[:, 0], errors_t2[:, 1], errors_t2[:, 2]
    rpy_r, rpy_p, rpy_yaw = errors_rpy[:, 0], errors_rpy[:, 1], errors_rpy[:, 2]

    errorst2_mean = (t2_x.mean() + t2_y.mean() + t2_z.mean()) / 3
    errorst2_median = (t2_x.median() + t2_y.median() + t2_z.median()) / 3
    errorst2_std = (t2_x.std() + t2_y.std() + t2_z.std()) / 3

    errorrpy_mean = (rpy_r.mean() + rpy_p.mean() + rpy_yaw.mean()) / 3
    errorrpy_median = (rpy_r.median() + rpy_p.median() + rpy_yaw.median()) / 3
    errorrpy_std = (rpy_r.std() + rpy_p.std() + rpy_yaw.std()) / 3

    print(f"Mean Translation Error: {errors_t.mean():.4f}cm "
            f"     Mean Rotation Error: {errors_r.mean():.4f}°")
    print(f"Median Translation Error: {errors_t.median():.4f}cm "
            f"     Median Rotation Error: {errors_r.median():.4f}°")
    print(f"Std. Translation Error: {errors_t.std():.4f}cm "
            f"     Std. Rotation Error: {errors_r.std():.4f}°\n")
    
    print(f"Mean Translation X Error: {t2_x.mean():.4f}cm "
            f"     Median Translation X Error: {t2_x.median():.4f}cm "
            f"     Std. Translation X Error: {t2_x.std():.4f}cm")
    print(f"Mean Translation Y Error: {t2_y.mean():.4f}cm "
            f"     Median Translation Y Error: {t2_y.median():.4f}cm "
            f"     Std. Translation Y Error: {t2_y.std():.4f}cm")
    print(f"Mean Translation Z Error: {t2_z.mean():.4f}cm "
            f"     Median Translation Z Error: {t2_z.median():.4f}cm "
            f"     Std. Translation Z Error: {t2_z.std():.4f}cm\n")
    print(f"Mean Translation Error: {errorst2_mean:.4f}cm "
            f"     Median Translation Error: {errorst2_median:.4f}cm "
            f"     Std. Translation Error: {errorst2_std:.4f}cm\n")
    
    print(f"Mean Rotation Roll Error: {rpy_r.mean(): .4f}°"
            f"     Median Rotation Roll Error: {rpy_r.median():.4f}°"
            f"     Std. Rotation Roll Error: {rpy_r.std():.4f}°")
    print(f"Mean Rotation Pitch Error: {rpy_p.mean(): .4f}°"
            f"     Median Rotation Pitch Error: {rpy_p.median():.4f}°"
            f"     Std. Rotation Pitch Error: {rpy_p.std():.4f}°")
    print(f"Mean Rotation Yaw Error: {rpy_yaw.mean(): .4f}°"
            f"     Median Rotation Yaw Error: {rpy_yaw.median():.4f}°"
            f"     Std. Rotation Yaw Error: {rpy_yaw.std():.4f}°\n\n")
    print(f"Mean Rotation Error: {errorrpy_mean:.4f}°"
            f"     Median Rotation Error: {errorrpy_median:.4f}°"
            f"     Std. Rotation Error: {errorrpy_std:.4f}°\n\n\n")
    

    with open(os.path.join(results_path, 'results.txt'),
                'a', encoding='utf-8') as f:
        f.write("Translation Error && Rotation Error:\n")
        f.write(f"Mean Translation Error: {errors_t.mean():.4f}cm "
                f"     Mean Rotation Error: {errors_r.mean():.4f}°\n")
        f.write(f"Median Translation Error: {errors_t.median():.4f}cm "
                f"     Median Rotation Error: {errors_r.median():.4f}°\n")
        f.write(f"Std. Translation Error: {errors_t.std():.4f}cm "
                f"     Std. Rotation Error: {errors_r.std():.4f}°\n\n")
        
        f.write("Translation Error XYZ:\n")
        f.write(f"Mean Translation X Error: {t2_x.mean():.4f}cm "
                f"     Median Translation X Error: {t2_x.median():.4f}cm "
                f"     Std. Translation X Error: {t2_x.std():.4f}cm\n")
        f.write(f"Mean Translation Y Error: {t2_y.mean():.4f}cm "
                f"     Median Translation Y Error: {t2_y.median():.4f}cm "
                f"     Std. Translation Y Error: {t2_y.std():.4f}cm\n")
        f.write(f"Mean Translation Z Error: {t2_z.mean():.4f}cm "
                f"     Median Translation Z Error: {t2_z.median():.4f}cm "
                f"     Std. Translation Z Error: {t2_z.std():.4f}cm\n\n")
        f.write(f"Mean Translation Error: {errorst2_mean:.4f}cm "
                f"     Median Translation Error: {errorst2_median:.4f}cm "
                f"     Std. Translation Error: {errorst2_std:.4f}cm\n\n")
    
        
        f.write("Rotation Error RPY:\n")
        f.write(f"Mean Rotation Roll Error: {rpy_r.mean(): .4f}°"
                f"     Median Rotation Roll Error: {rpy_r.median():.4f}°"
                f"     Std. Rotation Roll Error: {rpy_r.std():.4f}°\n")
        f.write(f"Mean Rotation Pitch Error: {rpy_p.mean(): .4f}°"
                f"     Median Rotation Pitch Error: {rpy_p.median():.4f}°"
                f"     Std. Rotation Pitch Error: {rpy_p.std():.4f}°\n")
        f.write(f"Mean Rotation Yaw Error: {rpy_yaw.mean(): .4f}°"
                f"     Median Rotation Yaw Error: {rpy_yaw.median():.4f}°"
                f"     Std. Rotation Yaw Error: {rpy_yaw.std():.4f}°\n\n\n")
        f.write(f"Mean Rotation Error: {errorrpy_mean:.4f}°"
                f"     Median Rotation Error: {errorrpy_median:.4f}°"
                f"     Std. Rotation Error: {errorrpy_std:.4f}°\n\n\n")
        
    # errors_t = errors_t.cpu().numpy()
    # # errors_t = np.sort(errors_t, axis=0)[:-10]
    # errors_t = np.sort(errors_t, axis=0)
    # plt.hist(errors_t / 100, bins=50)

    # plt.xlabel('Absolute Translation Error (m)')
    # plt.ylabel('Number of instances')

    # plt.xticks(fontproperties='DejaVu Sans', size=plt_size)
    # plt.yticks(fontproperties='DejaVu Sans', size=plt_size)

    # plt.savefig(os.path.join(results_path, 'translation_error_distribution.png'))
    # plt.close('all')

    # errors_r = errors_r.cpu().numpy()
    # # errors_r = np.sort(errors_r, axis=0)[:-10]
    # errors_r = np.sort(errors_r, axis=0)
    # plt.hist(errors_r, bins=50)

    # plt.xlabel('Absolute Rotation Error (°)')
    # plt.ylabel('Number of instances')
    # plt.xticks(fontproperties='DejaVu Sans', size=plt_size)
    # plt.yticks(fontproperties='DejaVu Sans', size=plt_size)
    # plt.savefig(os.path.join(results_path, 'rotation_error_distribution.png'))
    # plt.close('all')

def write_results_to_txt(output, errors_t, errors_r, errors_t2, errors_rpy, i, message=None):
    if not os.path.exists(output):
        os.makedirs(output)
    with open(os.path.join(output, 'results.txt'),
                  'a', encoding='utf-8') as f:
        if message is not None:
            f.write(f"{message} \n ")

        # One 3-vector per test frame: stack to (N, 3) and take the axis columns.
        t2 = torch.stack(list(errors_t2[i])) if not isinstance(errors_t2[i], torch.Tensor) else errors_t2[i]
        rpy = torch.stack(list(errors_rpy[i])) if not isinstance(errors_rpy[i], torch.Tensor) else errors_rpy[i]
        t2_x, t2_y, t2_z = t2[:, 0], t2[:, 1], t2[:, 2]
        rpy_r, rpy_p, rpy_yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]

        errorst2_mean = (t2_x.mean() + t2_y.mean() + t2_z.mean()) / 3
        errorst2_median = (t2_x.median() + t2_y.median() + t2_z.median()) / 3
        errorst2_std = (t2_x.std() + t2_y.std() + t2_z.std()) / 3

        errorrpy_mean = (rpy_r.mean() + rpy_p.mean() + rpy_yaw.mean()) / 3
        errorrpy_median = (rpy_r.median() + rpy_p.median() + rpy_yaw.median()) / 3
        errorrpy_std = (rpy_r.std() + rpy_p.std() + rpy_yaw.std()) / 3

        f.write(f"Iteration {i}: \n")
        f.write("Translation Error && Rotation Error:\n")
        f.write(f"Iteration {i}: \tMean Translation Error: {errors_t[i].mean():.4f} cm "
                f"     Mean Rotation Error: {errors_r[i].mean():.4f} °\n")
        f.write(f"Iteration {i}: \tMedian Translation Error: {errors_t[i].median():.4f} cm "
                f"     Median Rotation Error: {errors_r[i].median():.4f} °\n")
        f.write(f"Iteration {i}: \tStd. Translation Error: {errors_t[i].std():.4f} cm "
                f"     Std. Rotation Error: {errors_r[i].std():.4f} °\n\n")

        # translation xyz
        f.write("Translation Error XYZ:\n")
        f.write(f"Iteration {i}: \tMean Translation X Error: {t2_x.mean():.4f} cm "
                f"     Median Translation X Error: {t2_x.median():.4f} cm "
                f"     Std. Translation X Error: {t2_x.std():.4f} cm \n")
        f.write(f"Iteration {i}: \tMean Translation Y Error: {t2_y.mean():.4f} cm "
                f"     Median Translation Y Error: {t2_y.median():.4f} cm "
                f"     Std. Translation Y Error: {t2_y.std():.4f} cm \n")
        f.write(f"Iteration {i}: \tMean Translation Z Error: {t2_z.mean():.4f} cm "
                f"     Median Translation Z Error: {t2_z.median():.4f} cm "
                f"     Std. Translation Z Error: {t2_z.std():.4f} cm \n\n")
        f.write(f"Iteration {i}: \tMean Translation Error: {errorst2_mean:.4f} cm "
                f"     Median Translation Error: {errorst2_median:.4f} cm "
                f"     Std. Translation Error: {errorst2_std:.4f} cm \n\n")

        # rotation rpy
        f.write("Rotation Error RPY:\n")
        f.write(f"Iteration {i}: \tMean Rotation Roll Error: {rpy_r.mean(): .4f} °"
                f"     Median Rotation Roll Error: {rpy_r.median():.4f} °"
                f"     Std. Rotation Roll Error: {rpy_r.std():.4f} °\n")
        f.write(f"Iteration {i}: \tMean Rotation Pitch Error: {rpy_p.mean(): .4f} °"
                f"     Median Rotation Pitch Error: {rpy_p.median():.4f} °"
                f"     Std. Rotation Pitch Error: {rpy_p.std():.4f} °\n")
        f.write(f"Iteration {i}: \tMean Rotation Yaw Error: {rpy_yaw.mean(): .4f} °"
                f"     Median Rotation Yaw Error: {rpy_yaw.median():.4f} °"
                f"     Std. Rotation Yaw Error: {rpy_yaw.std():.4f} °\n\n")
        f.write(f"Iteration {i}: \tMean Rotation Error: {errorrpy_mean:.4f} °"
                f"     Median Rotation Error: {errorrpy_median:.4f} °"
                f"     Std. Rotation Error: {errorrpy_std:.4f} °\n\n\n")

def plot_figures(errors_r, errors_t, errors_t2, errors_rpy, mis_calib_input, results_path, N=100):
    plt_size = 10.5
    plt.rcParams['axes.unicode_minus'] = False
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for i in range(len(errors_t2)):
        errors_t2[i] = torch.stack(errors_t2[i]).abs() / 100
        errors_rpy[i] = torch.stack(errors_rpy[i]).abs()

    plot_x = np.zeros((mis_calib_input.shape[0], 2))
    plot_x[:, 0] = mis_calib_input[:, 0].cpu().numpy()
    plot_x[:, 1] = errors_t2[-1][:, 0].cpu().numpy()
    plot_x = plot_x[np.lexsort(plot_x[:, ::-1].T)]

    plot_y = np.zeros((mis_calib_input.shape[0], 2))
    plot_y[:, 0] = mis_calib_input[:, 1].cpu().numpy()
    plot_y[:, 1] = errors_t2[-1][:, 1].cpu().numpy()
    plot_y = plot_y[np.lexsort(plot_y[:, ::-1].T)]

    plot_z = np.zeros((mis_calib_input.shape[0], 2))
    plot_z[:, 0] = mis_calib_input[:, 2].cpu().numpy()
    plot_z[:, 1] = errors_t2[-1][:, 2].cpu().numpy()
    plot_z = plot_z[np.lexsort(plot_z[:, ::-1].T)]

    N_interval = plot_x.shape[0] // N
    plot_x = plot_x[::N_interval]
    plot_y = plot_y[::N_interval]
    plot_z = plot_z[::N_interval]

    plt.plot(plot_x[:, 0], plot_x[:, 1], c='red', label='X')
    plt.plot(plot_y[:, 0], plot_y[:, 1], c='blue', label='Y')
    plt.plot(plot_z[:, 0], plot_z[:, 1], c='green', label='Z')
    # plt.legend(loc='best')

    plt.xlabel('Miscalibration (m)')
    plt.ylabel('Absolute Error (m)')
    plt.legend(loc='best')

    plt.xticks(fontproperties='DejaVu Sans', size=plt_size)
    plt.yticks(fontproperties='DejaVu Sans', size=plt_size)

    plt.savefig(os.path.join(results_path, 'xyz_plot.png'))
    plt.close('all')

    errors_t = errors_t[-1].numpy()
    errors_t = np.sort(errors_t, axis=0)[:-10]
    plt.hist(errors_t / 100, bins=50)

    plt.xlabel('Absolute Translation Error (m)')
    plt.ylabel('Number of instances')

    plt.xticks(fontproperties='DejaVu Sans', size=plt_size)
    plt.yticks(fontproperties='DejaVu Sans', size=plt_size)

    plt.savefig(os.path.join(results_path, 'translation_error_distribution.png'))
    plt.close('all')

    plot_pitch = np.zeros((mis_calib_input.shape[0], 2))
    plot_pitch[:, 0] = mis_calib_input[:, 3].cpu().numpy() * (180.0 / 3.141592)
    plot_pitch[:, 1] = errors_rpy[-1][:, 1].cpu().numpy()
    plot_pitch = plot_pitch[np.lexsort(plot_pitch[:, ::-1].T)]

    plot_yaw = np.zeros((mis_calib_input.shape[0], 2))
    plot_yaw[:, 0] = mis_calib_input[:, 4].cpu().numpy() * (180.0 / 3.141592)
    plot_yaw[:, 1] = errors_rpy[-1][:, 2].cpu().numpy()
    plot_yaw = plot_yaw[np.lexsort(plot_yaw[:, ::-1].T)]

    plot_roll = np.zeros((mis_calib_input.shape[0], 2))
    plot_roll[:, 0] = mis_calib_input[:, 5].cpu().numpy() * (180.0 / 3.141592)
    plot_roll[:, 1] = errors_rpy[-1][:, 0].cpu().numpy()
    plot_roll = plot_roll[np.lexsort(plot_roll[:, ::-1].T)]

    N_interval = plot_roll.shape[0] // N
    plot_pitch = plot_pitch[::N_interval]
    plot_yaw = plot_yaw[::N_interval]
    plot_roll = plot_roll[::N_interval]

    plt.plot(plot_yaw[:, 0], plot_yaw[:, 1], c='red', label='Yaw(Y)')
    plt.plot(plot_pitch[:, 0], plot_pitch[:, 1], c='blue', label='Pitch(X)')
    plt.plot(plot_roll[:, 0], plot_roll[:, 1], c='green', label='Roll(Z)')
    plt.xlabel('Miscalibration (°)')
    plt.ylabel('Absolute Error (°)')
    plt.legend(loc='best')

    plt.xticks(fontproperties='DejaVu Sans', size=plt_size)
    plt.yticks(fontproperties='DejaVu Sans', size=plt_size)
    plt.savefig(os.path.join(results_path, 'rpy_plot.png'))
    plt.close('all')

    errors_r = errors_r[-1].numpy()
    errors_r = np.sort(errors_r, axis=0)[:-10] 
    plt.hist(errors_r, bins=50)

    plt.xlabel('Absolute Rotation Error (°)')
    plt.ylabel('Number of instances')

    plt.xticks(fontproperties='DejaVu Sans', size=plt_size)
    plt.yticks(fontproperties='DejaVu Sans', size=plt_size)
    plt.savefig(os.path.join(results_path, 'rotation_error_distribution.png'))
    plt.close('all')

def compose_poses(translations, rotations):
    final_transformation = torch.eye(4, device=rotations[0].device)

    for translation, rotation in zip(translations, rotations):
        translation_matrix = tvector2mat(translation)
        rotation_matrix = quat2mat(rotation)
        matrix = torch.mm(translation_matrix, rotation_matrix)

        final_transformation = final_transformation @ matrix

    return final_transformation

def lidar_to_spherical(x, y, z):
    # Convert Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, radius)
    azimuth = torch.atan2(y, x)  # Angle in the xy-plane
    elevation = torch.atan2(z, torch.sqrt(x**2 + y**2))  # Angle from the xy-plane
    radius = torch.sqrt(x**2 + y**2 + z**2)  # Distance from the origin
    return azimuth, elevation, radius

def spherical_to_image(azimuth, elevation, img_width, img_height):
    # Normalize azimuth and elevation to [0, 1]
    azimuth_norm = (azimuth + np.pi) / (2 * np.pi)  # Map azimuth from [-pi, pi] to [0, 1]
    elevation_norm = (elevation + (np.pi / 2)) / np.pi  # Map elevation from [-pi/2, pi/2] to [0, 1]
    
    # Map normalized coordinates to image dimensions
    u = (azimuth_norm * img_width).long()
    v = ((1 - elevation_norm) * img_height).long()  # Flip vertically
    
    # Clip to ensure indices are within image boundaries
    u = torch.clamp(u, 0, img_width - 1)
    v = torch.clamp(v, 0, img_height - 1)
    
    return u, v

def lidar_project_depth_spherical(pointcloud, img_shape):
    device = pointcloud.device
    # Assuming point cloud has shape (C, N) where N is the number of points and C is the number of channels
    pc_xyz = pointcloud[:3, :].to(device)
    additional_channels = pointcloud[3:, :].to(device)

    x = pc_xyz[0, :]
    y = pc_xyz[1, :]
    z = pc_xyz[2, :]
    
    # Convert to spherical coordinates
    azimuth, elevation, radius = lidar_to_spherical(x, y, z)
    
    # Project to image coordinates
    mask = (radius > 0) & (radius <= 35)
    u, v = spherical_to_image(azimuth[mask], elevation[mask], img_shape[1], img_shape[0])
    radius = radius[mask]
    additional_channels = additional_channels[:, mask]

    # Empty clouds (zero-length VoD lidar files, or no points within range) give a blank depth image.
    if radius.numel() == 0:
        depth_img = torch.zeros((pointcloud.shape[0] - 2, img_shape[0], img_shape[1]),
                                dtype=torch.float32, device=device)
        return depth_img, torch.stack((u, v))

    min_radius = radius.min()
    max_radius = radius.max()
    # A single surviving point (or a degenerate range) would divide by zero.
    radius_normalized = (max_radius - radius) / torch.clamp(max_radius - min_radius, min=1e-6)

    # Scale normalized radius to [0, 255]
    depth_values = (radius_normalized * 255).to(torch.float32)    

    depth_img = torch.zeros((img_shape[0], img_shape[1], pointcloud.shape[0] - 2), dtype=torch.float32, device=device)
    depth_img[v, u, 0] = depth_values

    # Add additional channels to the depth image
    num_additional_channels = additional_channels.shape[0]
    if num_additional_channels > 0:
        for i in range(num_additional_channels):
            depth_img[v, u, i + 1] = additional_channels[i, :]

    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, torch.stack((u, v))

def lidar_project_depth(pointcloud, intrinsic, img_shape, zbuffer=False):
    """Project a point cloud to a depth image.

    With `zbuffer=False` (default) points falling on the same pixel are written in array order.
    `zbuffer=True` keeps the nearest point per pixel; it is used only for visualisation.
    """
    device = pointcloud.device
    # Assuming point cloud has shape (C, N) where N is the number of points and C is the number of channels
    pc_xyz = pointcloud[:3, :].to(device)
    intrinsic = intrinsic.to(device).to(pc_xyz.dtype)
    additional_channels = pointcloud[3:, :].to(device)

    projected_points = torch.matmul(intrinsic, pc_xyz)

    projected_points[:2, :] /= projected_points[2:, :]
    pcl_uv = projected_points[:2, :].round().long()
    pcl_z = pc_xyz[2, :]

    mask = (pcl_uv[0, :] >= 0) & (pcl_uv[0, :] < img_shape[1]) & \
           (pcl_uv[1, :] >= 0) & (pcl_uv[1, :] < img_shape[0]) & \
           (pcl_z > 0)
    pcl_uv = pcl_uv[:, mask]
    pcl_z = pcl_z[mask]
    additional_channels = additional_channels[:, mask]

    if zbuffer and pcl_z.numel() > 0:
        # Write farthest first so the nearest return lands last and survives the scatter below.
        order = torch.argsort(pcl_z, descending=True)
        pcl_uv = pcl_uv[:, order]
        pcl_z = pcl_z[order]
        additional_channels = additional_channels[:, order]

    depth_img = torch.zeros((img_shape[0], img_shape[1], pointcloud.shape[0]-2), dtype=torch.float32, device=device)
    depth_img[pcl_uv[1], pcl_uv[0], 0] = pcl_z

    # Add additional channels to the depth image
    num_additional_channels = additional_channels.shape[0]
    if num_additional_channels > 0:
        for i in range(num_additional_channels):
            depth_img[pcl_uv[1], pcl_uv[0], i + 1] = additional_channels[i, :]

    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv

def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    orig_pc = PC.clone()
    if PC.shape[0] == 3:
        ones_row = torch.ones(1, PC.shape[1],device=PC.device)
        PC = torch.cat((PC, ones_row), dim=0) # x, y, z, 1 (homogeneous coordinates)
    else:
        PC = PC[:4, :] # x, y, z, _
        PC[3, :] = 1. # x, y, z, 1 (homogeneous coordinates)
    PC = torch.mm(RT, PC)
    orig_pc[:3, :] = PC[:3, :] #update x,y,z

    return orig_pc

def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (np.array/list): can be either:
            * (list) euler angles of the rotation part, in this case T cannot be None
            * (np.array shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (np.array shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (np.array/list): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    return rotate_points_torch(PC, R, T, inverse=True)

def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    return rotate_points_torch(PC_ROTATED, R, T, inverse=False)

def rotation_matrix_from_vector(rotation_vector):
    """
    Convert a rotation vector to a rotation matrix using Rodrigues' rotation formula.

    Parameters:
        rotation_vector (array-like): 3-element rotation vector representing axis-angle rotation.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle if angle != 0 else np.array([1, 0, 0])  # Normalize axis
    skew_symmetric = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * skew_symmetric + (1 - np.cos(angle)) * np.dot(skew_symmetric, skew_symmetric)
    return rotation_matrix

def rotation_vector_from_matrix(rotation_matrix):
    """
    Convert a rotation matrix to a rotation vector using axis-angle representation.

    Parameters:
        rotation_matrix (numpy.ndarray or torch.Tensor): 3x3 rotation matrix.

    Returns:
        numpy.ndarray or torch.Tensor: 3-element rotation vector representing axis-angle rotation.
    """
    if isinstance(rotation_matrix, np.ndarray):
        trace = np.trace(rotation_matrix)
        arccos = np.arccos
        sin = np.sin
        array = np.array
    elif isinstance(rotation_matrix, torch.Tensor):
        trace = torch.trace(rotation_matrix)
        arccos = torch.arccos
        sin = torch.sin
        array = torch.tensor
        
        # Ensure all tensors are on the same device as the input matrix
        device = rotation_matrix.device
    else:
        raise TypeError("rotation_matrix must be a numpy.ndarray or torch.Tensor")

    angle = arccos((trace - 1) / 2.0)

    if isinstance(rotation_matrix, torch.Tensor):
        axis = array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                      rotation_matrix[0, 2] - rotation_matrix[2, 0],
                      rotation_matrix[1, 0] - rotation_matrix[0, 1]], device=device)
    else:
        axis = array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                      rotation_matrix[0, 2] - rotation_matrix[2, 0],
                      rotation_matrix[1, 0] - rotation_matrix[0, 1]])

    axis /= (2 * sin(angle))
    rotation_vector = axis * angle
    return rotation_vector

def euler_to_quaternion(euler):
    """
    Convert Euler angles to quaternion representation.

    Parameters:
        euler (numpy.ndarray): Array of Euler angles [phi, theta, psi].

    Returns:
        numpy.ndarray: Quaternion representation [q0, q1, q2, q3].
    """

    phi, theta, psi = euler / 2
    cos_phi_2 = np.cos(phi)
    sin_phi_2 = np.sin(phi)
    cos_theta_2 = np.cos(theta)
    sin_theta_2 = np.sin(theta)
    cos_psi_2 = np.cos(psi)
    sin_psi_2 = np.sin(psi)

    q0 = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2
    q1 = sin_phi_2 * cos_theta_2 * cos_psi_2 - cos_phi_2 * sin_theta_2 * sin_psi_2
    q2 = cos_phi_2 * sin_theta_2 * cos_psi_2 + sin_phi_2 * cos_theta_2 * sin_psi_2
    q3 = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * sin_theta_2 * cos_psi_2

    return np.array([q0, q1, q2, q3])

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles.

    Parameters:
        q (numpy.ndarray): Quaternion representation [q0, q1, q2, q3].

    Returns:
        numpy.ndarray: Array of Euler angles [phi, theta, psi].
    """
    q0, q1, q2, q3 = q
    phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
    psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return np.array([phi, theta, psi])

def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrix using PyTorch tensors.
    
    Args:
    euler_angles (torch.Tensor): Euler angles in radians, shape (3,).

    Returns:
    torch.Tensor: Rotation matrix, shape (3, 3).
    """
    roll, pitch, yaw = euler_angles
    
    # Compute rotation matrices for each axis
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(roll), -torch.sin(roll)],
                        [0, torch.sin(roll), torch.cos(roll)]], dtype=euler_angles.dtype)
    
    R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                        [0, 1, 0],
                        [-torch.sin(pitch), 0, torch.cos(pitch)]], dtype=euler_angles.dtype)
    
    R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                        [torch.sin(yaw), torch.cos(yaw), 0],
                        [0, 0, 1]], dtype=euler_angles.dtype)
    
    # Compute the final rotation matrix
    return torch.mm(R_z, torch.mm(R_y, R_x))

def transformation_matrix(rotation_matrix, translation_vector):
    """
    Create a 4x4 transformation matrix from rotation matrix and translation vector.

    Args:
    rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
    translation_vector (array-like): Translation vector, shape (3,).

    Returns:
    numpy.ndarray: 4x4 transformation matrix.
    """
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation_vector
    return transformation

def invert_4x4_matrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix")

    # Split the matrix into rotation and translation components
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Invert the rotation component
    inv_rotation = rotation.T

    # Calculate the inverse translation component
    inv_translation = -np.dot(inv_rotation, translation)

    # Combine the inverted rotation and translation to form the inverted matrix
    inverted_matrix = np.zeros_like(matrix)
    inverted_matrix[:3, :3] = inv_rotation
    inverted_matrix[:3, 3] = inv_translation
    inverted_matrix[3, 3] = 1

    return inverted_matrix

def invert_quaternion(q):
    """
    Inverts a quaternion represented as a PyTorch tensor.
    Assumes the quaternion is normalized.
    
    Args:
    q (torch.Tensor): A tensor of shape (..., 4) representing the quaternion(s).
    
    Returns:
    torch.Tensor: The inverted quaternion(s).
    """
    # Ensure the quaternion is normalized
    q = q / q.norm(dim=-1, keepdim=True)
    
    # Compute the conjugate of the quaternion
    q_conjugate = q.clone()
    q_conjugate[..., 1:] = -q_conjugate[..., 1:]
    
    return q_conjugate

def invert_translation(t):
    """
    Inverts a translation vector represented as a PyTorch tensor.
    
    Args:
    t (torch.Tensor): A tensor of shape (..., 3) representing the translation vector(s).
    
    Returns:
    torch.Tensor: The inverted translation vector(s).
    """
    return -t

def invert_pose(R, T):
    """
    Inverts a 3D pose given by a rotation vector and translation vector.

    Parameters:
        R (numpy.ndarray): Rotation vector representing the rotation.
        T (numpy.ndarray): Translation vector.

    Returns:
        tuple: Tuple containing the inverted rotation vector and translation vector.
    """
    rotation_matrix = rotation_matrix_from_vector(R)
    inv_rotation_matrix = rotation_matrix.T
    inv_translation = np.dot(-inv_rotation_matrix, T)
    inv_rotation_vector = rotation_vector_from_matrix(inv_rotation_matrix)
    
    return inv_rotation_vector, inv_translation


# def merge_inputs(queries):
#     point_clouds = []
#     imgs = []
#     reflectances = []
#     returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
#                if key != 'point_cloud' and key != 'rgb' and key != 'reflectance'}
#     for input in queries:
#         point_clouds.append(input['point_cloud'])
#         imgs.append(input['rgb'])
#         if 'reflectance' in input:
#             reflectances.append(input['reflectance'])
#     returns['point_cloud'] = point_clouds
#     returns['rgb'] = imgs
#     if len(reflectances) > 0:
#         returns['reflectance'] = reflectances
#     return returns

def merge_inputs(queries):
    lidar = []
    camera = []
    radar = []
    point_cloud = []
    depth = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'lidar' and key != 'rgb' and key != 'radar' and key != 'point_cloud' and key != 'depth'}
    for input in queries:
        if 'depth' in input:
            depth.append(input['depth'])
        if 'lidar' in input:
            lidar.append(input['lidar'])
        if 'rgb' in input:
            camera.append(input['rgb'])
        if 'radar' in input:
            radar.append(input['radar'])
        if 'point_cloud' in input:
            point_cloud.append(input['point_cloud'])
    returns['rgb'] = camera
    if len(radar) > 0:
        returns['radar'] = radar
    if len(lidar) > 0:
        returns['lidar'] = lidar
    if len(point_cloud) > 0:
        returns['point_cloud'] = point_cloud
    if len(depth) > 0:
        returns['depth'] = depth
    return returns

def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()

def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat

def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat

def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)

def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT

def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.autumn(lidar)
    lidar_color[:, :, 3] = 0.9
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    #io.imshow(blended_img)
    #io.show()
    #plt.figure()
    #plt.imshow(blended_img)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img
