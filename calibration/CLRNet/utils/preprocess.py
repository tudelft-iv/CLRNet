import torch
import torch.nn.functional as F

from CLRNet.utils.utils import (rotate_back, lidar_project_depth, lidar_project_depth_spherical, tvector2mat, quat2mat)

def preprocess_RTs_pairwise(sample):
    RTs = []
    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()
    for idx in range(len(sample['rot_error'])):
        RT = torch.mm(tvector2mat(sample['tr_error'][idx]), quat2mat(sample['rot_error'][idx]))
        RTs.append(RT.clone())
    return RTs

def preprocess_RTs_CLR(sample):
    RTs_CL = []
    RTs_CR = []
    RTs_LR = []
    sample['tr_error_CL'] = sample['tr_error_CL'].cuda()
    sample['rot_error_CL'] = sample['rot_error_CL'].cuda()
    sample['tr_error_CR'] = sample['tr_error_CR'].cuda()
    sample['rot_error_CR'] = sample['rot_error_CR'].cuda()
    sample['tr_error_LR'] = sample['tr_error_LR'].cuda()
    sample['rot_error_LR'] = sample['rot_error_LR'].cuda()
    for idx in range(len(sample['rot_error_CL'])):
        RT_CL = torch.mm(tvector2mat(sample['tr_error_CL'][idx]), quat2mat(sample['rot_error_CL'][idx]))
        RTs_CL.append(RT_CL.clone())
        RT_CR = torch.mm(tvector2mat(sample['tr_error_CR'][idx]), quat2mat(sample['rot_error_CR'][idx]))
        RTs_CR.append(RT_CR.clone())
        RT_LR = torch.mm(tvector2mat(sample['tr_error_LR'][idx]), quat2mat(sample['rot_error_LR'][idx]))
        RTs_LR.append(RT_LR.clone())
    return RTs_CL, RTs_CR, RTs_LR

def preprocess_sample_CL(sample, _config, RT_input = None):
    radar_input = []
    rgb_input = []
    rgb_depth_input = []

    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()

    for idx in range(len(sample['rgb'])):
        sample['lidar'][idx] = sample['lidar'][idx].cuda()
        radar_multi_input = []
        rgb_multi_input = []
        rgb_depth_multi_input = []

        if RT_input is not None:
            if isinstance(RT_input, list):
                RT = RT_input[idx]
            else:
                RT = RT_input
        else:
            RT = torch.mm(tvector2mat(sample['tr_error'][idx]), quat2mat(sample['rot_error'][idx]))

        for i in range(_config['multiple_input']):
            rgb = sample['rgb'][(idx+i) % len(sample['rgb'])].cuda()
            rgb_depth = sample['depth'][(idx+i) % len(sample['rgb'])].cuda()
            lidar = sample['lidar'][(idx+i) % len(sample['rgb'])].cuda()

            lidar_rcs = lidar[3, :]
            lidar = lidar[:3, :]
            if _config['use_reflectance_LIDAR']:
                lidar = torch.cat((lidar, lidar_rcs.unsqueeze(0)), dim=0)

            pc_radar_rotated = rotate_back(lidar, RT)

            if _config['max_depth'] < 80.:
                pc_radar_rotated = pc_radar_rotated[:, pc_radar_rotated[0, :] < _config['max_depth']].clone()

            if _config['spherical_projection']:
                depth_img_radar, _ = lidar_project_depth_spherical(pc_radar_rotated, [1024,2048])

            else:
                depth_img_radar, _ = lidar_project_depth(pc_radar_rotated, sample['calib'][i], [1024,2048])

            depth_img_radar /= _config['max_depth']

            rgb_multi_input.append(rgb)
            radar_multi_input.append(depth_img_radar)
            rgb_depth_multi_input.append(rgb_depth)

        radar_multi_input = torch.stack(radar_multi_input).cuda()
        rgb_multi_input = torch.stack(rgb_multi_input).cuda()
        rgb_depth_multi_input = torch.stack([x for x in rgb_depth_multi_input if x is not None]).cuda()

        radar_multi_input = F.interpolate(radar_multi_input, _config['input_size'], mode="bilinear")
        rgb_multi_input = F.interpolate(rgb_multi_input, _config['input_size'], mode="bilinear")
        if rgb_depth_multi_input.shape[0] > 0:
            rgb_depth_multi_input = F.interpolate(rgb_depth_multi_input, _config['input_size'], mode="bilinear")

        radar_input.append(radar_multi_input)
        rgb_input.append(rgb_multi_input)
        rgb_depth_input.append(rgb_depth_multi_input)

    radar_input = torch.stack(radar_input).cuda()
    rgb_input = torch.stack(rgb_input).cuda()
    rgb_depth_input = torch.stack([x for x in rgb_depth_input if x is not None]).cuda()

    if _config['use_depth'] is False:
        rgb_depth_input = None

    return rgb_input, radar_input, rgb_depth_input

def preprocess_sample_CL_LCC(sample, _config, RT_input = None):
    radar_input = []
    rgb_input = []

    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()

    for idx in range(len(sample['rgb'])):
        sample['lidar'][idx] = sample['lidar'][idx].cuda()

        if RT_input is not None:
            if isinstance(RT_input, list):
                RT = RT_input[idx]
            else:
                RT = RT_input
        else:
            RT = torch.mm(tvector2mat(sample['tr_error'][idx]), quat2mat(sample['rot_error'][idx]))

        rgb = sample['rgb'][idx].cuda()
        lidar = sample['lidar'][idx].cuda()

        lidar_rcs = lidar[3, :]
        lidar = lidar[:3, :]
        if _config['use_reflectance_LIDAR']:
            lidar = torch.cat((lidar, lidar_rcs.unsqueeze(0)), dim=0)

        pc_radar_rotated = rotate_back(lidar, RT)

        if _config['max_depth'] < 80.:
            pc_radar_rotated = pc_radar_rotated[:, pc_radar_rotated[0, :] < _config['max_depth']].clone()

        if _config['spherical_projection']:
            depth_img_radar, _ = lidar_project_depth_spherical(pc_radar_rotated, [1024,2048])

        else:
            depth_img_radar, _ = lidar_project_depth(pc_radar_rotated, sample['calib'][idx], [1024,2048])

        depth_img_radar /= _config['max_depth']

        rgb_input.append(rgb)
        radar_input.append(depth_img_radar)

    radar_input = torch.stack(radar_input).cuda()
    rgb_input = torch.stack(rgb_input).cuda()

    radar_input = F.interpolate(radar_input, _config['input_size'], mode="bilinear")
    rgb_input = F.interpolate(rgb_input, _config['input_size'], mode="bilinear")

    rgb_depth_input = None

    return rgb_input, radar_input, rgb_depth_input

def preprocess_sample_CR(sample, _config, RT_input = None):
    radar_input = []
    rgb_input = []
    rgb_depth_input = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()

    for idx in range(len(sample['rgb'])):
        sample['radar'][idx] = sample['radar'][idx].cuda()
        radar_multi_input = []
        rgb_multi_input = []
        rgb_depth_multi_input = []

        if RT_input is not None:
            if isinstance(RT_input, list):
                RT = RT_input[idx]
            else:
                RT = RT_input
        else:
            RT = torch.mm(tvector2mat(sample['tr_error'][idx]), quat2mat(sample['rot_error'][idx]))

        for i in range(_config['multiple_input']):
            rgb = sample['rgb'][(idx+i) % len(sample['rgb'])].cuda()
            rgb_depth = sample['depth'][(idx+i) % len(sample['rgb'])].cuda()
            radar = sample['radar'][(idx+i) % len(sample['rgb'])].cuda()

            radar_rcs = radar[3, :]
            radar_velocity = radar[5, :]
            radar_time = radar[6, :]
            radar = radar[:3, :]
            if _config['use_reflectance_RADAR']:
                radar = torch.cat((radar, radar_rcs.unsqueeze(0)), dim=0)
            if _config['use_velocity_RADAR']:
                radar = torch.cat((radar, radar_velocity.unsqueeze(0)), dim=0)
            if _config['use_time_RADAR']:
                radar = torch.cat((radar, radar_time.unsqueeze(0)), dim=0)

            pc_radar_rotated = rotate_back(radar, RT)

            if _config['max_depth'] < 80.:
                pc_radar_rotated = pc_radar_rotated[:, pc_radar_rotated[0, :] < _config['max_depth']].clone()

            if _config['spherical_projection']:
                depth_img_radar, _ = lidar_project_depth_spherical(pc_radar_rotated, [1024,2048])

            else:
                depth_img_radar, _ = lidar_project_depth(pc_radar_rotated, sample['calib'][i], [1024,2048])

            depth_img_radar /= _config['max_depth']

            rgb_multi_input.append(rgb)
            radar_multi_input.append(depth_img_radar)
            rgb_depth_multi_input.append(rgb_depth)

        radar_multi_input = torch.stack(radar_multi_input).cuda()
        rgb_multi_input = torch.stack(rgb_multi_input).cuda()
        rgb_depth_multi_input = torch.stack([x for x in rgb_depth_multi_input if x is not None]).cuda()

        radar_multi_input = F.interpolate(radar_multi_input, _config['input_size'], mode="bilinear")
        rgb_multi_input = F.interpolate(rgb_multi_input, _config['input_size'], mode="bilinear")
        if rgb_depth_multi_input.shape[0] > 0:
            rgb_depth_multi_input = F.interpolate(rgb_depth_multi_input, _config['input_size'], mode="bilinear")

        radar_input.append(radar_multi_input)
        rgb_input.append(rgb_multi_input)
        rgb_depth_input.append(rgb_depth_multi_input)

    radar_input = torch.stack(radar_input).cuda()
    rgb_input = torch.stack(rgb_input).cuda()
    rgb_depth_input = torch.stack([x for x in rgb_depth_input if x is not None]).cuda()

    if _config['use_depth'] is False:
        rgb_depth_input = None

    return rgb_input, radar_input, rgb_depth_input

def preprocess_sample_CLR(sample, _config, RT_CL_input=None, RT_CR_input=None):
    lidar_input = []
    radar_input = []
    rgb_input = []
    rgb_depth_input = []

    sample['tr_error_CL'] = sample['tr_error_CL'].cuda()
    sample['rot_error_CL'] = sample['rot_error_CL'].cuda()
    sample['tr_error_CR'] = sample['tr_error_CR'].cuda()
    sample['rot_error_CR'] = sample['rot_error_CR'].cuda()
    sample['tr_error_LR'] = sample['tr_error_LR'].cuda()
    sample['rot_error_LR'] = sample['rot_error_LR'].cuda()

    for idx in range(len(sample['rgb'])):
        lidar_multi_input = []
        radar_multi_input = []
        rgb_multi_input = []
        rgb_depth_multi_input = []

        if RT_CL_input is not None:
            if isinstance(RT_CL_input, list):
                RT_CL = RT_CL_input[idx]
            else:
                RT_CL = RT_CL_input
        else:
            RT_CL = torch.mm(tvector2mat(sample['tr_error_CL'][idx]), quat2mat(sample['rot_error_CL'][idx]))
        if RT_CR_input is not None:
            if isinstance(RT_CR_input, list):
                RT_CR = RT_CR_input[idx]
            else:
                RT_CR = RT_CR_input
        else:
            RT_CR = torch.mm(tvector2mat(sample['tr_error_CR'][idx]), quat2mat(sample['rot_error_CR'][idx]))

        for i in range(_config['multiple_input']):
            rgb = sample['rgb'][(idx+i)% len(sample['rgb'])].cuda()
            rgb_depth = sample['depth'][(idx+i)% len(sample['rgb'])].cuda()
            radar = sample['radar'][(idx+i)% len(sample['rgb'])].cuda()
            lidar = sample['lidar'][(idx+i)% len(sample['rgb'])].cuda()

            if not _config['use_reflectance_LIDAR']:
                lidar = lidar[:3, :]

            radar_rcs = radar[3, :]
            radar_velocity = radar[5, :]
            radar_time = radar[6, :]
            radar = radar[:3, :]
            if _config['use_reflectance_RADAR']:
                radar = torch.cat((radar, radar_rcs.unsqueeze(0)), dim=0)
            if _config['use_velocity_RADAR']:
               radar = torch.cat((radar, radar_velocity.unsqueeze(0)), dim=0)
            if _config['use_time_RADAR']:
               radar = torch.cat((radar, radar_time.unsqueeze(0)), dim=0)

            pc_lidar_rotated = rotate_back(lidar, RT_CL)
            pc_radar_rotated = rotate_back(radar, RT_CR)

            if _config['max_depth'] < 80.:
                pc_lidar_rotated = pc_lidar_rotated[:, pc_lidar_rotated[0, :] < _config['max_depth']].clone()
                pc_radar_rotated = pc_radar_rotated[:, pc_radar_rotated[0, :] < _config['max_depth']].clone()

            if _config['spherical_projection']:
                depth_img_lidar, _ = lidar_project_depth_spherical(pc_lidar_rotated, [1024,2048])
                depth_img_radar, _ = lidar_project_depth_spherical(pc_radar_rotated, [1024,2048])

            else:
                depth_img_lidar, _ = lidar_project_depth(pc_lidar_rotated, sample['calib'][i], [1024,2048])
                depth_img_radar, _ = lidar_project_depth(pc_radar_rotated, sample['calib'][i], [1024,2048])

            depth_img_lidar /= _config['max_depth']
            depth_img_radar /= _config['max_depth']

            rgb_multi_input.append(rgb)
            lidar_multi_input.append(depth_img_lidar)
            radar_multi_input.append(depth_img_radar)
            rgb_depth_multi_input.append(rgb_depth)

        lidar_multi_input = torch.stack(lidar_multi_input).cuda()
        radar_multi_input = torch.stack(radar_multi_input).cuda()
        rgb_multi_input = torch.stack(rgb_multi_input).cuda()
        rgb_depth_multi_input = torch.stack([x for x in rgb_depth_multi_input if x is not None]).cuda()

        lidar_multi_input = F.interpolate(lidar_multi_input, _config['input_size'], mode="bilinear")
        radar_multi_input = F.interpolate(radar_multi_input, _config['input_size'], mode="bilinear")
        rgb_multi_input = F.interpolate(rgb_multi_input, _config['input_size'], mode="bilinear")
        if rgb_depth_multi_input.shape[0] > 0:
            rgb_depth_multi_input = F.interpolate(rgb_depth_multi_input, _config['input_size'], mode="bilinear")

        lidar_input.append(lidar_multi_input)
        radar_input.append(radar_multi_input)
        rgb_input.append(rgb_multi_input)
        rgb_depth_input.append(rgb_depth_multi_input)

    lidar_input = torch.stack(lidar_input).cuda()
    radar_input = torch.stack(radar_input).cuda()
    rgb_input = torch.stack(rgb_input).cuda()
    rgb_depth_input = torch.stack([x for x in rgb_depth_input if x is not None]).cuda()

    if _config['use_depth'] is False:
        rgb_depth_input = None

    return rgb_input, lidar_input, radar_input, rgb_depth_input

def preprocess_sample_CRScholler(sample, _config, RT_CR_input=None):
    radar_input = []
    rgb_input = []

    sample['rot_error'] = sample['rot_error'].cuda()
    sample['tr_error'] = sample['tr_error'].cuda()


    for idx in range(len(sample['rgb'])):
        sample['radar'][idx] = sample['radar'][idx].cuda()
        rgb = sample['rgb'][idx].cuda()
        radar = sample['radar'][idx].cuda()

        if RT_CR_input is not None:
            if isinstance(RT_CR_input, list):
                RT_CR = RT_CR_input[idx]
            else:
                RT_CR = RT_CR_input
        else:
            RT_CR = quat2mat(sample['rot_error'][idx])

        pc_radar_rotated = rotate_back(radar[:3, :], RT_CR)

        depth_img_radar, _ = lidar_project_depth(pc_radar_rotated, sample['calib'][idx], [150,240])

        radar_input.append(depth_img_radar)
        rgb_input.append(rgb)

    radar_input = torch.stack(radar_input).cuda()
    rgb_input = torch.stack(rgb_input).cuda()

    radar_input = F.interpolate(radar_input, _config['input_size'], mode="bilinear")
    rgb_input = F.interpolate(rgb_input, _config['input_size'], mode="bilinear")

    return rgb_input, radar_input

# ---------------------------------------------------------------------------------------------
# Dual-Radar (Arbe Phoenix) support. Points have 5 columns.
# ---------------------------------------------------------------------------------------------

def preprocess_sample_CR_Dual(sample, _config, RT_input = None):
    radar_input = []
    rgb_input = []
    rgb_depth_input = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample['tr_error'] = sample['tr_error'].cuda()
    sample['rot_error'] = sample['rot_error'].cuda()

    for idx in range(len(sample['rgb'])):
        sample['radar'][idx] = sample['radar'][idx].cuda()
        radar_multi_input = []
        rgb_multi_input = []
        rgb_depth_multi_input = []

        if RT_input is not None:
            if isinstance(RT_input, list):
                RT = RT_input[idx]
            else:
                RT = RT_input
        else:
            RT = torch.mm(tvector2mat(sample['tr_error'][idx]), quat2mat(sample['rot_error'][idx]))

        for i in range(_config['multiple_input']):
            rgb = sample['rgb'][(idx+i) % len(sample['rgb'])].cuda()
            rgb_depth = sample['depth'][(idx+i) % len(sample['rgb'])].cuda()
            radar = sample['radar'][(idx+i) % len(sample['rgb'])].cuda()

            # Column 3 is fed as the velocity channel and column 4 as the reflectance channel
            # (the dataset spec lists them as RCS and Doppler, respectively).
            radar_rcs = radar[4, :]
            radar_velocity = radar[3, :]
            radar = radar[:3, :]
            if _config['use_reflectance_RADAR']:
                radar = torch.cat((radar, radar_rcs.unsqueeze(0)), dim=0)
            if _config['use_velocity_RADAR']:
                radar = torch.cat((radar, radar_velocity.unsqueeze(0)), dim=0)
            if _config['use_time_RADAR']:
                #cat on dim 0 with zeros
                radar_time = torch.zeros_like(radar[0, :]).to(device)
                radar = torch.cat((radar, radar_time.unsqueeze(0)), dim=0)

            pc_radar_rotated = rotate_back(radar, RT)

            if _config['max_depth'] < 80.:
                pc_radar_rotated = pc_radar_rotated[:, pc_radar_rotated[0, :] < _config['max_depth']].clone()

            if _config['spherical_projection']:
                depth_img_radar, _ = lidar_project_depth_spherical(pc_radar_rotated, [1024,2048])

            else:
                depth_img_radar, _ = lidar_project_depth(pc_radar_rotated, sample['calib'][i], [1024,2048])

            depth_img_radar /= _config['max_depth']

            rgb_multi_input.append(rgb)
            radar_multi_input.append(depth_img_radar)
            rgb_depth_multi_input.append(rgb_depth)

        radar_multi_input = torch.stack(radar_multi_input).cuda()
        rgb_multi_input = torch.stack(rgb_multi_input).cuda()
        rgb_depth_multi_input = torch.stack([x for x in rgb_depth_multi_input if x is not None]).cuda()

        radar_multi_input = F.interpolate(radar_multi_input, _config['input_size'], mode="bilinear")
        rgb_multi_input = F.interpolate(rgb_multi_input, _config['input_size'], mode="bilinear")
        if rgb_depth_multi_input.shape[0] > 0:
            rgb_depth_multi_input = F.interpolate(rgb_depth_multi_input, _config['input_size'], mode="bilinear")

        radar_input.append(radar_multi_input)
        rgb_input.append(rgb_multi_input)
        rgb_depth_input.append(rgb_depth_multi_input)

    radar_input = torch.stack(radar_input).cuda()
    rgb_input = torch.stack(rgb_input).cuda()
    rgb_depth_input = torch.stack([x for x in rgb_depth_input if x is not None]).cuda()

    if _config['use_depth'] is False:
        rgb_depth_input = None

    return rgb_input, radar_input, rgb_depth_input
