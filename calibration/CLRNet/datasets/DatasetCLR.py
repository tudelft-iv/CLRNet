import csv
import os
import cv2

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from CLRNet.utils.utils import invert_pose, quat2mat, quaternion_from_matrix, quaternion_to_euler, euler_to_quaternion, quaternion_from_matrix, rotate_back

from collections import namedtuple
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Depth-Anything-V2")))
from depth_anything_v2.dpt import DepthAnythingV2


def depth_anything_weights(encoder):
    """Depth-Anything-V2 weights: <data>/checkpoints if present, else the copy the Docker image downloads."""
    for folder in ('/home/appuser/data/checkpoints', '/home/appuser/models'):
        path = os.path.join(folder, f'depth_anything_v2_{encoder}.pth')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'depth_anything_v2_{encoder}.pth not found in /home/appuser/data/checkpoints '
                            'or /home/appuser/models')

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def load_calib(filepath):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        data = {}

        # Load the calibration file
        filedata = read_calib_file(filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        data['T_cam0_velo'] = np.reshape(filedata['Tr_velo_to_cam'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)

        return namedtuple('CalibData', data.keys())(*data.values())

def read_indexes(filename):
    indexes = []
    with open(filename, 'r') as file:
        for line in file:
            indexes.append(str(line.strip()))
    return indexes


class DatasetCLRVoD(Dataset):

    def __init__(self, dataset_dir,
                 max_t=1.5, max_r=20., split='val', test_same_miscalib_number = None):
        super(DatasetCLRVoD, self).__init__()
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.all_files = []
        self.split = split

        self.K = {}

        self.GTs_R_CL = {}
        self.GTs_T_CL = {}
        self.GTs_RT_CL = {}

        self.GTs_R_CR = {}
        self.GTs_T_CR = {}
        self.GTs_RT_CR = {}
        self.radar_frames = 'radar_5frames'
        # self.radar_frames = 'radar'

        if split == 'val':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'val.txt'))
        if split == 'train':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'train.txt'))
        if split == 'full':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'full.txt'))
        if split == 'test':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'test.txt'))
        if split == 'demo':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'demo.txt'))


        #Reading calibration data
        calibCL = load_calib(os.path.join(dataset_dir, 'lidar', 'training', 'calib', '00210.txt')) #Reading 00000.txt since all of them are the same
        self.K = calibCL.K_cam2 # 3x3
        self.GTs_RT_CL = calibCL.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
        self.GTs_RT_CL = self.GTs_RT_CL.astype(np.float32)


        calibCR = load_calib(os.path.join(dataset_dir, self.radar_frames, 'training', 'calib', '00210.txt')) #Reading 00000.txt since all of them are the same
        self.GTs_RT_CR = calibCR.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
        self.GTs_RT_CR = self.GTs_RT_CR.astype(np.float32)

        depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2')

        if not os.path.exists(depth_path):
            os.makedirs(depth_path)
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

            model = DepthAnythingV2(**model_configs[encoder])
            print(f'Loading model: {encoder}')
            model.load_state_dict(torch.load(depth_anything_weights(encoder), map_location='cpu', weights_only=True))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()
            for item in tqdm(os.listdir(os.path.join(self.root_dir, 'lidar', 'training', 'image_2')), desc=f'Generating depth images'):
                img = cv2.imread(os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item))
                file_name, _ = os.path.splitext(item)
                rgb_depth = model.infer_image(img)
                cv2.imwrite(os.path.join(depth_path, file_name+'.png'), rgb_depth)

        self.val_RT_CL = []
        self.val_RT_CR = []
        if test_same_miscalib_number is None:
            test_same_miscalib_number = 1
        if test_same_miscalib_number == 0:
            test_same_miscalib_number = len(self.indexes)
        if split == 'val' or split == 'test' or split == 'demo':
            val_RT_file = os.path.join(dataset_dir, 'lidar', 'training',
                                       f'val_RT_CL_{split}_{test_same_miscalib_number}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file) and test_same_miscalib_number == 1:
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT_CL.append(list(row))
            else:
                # Only single-frame miscalibrations are replayed; other draws are not stored.
                val_RT_file = open(val_RT_file if test_same_miscalib_number == 1 else os.devnull, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.indexes)):
                    if i % test_same_miscalib_number == 0:
                        rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT_CL.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT_CL) == len(self.indexes), "Something wrong with test RTs"

        if split == 'val' or split == 'test' or split == 'demo':
            val_RT_file = os.path.join(dataset_dir, self.radar_frames, 'training',
                                       f'val_RT_CR_{split}_{test_same_miscalib_number}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file) and test_same_miscalib_number == 1:
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT_CR.append(list(row))
            else:
                # Only single-frame miscalibrations are replayed; other draws are not stored.
                val_RT_file = open(val_RT_file if test_same_miscalib_number == 1 else os.devnull, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.indexes)):
                    if i % test_same_miscalib_number == 0:
                        rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT_CR.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                         float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT_CR) == len(self.indexes), "Something wrong with test RTs"

    def custom_transform(self, rgb):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb
    
    def parse_point_cloud(self, scan, dim, RT):
        pc = scan.reshape((-1, dim))
        #correct wrong x,y,z calculations from VoD for Radar
        if dim == 7:
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            alpha = np.arctan2(y, x)
            cos_alpha = np.cos(alpha)

            # Handle potential division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                safe_cos_alpha = np.where(cos_alpha == 0, np.nan, cos_alpha)
                z = z / safe_cos_alpha
                z = np.nan_to_num(z, nan=np.inf)

            pc[:, 2] = z
        #filter out points closer than 2m
        valid_indices = pc[:, 0] < -2.
        valid_indices = valid_indices | (pc[:, 0] > 2.)
        valid_indices = valid_indices | (pc[:, 1] < -2.)
        valid_indices = valid_indices | (pc[:, 1] > 2.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))

        pc_org = pc_org.t()
        pc_in = rotate_back(pc_org, torch.tensor(RT))

        return pc_in

    def get_QT(self, valfile = None, idx = None):
        if valfile is None:
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_RT = None
        else:
            initial_RT = valfile[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]
        R, T = np.array([rotx,roty,rotz]), np.array([transl_x, transl_y, transl_z])

        R, T = invert_pose(R, T)
        Q = euler_to_quaternion(R)
        Q, T = torch.tensor(Q), torch.tensor(T)

        return R, Q, T, initial_RT

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):

        item = self.indexes[idx]
        img_path = os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item+'.jpg')
        radar_path = os.path.join(self.root_dir, self.radar_frames, 'training', 'velodyne', item+'.bin')
        lidar_path = os.path.join(self.root_dir, 'lidar', 'training', 'velodyne', item+'.bin')
        radar_scan = np.fromfile(radar_path, dtype=np.float32)
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        lidar_pc = self.parse_point_cloud(lidar_scan, 4, self.GTs_RT_CL)
        radar_pc = self.parse_point_cloud(radar_scan, 7, self.GTs_RT_CR)

        #depth
        depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2', item+'.png')
        depth_img = Image.open(depth_path)
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.5], std=[0.5])
        depth_img = to_tensor(depth_img)
        depth_img = normalization(depth_img)

        original_img = Image.open(img_path)
        img = self.custom_transform(original_img)
        original_img = np.array(original_img)

        if self.split == 'train':
            _, Q_CL, T_CL, initial_RT_CL = self.get_QT()
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT()
        else:
            _, Q_CL, T_CL, initial_RT_CL = self.get_QT(self.val_RT_CL, idx)
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT(self.val_RT_CR, idx)
            initial_RT_CL = torch.tensor(initial_RT_CL)
            initial_RT_CR = torch.tensor(initial_RT_CR)

        RT_CL = quat2mat(Q_CL)
        RT_CL[:3, 3] = T_CL
        RT_CR = quat2mat(Q_CR)
        RT_CR[:3, 3] = T_CR
        RT_LC = RT_CL.inverse()
        RT_LR = RT_LC.matmul(RT_CR)
        Q_LR = quaternion_from_matrix(RT_LR[:3, :3])
        T_LR = RT_LR[:3, 3]
        R_LR = quaternion_to_euler(Q_LR)

        #create initial_RT_LR list from R_LR and T_LR
        initial_RT_LR = [0, 0, 0, 0, 0, 0, 0]
        initial_RT_LR[1] = T_LR[0]
        initial_RT_LR[2] = T_LR[1]
        initial_RT_LR[3] = T_LR[2]
        initial_RT_LR[4] = R_LR[0]
        initial_RT_LR[5] = R_LR[1]
        initial_RT_LR[6] = R_LR[2]
        initial_RT_LR = torch.tensor(initial_RT_LR)

        GTs_RT_LC = torch.tensor(self.GTs_RT_CL).inverse()
        GTs_RT_LR = torch.mm(GTs_RT_LC, torch.tensor(self.GTs_RT_CR))

        calib = self.K

        GTs_RT_CL = self.GTs_RT_CL
        GTs_RT_CR = self.GTs_RT_CR

        if self.split == 'test' or self.split == 'demo':
            sample = {'rgb': img, 'lidar': lidar_pc, 'radar': radar_pc, 'calib': calib, 'depth': depth_img, 'orig_img': original_img,
                      'tr_error_CL': T_CL, 'rot_error_CL': Q_CL, 'tr_error_CR': T_CR, 'rot_error_CR': Q_CR, 'tr_error_LR': T_LR, 'rot_error_LR': Q_LR,
                      'img_path': img_path,
                      'rgb_name': item + '.jpg', 'item': item, 'extrin_CL': GTs_RT_CL, 'extrin_CR': GTs_RT_CR, 'extrin_LR': GTs_RT_LR,
                      'initial_RT_CL': initial_RT_CL, 'initial_RT_CR': initial_RT_CR, 'initial_RT_LR': initial_RT_LR}
        else:
            sample = {'rgb': img, 'lidar': lidar_pc, 'radar': radar_pc, 'depth': depth_img, 'calib': calib,
                      'tr_error_CL': T_CL, 'rot_error_CL': Q_CL, 'tr_error_CR': T_CR, 'rot_error_CR': Q_CR, 'tr_error_LR': T_LR, 'rot_error_LR': Q_LR,
            }
        return sample

class DatasetCRVoD(Dataset):

    def __init__(self, dataset_dir, max_t=1.5, max_r=20., split='val', test_same_miscalib_number=None, rgb_depth_input=False, radar_frames='radar_5frames'):
        super(DatasetCRVoD, self).__init__()
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.all_files = []
        self.split = split
        self.rgb_depth_input = rgb_depth_input

        self.K = {}

        self.GTs_RT_CR = {}
        self.radar_frames = radar_frames  # default 'radar_5frames'; config can select 'radar' (non-accumulated)

        if split == 'val':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'val.txt'))
        if split == 'train':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'train.txt'))
        if split == 'full':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'full.txt'))
        if split == 'test':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'test.txt'))
        if split == 'demo':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'demo.txt'))

        # Reading calibration data
        calibCR = load_calib(os.path.join(dataset_dir, self.radar_frames, 'training', 'calib', '00210.txt'))  # Reading 00000.txt since all of them are the same
        self.K = calibCR.K_cam2  # 3x3 camera intrinsic matrix
        self.GTs_RT_CR = calibCR.T_cam2_velo  # Ground truth pose from camera to radar (T_cam02_velo: 4x4)
        self.GTs_RT_CR = self.GTs_RT_CR.astype(np.float32)

        if rgb_depth_input:

            # Handling depth data
            depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2')
            if not os.path.exists(depth_path):
                os.makedirs(depth_path)
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                encoder = 'vitl'
                model = DepthAnythingV2(**model_configs[encoder])
                print(f'Loading model: {encoder}')
                model.load_state_dict(torch.load(depth_anything_weights(encoder), map_location='cpu', weights_only=True))
                model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                model.eval()
                for item in tqdm(os.listdir(os.path.join(self.root_dir, 'lidar', 'training', 'image_2')), desc=f'Generating depth images'):
                    img = cv2.imread(os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item))
                    file_name, _ = os.path.splitext(item)
                    rgb_depth = model.infer_image(img)
                    cv2.imwrite(os.path.join(depth_path, file_name+'.png'), rgb_depth)

        # Prepare validation and test miscalibration matrices
        self.val_RT_CR = []
        if test_same_miscalib_number is None:
            test_same_miscalib_number = 1
        if test_same_miscalib_number == 0:
            test_same_miscalib_number = len(self.indexes)
        if split == 'val' or split == 'test' or split == 'demo':
            val_RT_file = os.path.join(dataset_dir, self.radar_frames, 'training',
                                       f'val_RT_CR_{split}_{test_same_miscalib_number}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file) and test_same_miscalib_number == 1:
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT_CR.append(list(row))
            else:
                # Only single-frame miscalibrations are replayed; other draws are not stored.
                val_RT_file = open(val_RT_file if test_same_miscalib_number == 1 else os.devnull, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.indexes)):
                    if i % test_same_miscalib_number == 0:
                        rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                    self.val_RT_CR.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                           float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT_CR) == len(self.indexes), "Something wrong with test RTs"

    def custom_transform(self, rgb):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def parse_point_cloud(self, scan, dim, RT):
        pc = scan.reshape((-1, dim))
        # Correct wrong x, y, z calculations from VoD for Radar
        if dim == 7:
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            alpha = np.arctan2(y, x)
            cos_alpha = np.cos(alpha)

            with np.errstate(divide='ignore', invalid='ignore'):
                safe_cos_alpha = np.where(cos_alpha == 0, np.nan, cos_alpha)
                z = z / safe_cos_alpha
                z = np.nan_to_num(z, nan=np.inf)

            pc[:, 2] = z
        valid_indices = pc[:, 0] < -2.
        valid_indices = valid_indices | (pc[:, 0] > 2.)
        valid_indices = valid_indices | (pc[:, 1] < -2.)
        valid_indices = valid_indices | (pc[:, 1] > 2.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))

        pc_org = pc_org.t()
        pc_in = rotate_back(pc_org, torch.tensor(RT))

        return pc_in

    def get_QT(self, valfile=None, idx=None):
        if valfile is None:
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_RT = None
        else:
            initial_RT = valfile[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]
        R, T = np.array([rotx, roty, rotz]), np.array([transl_x, transl_y, transl_z])

        R, T = invert_pose(R, T)
        Q = euler_to_quaternion(R)
        Q, T = torch.tensor(Q), torch.tensor(T)

        return R, Q, T, initial_RT

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        item = self.indexes[idx]
        # Original camera path
        cam_image_path = os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item + '.jpg')
        orig_img = Image.open(cam_image_path)
        W, H = orig_img.size
        cam_img = self.custom_transform(orig_img)
        orig_img = np.array(orig_img)

        # Original depth path
        if self.rgb_depth_input:
            depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2', item + '.png')
            depth_img = Image.open(depth_path).convert('L')
            depth_img = depth_img.resize((W, H), Image.BICUBIC)
            depth_img = transforms.ToTensor()(depth_img)
        else:
            depth_img = torch.zeros((1, H, W), dtype=torch.float32)
            
        # Point Cloud path
        radar_path = os.path.join(self.root_dir, self.radar_frames, 'training', 'velodyne', item + '.bin')
        radar_bin = np.fromfile(radar_path, dtype=np.float32)
        radar_pc = self.parse_point_cloud(radar_bin, 7, self.GTs_RT_CR)

        if self.split == 'train':
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT()
        else:
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT(self.val_RT_CR, idx)
            initial_RT_CR = torch.tensor(initial_RT_CR)

        calib = self.K
    
        if self.split == 'test' or self.split == 'demo':
            sample = {'rgb': cam_img, 'radar': radar_pc, 'depth': depth_img, 'tr_error': T_CR, 'rot_error': Q_CR,
                      'img_path': cam_image_path, 'rgb_name': item + '.jpg', 'item': item, 'extrin': self.GTs_RT_CR,
                      'initial_RT': initial_RT_CR, 'calib': calib, 'orig_img': orig_img}
        else:
            sample = {'rgb': cam_img, 'radar': radar_pc, 'depth': depth_img, 'tr_error': T_CR, 'rot_error': Q_CR, 'calib': calib}

        return sample

class DatasetCLVoD(Dataset):

    def __init__(self, dataset_dir, max_t=1.5, max_r=20., split='val', test_same_miscalib_number=None, radar_frames='radar_5frames'):
        super(DatasetCLVoD, self).__init__()
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.all_files = []
        self.split = split

        self.K = {}

        self.GTs_RT_CR = {}
        self.radar_frames = radar_frames  # default 'radar_5frames'; config can select 'radar' (non-accumulated)

        if split == 'val':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'val.txt'))
        if split == 'train':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'train.txt'))
        if split == 'full':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'full.txt'))
        if split == 'test':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'test.txt'))
        if split == 'demo':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/KITTI_Second_TIER_05_29/uvod/radar_5frames/ImageSets', 'demo.txt'))

        # Reading calibration data
        calibCR = load_calib(os.path.join(dataset_dir, 'lidar', 'training', 'calib', '00210.txt'))  # Reading 00000.txt since all of them are the same
        self.K = calibCR.K_cam2  # 3x3 camera intrinsic matrix
        self.GTs_RT_CR = calibCR.T_cam2_velo  # Ground truth pose from camera to radar (T_cam02_velo: 4x4)
        self.GTs_RT_CR = self.GTs_RT_CR.astype(np.float32)

        # Handling depth data
        depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2')
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            encoder = 'vitl'
            model = DepthAnythingV2(**model_configs[encoder])
            print(f'Loading model: {encoder}')
            model.load_state_dict(torch.load(depth_anything_weights(encoder), map_location='cpu', weights_only=True))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()
            for item in tqdm(os.listdir(os.path.join(self.root_dir, 'lidar', 'training', 'image_2')), desc=f'Generating depth images'):
                img = cv2.imread(os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item))
                file_name, _ = os.path.splitext(item)
                rgb_depth = model.infer_image(img)
                cv2.imwrite(os.path.join(depth_path, file_name+'.png'), rgb_depth)

        # Prepare validation and test miscalibration matrices
        self.val_RT_CR = []
        if test_same_miscalib_number is None:
            test_same_miscalib_number = 1
        if test_same_miscalib_number == 0:
            test_same_miscalib_number = len(self.indexes)
        if split == 'val' or split == 'test' or split == 'demo':
            val_RT_file = os.path.join(dataset_dir, self.radar_frames, 'training',
                                       f'val_RT_CR_{split}_{test_same_miscalib_number}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file) and test_same_miscalib_number == 1:
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT_CR.append(list(row))
            else:
                # Only single-frame miscalibrations are replayed; other draws are not stored.
                val_RT_file = open(val_RT_file if test_same_miscalib_number == 1 else os.devnull, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.indexes)):
                    if i % test_same_miscalib_number == 0:
                        rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                    self.val_RT_CR.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                           float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT_CR) == len(self.indexes), "Something wrong with test RTs"

    def custom_transform(self, rgb):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def parse_point_cloud(self, scan, dim, RT):
        pc = scan.reshape((-1, dim))
        # Correct wrong x, y, z calculations from VoD for Radar
        if dim == 7:
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            alpha = np.arctan2(y, x)
            cos_alpha = np.cos(alpha)

            with np.errstate(divide='ignore', invalid='ignore'):
                safe_cos_alpha = np.where(cos_alpha == 0, np.nan, cos_alpha)
                z = z / safe_cos_alpha
                z = np.nan_to_num(z, nan=np.inf)

            pc[:, 2] = z
        valid_indices = pc[:, 0] < -2.
        valid_indices = valid_indices | (pc[:, 0] > 2.)
        valid_indices = valid_indices | (pc[:, 1] < -2.)
        valid_indices = valid_indices | (pc[:, 1] > 2.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))

        pc_org = pc_org.t()
        pc_in = rotate_back(pc_org, torch.tensor(RT))

        return pc_in

    def get_QT(self, valfile=None, idx=None):
        if valfile is None:
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_RT = None
        else:
            initial_RT = valfile[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]
        R, T = np.array([rotx, roty, rotz]), np.array([transl_x, transl_y, transl_z])

        R, T = invert_pose(R, T)
        Q = euler_to_quaternion(R)
        Q, T = torch.tensor(Q), torch.tensor(T)

        return R, Q, T, initial_RT

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        item = self.indexes[idx]
        # Original camera path
        cam_image_path = os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item + '.jpg')
        orig_img = Image.open(cam_image_path)
        W, H = orig_img.size
        cam_img = self.custom_transform(orig_img)
        orig_img = np.array(orig_img)

        # Original depth path
        depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2', item + '.png')
        depth_img = Image.open(depth_path).convert('L')
        depth_img = depth_img.resize((W, H), Image.BICUBIC)
        depth_img = transforms.ToTensor()(depth_img)

        # Point Cloud path
        radar_path = os.path.join(self.root_dir, 'lidar', 'training', 'velodyne', item + '.bin')
        radar_bin = np.fromfile(radar_path, dtype=np.float32)
        radar_pc = self.parse_point_cloud(radar_bin, 4, self.GTs_RT_CR)

        if self.split == 'train':
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT()
        else:
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT(self.val_RT_CR, idx)
            initial_RT_CR = torch.tensor(initial_RT_CR, dtype=torch.float32)

        calib = self.K
    
        if self.split == 'test' or self.split == 'demo':
            sample = {'rgb': cam_img, 'lidar': radar_pc, 'depth': depth_img, 'tr_error': T_CR, 'rot_error': Q_CR,
                      'img_path': cam_image_path, 'rgb_name': item + '.jpg', 'item': item, 'extrin': self.GTs_RT_CR,
                      'initial_RT': initial_RT_CR, 'calib': calib, 'orig_img': orig_img}
        else:
            sample = {'rgb': cam_img, 'lidar': radar_pc, 'depth': depth_img, 'tr_error': T_CR, 'rot_error': Q_CR, 'calib': calib}

        return sample        

# ---------------------------------------------------------------------------------------------
# Dual-Radar (Arbe Phoenix) support. Points have 5 columns.
# ---------------------------------------------------------------------------------------------

def load_calib_Dual(filepath):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        data = {}

        # Load the calibration file
        filedata = read_calib_file(filepath)

        # Create 3x4 projection matrices
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))

        data['P_rect_20'] = P_rect_20

        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]

        data['T_cam0_velo'] = np.reshape(filedata['Tr_arbe_to_cam'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam2'] = P_rect_20[0:3, 0:3]

        return namedtuple('CalibData', data.keys())(*data.values())


class DatasetCRDual(Dataset):

    def __init__(self, dataset_dir, max_t=1.5, max_r=20., split='val', test_same_miscalib_number=None, rgb_depth_input=False, radar_5frames=False, downsample_points=None):
        super(DatasetCRDual, self).__init__()
        # None/0 keeps every point (default); e.g. 400 for the radar-density experiment.
        self.downsample_points = downsample_points
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.all_files = []
        self.split = split
        self.rgb_depth_input = rgb_depth_input

        self.K = {}

        self.GTs_RT_CR = {}

        if split == 'val':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/DualRadar/lidar', 'ImageSets', 'val.txt'))
        if split == 'train':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/DualRadar/lidar', 'ImageSets', 'train.txt'))
        if split == 'test':
            self.indexes = read_indexes(os.path.join('/home/appuser/data/DualRadar/lidar', 'ImageSets', 'test.txt'))

        # Reading calibration data
        calibCR = load_calib_Dual(os.path.join(dataset_dir, 'radar', 'training', 'calib', '000210.txt'))  # Reading 00000.txt since all of them are the same
        self.K = calibCR.K_cam2  # 3x3 camera intrinsic matrix
        self.GTs_RT_CR = calibCR.T_cam2_velo  # Ground truth pose from camera to radar (T_cam02_velo: 4x4)
        self.GTs_RT_CR = self.GTs_RT_CR.astype(np.float32)

        #check if depthV2 folder exists and has the same amount of file as the image_2 folder
        depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2')
        img2_path = os.path.join(self.root_dir, 'lidar', 'training', 'image_2')
        depth_folder_exists = os.path.exists(depth_path)
        img2_files = [f for f in os.listdir(img2_path) if os.path.isfile(os.path.join(img2_path, f))]
        depth_files = [f for f in os.listdir(depth_path)] if depth_folder_exists else []
        depth_folder_valid = depth_folder_exists and (len(depth_files) == len(img2_files))

        if rgb_depth_input and not depth_folder_valid:
            print(f"Depth folder '{depth_path}' does not exist or is invalid. Generating depth images.")
            # Handling depth data
            if not os.path.exists(depth_path):
                os.makedirs(depth_path)
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            encoder = 'vitl'
            model = DepthAnythingV2(**model_configs[encoder])
            print(f'Loading model: {encoder}')
            model.load_state_dict(torch.load(depth_anything_weights(encoder), map_location='cpu', weights_only=True))
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()
            for item in tqdm(os.listdir(os.path.join(self.root_dir, 'lidar', 'training', 'image_2')), desc=f'Generating depth images'):
                img = cv2.imread(os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item))
                file_name, _ = os.path.splitext(item)
                # Check if the depth image already exists
                depth_file = os.path.join(depth_path, file_name+'.png')
                if not os.path.exists(depth_file):
                    if img is None:
                        print(f"Image {file_name} could not be loaded. Skipping inference and saving black image.")
                        rgb_depth = np.zeros((1920, 1200, 3), dtype=np.uint8)  # fallback default size
                    else:
                        try:
                            rgb_depth = model.infer_image(img)
                        except Exception as e:
                            print(f'Error generating depth for {file_name}: {e}. Saving black image.')
                            rgb_depth = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    cv2.imwrite(os.path.join(depth_path, file_name+'.png'), rgb_depth)
                    
        # Prepare validation and test miscalibration matrices
        self.val_RT_CR = []
        if test_same_miscalib_number is None:
            test_same_miscalib_number = 1
        if test_same_miscalib_number == 0:
            test_same_miscalib_number = len(self.indexes)
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir, 'radar', 'training',
                                       f'val_RT_CR_{split}_{test_same_miscalib_number}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file) and test_same_miscalib_number == 1:
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT_CR.append(list(row))
                print("Loaded valfile: ", val_RT_file)
            else:
                # Only single-frame miscalibrations are replayed; other draws are not stored.
                val_RT_file = open(val_RT_file if test_same_miscalib_number == 1 else os.devnull, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.indexes)):
                    if i % test_same_miscalib_number == 0:
                        rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-max_t, max_t)
                        transl_y = np.random.uniform(-max_t, max_t)
                        transl_z = np.random.uniform(-max_t, max_t)
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z, rotx, roty, rotz])
                    self.val_RT_CR.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                           float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT_CR) == len(self.indexes), "Something wrong with test RTs"

    def custom_transform(self, rgb):
        try:
            # Verify image is valid before transforming
            if not isinstance(rgb, Image.Image):
                rgb = Image.open(rgb)
            if rgb.mode != 'RGB':
                rgb = rgb.convert('RGB')
            to_tensor = transforms.ToTensor()
            normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

            if self.split == 'train':
                color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
                rgb = color_transform(rgb)

            rgb = to_tensor(rgb)
            rgb = normalization(rgb)
            return rgb
        except (OSError, IOError) as e:
            print(f"Error loading/processing image: {e}")
            raise RuntimeError(f"Failed to load or process image: {e}")

    # def custom_transform(self, rgb):
    #     try:
    #         # Verify image is valid before transforming
    #         if not isinstance(rgb, Image.Image):
    #             print(f"Input is not a PIL Image: {type(rgb)}")
    #             rgb = Image.open(rgb)
                
    #         if rgb.mode != 'RGB':
    #             print(f"Converting image from {rgb.mode} to RGB")
    #             rgb = rgb.convert('RGB')
                
    #         to_tensor = transforms.ToTensor()
    #         normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                         std=[0.229, 0.224, 0.225])

    #         if self.split == 'train':
    #             try:
    #                 color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
    #                 rgb = color_transform(rgb)
    #             except Exception as e:
    #                 print(f"Error during color transformation: {e}")
    #                 raise

    #         rgb = to_tensor(rgb)
    #         rgb = normalization(rgb)
    #         return rgb
            
    #     except (OSError, IOError) as e:
    #         print(f"Error loading/processing image: {e}")
    #         print(f"Image mode: {getattr(rgb, 'mode', 'N/A')}")
    #         print(f"Image size: {getattr(rgb, 'size', 'N/A')}")
    #         raise RuntimeError(f"Failed to load or process image: {e}")

    def parse_point_cloud(self, scan, dim, RT):
        pc = scan.reshape((-1, dim))

        valid_indices = pc[:, 0] < -2.
        valid_indices = valid_indices | (pc[:, 0] > 2.)
        valid_indices = valid_indices | (pc[:, 1] < -2.)
        valid_indices = valid_indices | (pc[:, 1] > 2.)
        pc = pc[valid_indices].copy()

        # CLR_VIZ_CROP=1 crops to the Dual-Radar authors' POINT_CLOUD_RANGE (visualisation only).
        if os.environ.get('CLR_VIZ_CROP') == '1':
            r = ((pc[:, 0] >= -40.) & (pc[:, 0] <= 40.) &
                 (pc[:, 1] >= -10.) & (pc[:, 1] <= 60.4) &
                 (pc[:, 2] >= -3.) & (pc[:, 2] <= 1.))
            pc = pc[r].copy()

        pc_org = torch.from_numpy(pc.astype(np.float32))

        # Radar-density study (paper: domain transfer). The Arbe Phoenix returns 6k-14k points per
        # frame while VoD's ZF radar returns ~400, so randomly subsampling to 400 isolates point
        # density from the other domain-gap factors (camera placement, intrinsics, illumination).
        # Was a hardcoded `downsample = False`; now driven by `downsample_points` in the config so
        # the experiment is a config change rather than a source edit.
        if self.downsample_points and pc_org.shape[0] > self.downsample_points:
            idxs = np.random.choice(pc_org.shape[0], self.downsample_points, replace=False)
            pc_org = pc_org[idxs]

        pc_org = pc_org.t()
        pc_in = rotate_back(pc_org, torch.tensor(RT))

        return pc_in

    def get_QT(self, valfile=None, idx=None):
        if valfile is None:
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            initial_RT = None
        else:
            initial_RT = valfile[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]
        R, T = np.array([rotx, roty, rotz]), np.array([transl_x, transl_y, transl_z])

        R, T = invert_pose(R, T)
        Q = euler_to_quaternion(R)
        Q, T = torch.tensor(Q), torch.tensor(T)

        return R, Q, T, initial_RT

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        item = self.indexes[idx]
        # Original camera path
        cam_image_path = os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item + '.png')
        orig_img = Image.open(cam_image_path)
        W, H = orig_img.size
        cam_img = self.custom_transform(orig_img)
        orig_img = np.array(orig_img)

        # try:
        #     item = self.indexes[idx]
        #     # Original camera path
        #     cam_image_path = os.path.join(self.root_dir, 'lidar', 'training', 'image_2', item + '.png')
        #     print(f"Attempting to load image: {cam_image_path}")  # Debug print
            
        #     try:
        #         orig_img = Image.open(cam_image_path)
        #         print(f"Successfully opened image: {cam_image_path}")  # Debug print
        #     except Exception as e:
        #         print(f"Failed to open image {cam_image_path}: {e}")
        #         raise
                
        #     W, H = orig_img.size
        #     try:
        #         cam_img = self.custom_transform(orig_img)
        #     except Exception as e:
        #         print(f"Failed to transform image {cam_image_path}: {e}")
        #         raise
        
        # except Exception as e:
        #     print(f"Error processing index {idx} with item {self.indexes[idx]}: {e}")
        #     raise

        # Original depth path
        if self.rgb_depth_input:
            depth_path = os.path.join(self.root_dir, 'lidar', 'training', 'depthV2', item + '.png')
            depth_img = Image.open(depth_path).convert('L')
            depth_img = depth_img.resize((W, H), Image.BICUBIC)
            depth_img = transforms.ToTensor()(depth_img)
        else:
            depth_img = torch.zeros((1, H, W), dtype=torch.float32)
            
        # Point Cloud path
        radar_path = os.path.join(self.root_dir, 'radar', 'training', 'velodyne', item + '.bin')
        radar_bin = np.fromfile(radar_path, dtype=np.float32)
        radar_pc = self.parse_point_cloud(radar_bin, 5, self.GTs_RT_CR)

        if self.split == 'train':
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT()
        else:
            _, Q_CR, T_CR, initial_RT_CR = self.get_QT(self.val_RT_CR, idx)
            initial_RT_CR = torch.tensor(initial_RT_CR)

        calib = self.K
    
        if self.split == 'test':
            sample = {'rgb': cam_img, 'radar': radar_pc, 'depth': depth_img, 'tr_error': T_CR, 'rot_error': Q_CR,
                      'img_path': cam_image_path, 'rgb_name': item + '.png', 'item': item, 'extrin': self.GTs_RT_CR,
                      'initial_RT': initial_RT_CR, 'calib': calib, 'orig_img': orig_img}
        else:
            sample = {'rgb': cam_img, 'radar': radar_pc, 'depth': depth_img, 'tr_error': T_CR, 'rot_error': Q_CR, 'calib': calib}

        return sample
