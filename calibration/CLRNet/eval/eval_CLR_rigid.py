import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
from tqdm import tqdm

from models.CLRNet import CLRNet, CLRHayoun
from CLRNet.datasets.DatasetCLR import DatasetCLRVoD
from CLRNet.utils.checkpoint import load_checkpoint, read_checkpoint_config, apply_checkpoint_config

from CLRNet.utils.utils import (merge_inputs, quat2mat, mat2xyzrpy,
                   print_results, 
                   tvector2mat, bcolors, read_config)

from CLRNet.utils.preprocess import preprocess_sample_CLR

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))

plt.rcParams['axes.unicode_minus'] = False

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CLR_GPU', '0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modes: CLR (CLRNet and CLRNet+4), Hayoun
mode = os.environ.get('CLR_MODE', 'CLR')

def _init_fn(worker_id, seed):
    seed = seed + worker_id + 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_eval(_config, weights=None):
    dataset_class = DatasetCLRVoD

    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=0)

    def init_fn(x):
        return _init_fn(x, 0)
    
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']

    average_matrix_CL, average_matrix_CR = None, None

    for iteration, weight in enumerate(weights):

        checkpoint = load_checkpoint(weight)
        apply_checkpoint_config(_config, checkpoint.get('config') or read_checkpoint_config(weight), stage=True)

        # A multi-frame model stacks `multiple_input` frames per sample; keep the frames per batch constant.
        TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                    shuffle=False,
                                                    batch_size=max(1, batch_size // _config['multiple_input']),
                                                    num_workers=num_worker,
                                                    worker_init_fn=init_fn,
                                                    collate_fn=merge_inputs,
                                                    drop_last=False,
                                                    pin_memory=False)

        if mode == 'CLR':
            model = CLRNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                        use_reflectance_LIDAR=_config['use_reflectance_LIDAR'], use_reflectance_RADAR=_config['use_reflectance_RADAR'], use_velocity_RADAR = _config['use_velocity_RADAR'], use_time_RADAR=_config['use_time_RADAR'], use_depth=_config['use_depth'],
                        dropout=_config['dropout'], res_num=18)
            
        if mode == 'Hayoun':
            model = CLRHayoun()
        
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        model = model.to(device)
        model.eval()

        RTs_CL_temp = []
        RTs_CR_temp = []
    
        for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
            if iteration == 0 and batch_idx == 0:
                sample['tr_error_CL'] = sample['tr_error_CL'].cuda()
                sample['rot_error_CL'] = sample['rot_error_CL'].cuda()
                sample['tr_error_CR'] = sample['tr_error_CR'].cuda()
                sample['rot_error_CR'] = sample['rot_error_CR'].cuda()
                average_matrix_CL = torch.mm(tvector2mat(sample['tr_error_CL'][-1]), quat2mat(sample['rot_error_CL'][-1]))
                average_matrix_CR = torch.mm(tvector2mat(sample['tr_error_CR'][-1]), quat2mat(sample['rot_error_CR'][-1]))

            rgb_input, lidar_input, radar_input, rgb_depth_input = preprocess_sample_CLR(sample, _config, average_matrix_CL, average_matrix_CR)

            # Run model
            with torch.no_grad():                
                
                T_predicted_CL, R_predicted_CL, _, _, T_predicted_CR, R_predicted_CR = model(rgb_input, lidar_input, radar_input, rgb_depth_input)

                for i in range(len(sample['initial_RT_CL'])):    

                    RT_predicted_CL = torch.mm(tvector2mat(T_predicted_CL[i]), quat2mat(R_predicted_CL[i]))
                    pred_temp_CL = torch.mm(average_matrix_CL, RT_predicted_CL.inverse())
                    RTs_CL_temp.append(pred_temp_CL)

                    RT_predicted_CR = torch.mm(tvector2mat(T_predicted_CR[i]), quat2mat(R_predicted_CR[i]))
                    pred_temp_CR = torch.mm(average_matrix_CR, RT_predicted_CR.inverse())
                    RTs_CR_temp.append(pred_temp_CR)

        # average_matrix_CL = torch.mean(torch.stack(all_RTs_CL), axis=0)
        # average_matrix_CR = torch.mean(torch.stack(all_RTs_CR), axis=0)

        average_matrix_CL = torch.median(torch.stack(RTs_CL_temp), axis=0).values
        average_matrix_CR = torch.median(torch.stack(RTs_CR_temp), axis=0).values

        rpy_CR = mat2xyzrpy(average_matrix_CR)[3:]
        xyz_CR = average_matrix_CR[:3, 3]
        print("rpy CR: ", rpy_CR * 180 / np.pi)
        print("xyz CR: ", xyz_CR * 100)

        rpy_CL = mat2xyzrpy(average_matrix_CL)[3:]
        xyz_CL = average_matrix_CL[:3, 3]
        print("rpy CL: ", rpy_CL * 180 / np.pi)
        print("xyz CL: ", xyz_CL * 100)

        # Clear the GPU memory for the current model
        del model
        torch.cuda.empty_cache()

    return average_matrix_CL, average_matrix_CR

CKPT = '/home/appuser/data/checkpoints/'
# The paper's rigid-rig cascade (CLRNet+4); it does not use the 20 cm model.
DEFAULT_WEIGHTS = [CKPT + 'clrnet_vod_100cm_20deg.pth', CKPT + 'clrnet_vod_50cm_5deg.pth',
                   CKPT + 'clrnet4_vod_5cm_0.5deg.pth']


def main():
    weights = os.environ['CLR_WEIGHTS'].split(',') if os.environ.get('CLR_WEIGHTS') else DEFAULT_WEIGHTS
    _config = read_config(os.environ.get('CLR_CONFIG', '/home/appuser/src/calibration/CLRNet/configs/VOD_CLR_config.yaml'))
    apply_checkpoint_config(_config, read_checkpoint_config(weights[0]))
    output = os.environ.get('CLR_OUTPUT', os.path.join('/home/appuser/data/output/rigid', mode,
                                                       os.path.splitext(os.path.basename(weights[0]))[0]))
    print("Output folder: ", output)

    # Each draw applies one random miscalibration to the whole test sequence.
    range_count = int(os.environ.get('CLR_DRAWS', 30))
    all_RTs_CL = []
    all_RTs_CR = []
    for i in range(range_count):
        print("Iteration: " + bcolors.OKGREEN + str(i+1) + "/" + bcolors.ENDC + str(range_count))
        average_RT_CL, average_RT_CR = run_eval(dict(_config), weights)
        all_RTs_CL.append(average_RT_CL)
        all_RTs_CR.append(average_RT_CR)
    print("Results CL")
    print_results(all_RTs_CL, results_path=os.path.join(output, 'CL_mean'))
    print("Results CR")
    print_results(all_RTs_CR, results_path=os.path.join(output, 'CR_mean'))

if __name__ == '__main__':
    main()