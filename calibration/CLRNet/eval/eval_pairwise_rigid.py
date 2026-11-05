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

from models.CLRNet import CRNet, CLNet, LCCNet, CRSchollerNet
from CLRNet.datasets.DatasetCLR import DatasetCLVoD, DatasetCRVoD, DatasetCRDual
from CLRNet.utils.checkpoint import load_checkpoint, read_checkpoint_config, apply_checkpoint_config, mode_from_config

from CLRNet.utils.utils import (merge_inputs, quat2mat, mat2xyzrpy,
                   print_results, 
                   tvector2mat, bcolors, read_config)

from CLRNet.utils.preprocess import preprocess_sample_CL, preprocess_sample_CR, preprocess_sample_CL_LCC, preprocess_sample_CRScholler, preprocess_sample_CR_Dual

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))

plt.rcParams['axes.unicode_minus'] = False

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CLR_GPU', '0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modes: CR, CRDual, CL, LCC, CRScholler
mode = os.environ.get('CLR_MODE', 'CR')

def _init_fn(worker_id, seed):
    seed = seed + worker_id + 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_eval(_config, weights=None):
    if mode == 'CRDual':
        dataset_val = DatasetCRDual(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test',
                                    test_same_miscalib_number=0, rgb_depth_input=_config.get('use_depth_input', _config['use_depth']))
    elif mode == 'CR':
        dataset_val = DatasetCRVoD(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test',
                                   test_same_miscalib_number=0, rgb_depth_input=_config.get('use_depth_input', _config['use_depth']))
    elif mode == 'CRScholler':
        dataset_val = DatasetCRVoD(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=0)
    else:
        dataset_val = DatasetCLVoD(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='test', test_same_miscalib_number=0)

    def init_fn(x):
        return _init_fn(x, 0)
    
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

    average_matrix = None

    for iteration, weight in enumerate(weights):
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
        
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        model = model.to(device)
        model.eval()

        RTs_temp = []
    
        for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
            if iteration == 0 and batch_idx == 0:
                sample['tr_error'] = sample['tr_error'].cuda()
                sample['rot_error'] = sample['rot_error'].cuda()
                average_matrix = torch.mm(tvector2mat(sample['tr_error'][-1]), quat2mat(sample['rot_error'][-1]))

            if mode == 'CL':
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CL(sample, _config, average_matrix)
            elif mode == 'CR':
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR(sample, _config, average_matrix)
            elif mode == 'CRDual':
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR_Dual(sample, _config, average_matrix)
            elif mode == 'LCC':
                rgb_input, radar_input, _ = preprocess_sample_CL_LCC(sample, _config, average_matrix)
            elif mode == 'CRScholler':
                rgb_input, radar_input = preprocess_sample_CRScholler(sample, _config, average_matrix)

            # Run model
            with torch.no_grad():                
                
                if mode == 'LCC':
                    T_predicted, R_predicted = model(rgb_input, radar_input)
                if mode == 'CRScholler':
                    R_predicted = model(rgb_input, radar_input)
                if mode in ('CL', 'CR', 'CRDual'):
                    T_predicted, R_predicted = model(rgb_input, radar_input, rgb_depth_input)

                for i in range(len(sample['initial_RT'])):    
                    if mode == 'CRScholler':
                        RT_predicted = quat2mat(R_predicted[i])
                    else:
                        RT_predicted = torch.mm(tvector2mat(T_predicted[i]), quat2mat(R_predicted[i]))

                    pred_temp = torch.mm(average_matrix, RT_predicted.inverse())
                    RTs_temp.append(pred_temp)

        # average_matrix = torch.mean(torch.stack(all_RTs), axis=0)

        average_matrix = torch.median(torch.stack(RTs_temp), axis=0).values

        rpy = mat2xyzrpy(average_matrix)[3:]
        xyz = average_matrix[:3, 3]
        print("rpy CL: ", rpy * 180 / np.pi)
        print("xyz CL: ", xyz * 100)

        # Clear the GPU memory for the current model
        del model
        torch.cuda.empty_cache()

    return average_matrix

def main():
    global mode
    weights = os.environ['CLR_WEIGHTS'].split(',')
    ckpt_config = read_checkpoint_config(weights[0])
    if not os.environ.get('CLR_MODE'):
        mode = mode_from_config(ckpt_config, mode)
    config_name = 'Dual_CR_config.yaml' if mode == 'CRDual' else 'VOD_' + mode + '_config.yaml'
    _config = read_config(os.environ.get('CLR_CONFIG', '/home/appuser/src/calibration/CLRNet/configs/' + config_name))
    apply_checkpoint_config(_config, ckpt_config)
    output = os.environ.get('CLR_OUTPUT', os.path.join('/home/appuser/data/output/rigid', mode,
                                                       os.path.splitext(os.path.basename(weights[0]))[0]))
    print('Mode:', mode, ' Output folder:', output)

    # Each draw applies one random miscalibration to the whole test sequence.
    range_count = int(os.environ.get('CLR_DRAWS', 25))
    all_RTs = []
    for i in range(range_count):
        print("Iteration: " + bcolors.OKGREEN + str(i+1) + "/" + bcolors.ENDC + str(range_count))
        average_RT = run_eval(dict(_config), weights)
        all_RTs.append(average_RT)
    print("Results")
    print_results(all_RTs, results_path=output)

if __name__ == '__main__':
    main()