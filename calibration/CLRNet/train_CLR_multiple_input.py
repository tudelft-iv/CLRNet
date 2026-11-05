import math
import os
import random
import time

import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from CLRNet.datasets.DatasetCLR import DatasetCLRVoD
from CLRNet.utils.losses import CLRLoss, CLRLoopLoss
from models.CLRNet import CLRNet, CLRHayoun

from CLRNet.utils.utils import merge_inputs, quaternion_distance, bcolors, read_config
from CLRNet.utils.preprocess import preprocess_sample_CLR


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("CLRCalib")
ex.captured_out_filter = apply_backspaces_and_linefeeds

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if device == torch.device("cuda"):
    print(f"Device: {bcolors.OKGREEN}{device}{bcolors.ENDC}")
else:
    print(f"Device: {bcolors.FAIL}{device}{bcolors.ENDC}")

EPOCH = 1
def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH*100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@ex.capture
def train(model, optimizer, rgb_img, lidar_depth, radar_depth, rgb_depth_input, target_transl_CL, target_rot_CL, target_transl_CR, target_rot_CR, target_transl_LR, target_rot_LR, loss_fn, lidar_pc, epoch_norm):
    model.train()

    optimizer.zero_grad()

    transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR = model(rgb_img, lidar_depth, radar_depth, rgb_depth_input)

    losses = loss_fn(lidar_pc, target_transl_CL, target_rot_CL, target_transl_LR, target_rot_LR, target_transl_CR, target_rot_CR, transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR, epoch_norm)
    losses['total_loss'].backward()
    optimizer.step()

    return losses

@ex.capture
def val(model, rgb_imgs, lidar_depths, radar_depths, rgb_depth_inputs, target_transl_CL, target_rot_CL, target_transl_CR, target_rot_CR, target_transl_LR, target_rot_LR, loss_fn, lidar_pc, epoch_norm):
    model.eval()

    # Run model
    with torch.no_grad():
        transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR = model(rgb_imgs, lidar_depths, radar_depths, rgb_depth_inputs)

    losses = loss_fn(lidar_pc, target_transl_CL, target_rot_CL, target_transl_LR, target_rot_LR, target_transl_CR, target_rot_CR, transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR, epoch_norm)

    total_transl_error_CL = torch.tensor(0.0).cuda()
    total_rot_error_CL = quaternion_distance(target_rot_CL, rot_CL, target_rot_CL.device)
    total_rot_error_CL = total_rot_error_CL * 180. / math.pi

    for j in range(rgb_imgs.shape[0]):
        total_transl_error_CL += torch.norm(target_transl_CL[j] - transl_CL[j]) * 100.

    total_transl_error_CR = torch.tensor(0.0).cuda()
    total_rot_error_CR = quaternion_distance(target_rot_CR, rot_CR, target_rot_CR.device)
    total_rot_error_CR = total_rot_error_CR * 180. / math.pi

    for j in range(rgb_imgs.shape[0]):
        total_transl_error_CR += torch.norm(target_transl_CR[j] - transl_CR[j]) * 100.

    total_transl_error_LR = torch.tensor(0.0).cuda()
    total_rot_error_LR = quaternion_distance(target_rot_LR, rot_LR, target_rot_LR.device)
    total_rot_error_LR = total_rot_error_LR * 180. / math.pi

    for j in range(rgb_imgs.shape[0]):
        total_transl_error_LR += torch.norm(target_transl_LR[j] - transl_LR[j]) * 100.

    return losses, total_transl_error_CL.item(), total_rot_error_CL.sum().item(), total_transl_error_LR.item(), total_rot_error_LR.sum().item(), total_transl_error_CR.item(), total_rot_error_CR.sum().item()

@ex.automain
def main(_run, seed):
    global EPOCH
    _config = read_config(os.environ.get('CLR_CONFIG', '/home/appuser/src/calibration/CLRNet/configs/VOD_CLR_config.yaml'))

    print('LIDAR: Using Reflectance: {}'.format(_config['use_reflectance_LIDAR']))
    print('RADAR: Using Reflectance: {}'.format(_config['use_reflectance_RADAR']))
    print('RADAR: Using Velocity: {}'.format(_config['use_velocity_RADAR']))
    print('RADAR: Using Time: {}'.format(_config['use_time_RADAR']))
    print('Using Depth: {}'.format(_config['use_depth']))
    print('Using multiple input: {}'.format(_config['multiple_input']))

    dataset_class = DatasetCLRVoD

    dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                  split='train')
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                split='val')


    model_savepath = os.path.join(_config['checkpoints'], 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def init_fn(x): return _init_fn(x, seed)

    train_dataset_size = len(dataset_train)
    val_dataset_size = len(dataset_val)
    print('Size of the train dataset: {}'.format(train_dataset_size))
    print('Size of the val dataset: {}'.format(val_dataset_size))

    # Training and validation set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=num_worker,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)


    if _config['loss'] == 'clr':
        loss_fn = CLRLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    elif _config['loss'] == 'clr_loop':
        loss_fn = CLRLoopLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'], _config['lambda_loop'])
    else:
        raise ValueError("Unknown Loss Function")

    model = CLRNet(input_dims = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4,
                    use_reflectance_LIDAR=_config['use_reflectance_LIDAR'], use_reflectance_RADAR=_config['use_reflectance_RADAR'], use_velocity_RADAR = _config['use_velocity_RADAR'], use_time_RADAR=_config['use_time_RADAR'], use_depth=_config['use_depth'],
                    dropout=_config['dropout'], res_num=18)
    
    # model = CLRHayoun()

    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu', weights_only=True)
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)
    model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
    # Probably this scheduler is not used
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu', weights_only=True)
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0:
            starting_epoch = checkpoint['epoch']

    start_full_time = time.time()
    BEST_VAL_LOSS = 10000.
    old_save_filename = None

    train_iter = 0
    val_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        # print('This is epoch: %d' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        total_loop_loss = 0
        total_CL_loss = 0
        total_CR_loss = 0
        total_LR_loss = 0

        # epoch_norm is the ramp fraction in [0,1]; the loop weight applied in the loss is
        # self.lambda_loop * epoch_norm, so a constant configuration (epoch_norm = 1.0) yields an
        # effective loop weight equal to the configured lambda_loop.
        if _config['lambda_loop_increase']:
            epoch_norm = epoch / _config['epochs']
        else:
            epoch_norm = 1.0

        local_loss = 0.

        _run.log_scalar("LR", scheduler.get_last_lr()[0])

        ## Training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            
            rgb_input, lidar_input, radar_input, rgb_depth_input = preprocess_sample_CLR(sample, _config)

            loss = train(model, optimizer, rgb_input, lidar_input, radar_input, rgb_depth_input,
                                                   sample['tr_error_CL'], sample['rot_error_CL'],
                                                   sample['tr_error_CR'], sample['rot_error_CR'],
                                                   sample['tr_error_LR'], sample['rot_error_LR'],
                                                   loss_fn, sample['lidar'], epoch_norm)

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            local_loss += loss['total_loss'].item()

            if batch_idx % 10 == 0 and batch_idx != 0:
                _run.log_scalar("Loss", local_loss/10, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            if _config['loss'] == 'clr_loop' or _config['loss'] == 'clr_loop_woPC':
                total_loop_loss += loss['loop_closure_loss'].item() * len(sample['rgb'])
            total_CL_loss += loss['total_loss_CL'].item() * len(sample['rgb'])
            total_CR_loss += loss['total_loss_CR'].item() * len(sample['rgb'])
            total_LR_loss += loss['total_loss_LR'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print('epoch %d: Total epoch time: %.2f, total training loss = %.3f, loop_loss = %.3f, CL_loss = %.3f, CR_loss = %.3f, LR_loss = %.3f' %
              (epoch, time.time() - epoch_start_time, total_train_loss / len(dataset_train), total_loop_loss / len(dataset_train), total_CL_loss / len(dataset_train), total_CR_loss / len(dataset_train), total_LR_loss / len(dataset_train)))

        if epoch % 5 == 0:
            ## Validation ##
            total_val_loss = 0.
            total_loop_loss = 0.
            total_CL_loss = 0.
            total_CR_loss = 0.
            total_LR_loss = 0.

            total_val_t = 0.
            total_val_r = 0.
            total_val_t_CL = 0.
            total_val_r_CL = 0.
            total_val_t_CR = 0.
            total_val_r_CR = 0.
            total_val_t_LR = 0.
            total_val_r_LR = 0.

            local_loss = 0.0
            for batch_idx, sample in enumerate(ValImgLoader):
                rgb_input, lidar_input, radar_input, rgb_depth_input = preprocess_sample_CLR(sample, _config)

                loss, transl_e_CL, rot_e_CL, transl_e_LR, rot_e_LR, transl_e_CR, rot_e_CR = val(model, rgb_input, lidar_input, radar_input, rgb_depth_input,
                                                                    sample['tr_error_CL'], sample['rot_error_CL'],
                                                                    sample['tr_error_CR'], sample['rot_error_CR'],
                                                                    sample['tr_error_LR'], sample['rot_error_LR'],
                                                                    loss_fn, sample['lidar'], epoch_norm)

                for key in loss.keys():
                    if loss[key].item() != loss[key].item():
                        raise ValueError("Loss {} is NaN".format(key))

                total_val_t += transl_e_CL + transl_e_CR + transl_e_LR
                total_val_r += rot_e_CL + rot_e_CR + rot_e_LR
                local_loss += loss['total_loss'].item()

                total_val_t_CL += transl_e_CL
                total_val_r_CL += rot_e_CL
                total_val_t_CR += transl_e_CR
                total_val_r_CR += rot_e_CR
                total_val_t_LR += transl_e_LR
                total_val_r_LR += rot_e_LR

                if batch_idx % 10 == 0 and batch_idx != 0:
                    local_loss = 0.0
                total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
                if _config['loss'] == 'clr_loop' or _config['loss'] == 'clr_loop_woPC':
                    total_loop_loss += loss['loop_closure_loss'].item() * len(sample['rgb'])
                total_CL_loss += loss['total_loss_CL'].item() * len(sample['rgb'])
                total_CR_loss += loss['total_loss_CR'].item() * len(sample['rgb'])
                total_LR_loss += loss['total_loss_LR'].item() * len(sample['rgb'])

                val_iter += 1

            print("------------------------------------")
            print('Total val loss = %.3f, loop_loss = %.3f, CL_loss = %.3f, CR_loss = %.3f, LR_loss = %.3f' %
              (total_val_loss / len(dataset_val), total_loop_loss / len(dataset_val), total_CL_loss / len(dataset_val), total_CR_loss / len(dataset_val), total_LR_loss / len(dataset_val)))

            print('Total translation error: %.3f cm, Total rotation error: %.3f °' % ((total_val_t / len(dataset_val)), (total_val_r / len(dataset_val))))
            print('Camera - Lidar translation error: %.3f cm, Camera - Lidar rotation error: %.3f °' % ((total_val_t_CL / len(dataset_val)), (total_val_r_CL / len(dataset_val))))
            print('Camera - Radar translation error: %.3f cm, Camera - Radar rotation error: %.3f °' % ((total_val_t_CR / len(dataset_val)), (total_val_r_CR / len(dataset_val))))
            print('Lidar - Radar translation error: %.3f cm, Lidar - Radar rotation error: %.3f °' % ((total_val_t_LR / len(dataset_val)), (total_val_r_LR / len(dataset_val))))
            print("------------------------------------")

            _run.log_scalar("Val_Loss", total_val_loss / len(dataset_val), epoch)
            _run.log_scalar("Val_t_error", total_val_t / len(dataset_val), epoch)
            _run.log_scalar("Val_r_error", total_val_r / len(dataset_val), epoch)

            # SAVE
            val_loss = total_val_loss / len(dataset_val)
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                #_run.result = BEST_VAL_LOSS
                if _config['rescale_transl'] > 0:
                    _run.result = total_val_t / len(dataset_val)
                else:
                    _run.result = total_val_r / len(dataset_val)
                savefilename = f'{model_savepath}/checkpoint_clr_Rchannels{_config["use_reflectance_RADAR"]}_depth{_config["use_depth"]}_sph{_config["spherical_projection"]}_mi{_config["multiple_input"]}_l{_config["lambda_loop"]}_r{_config["max_r"]:.1f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.tar'
                torch.save({
                    'config': {**_config, 'model': 'CLRNet', 'dataset': 'VoD', 'use_depth_input': True},
                    'epoch': epoch,
                    # 'state_dict': model.state_dict(), # single gpu
                    'state_dict': model.module.state_dict(), # multi gpu
                    'optimizer': optimizer.state_dict(),
                    'train_loss': total_train_loss / len(dataset_train),
                    'val_loss': total_val_loss / len(dataset_val),
                }, savefilename)
                print(bcolors.OKGREEN + "Model saved as" +bcolors.ENDC)
                print(f'{savefilename}')
                if old_save_filename is not None:
                    if os.path.exists(old_save_filename):
                        os.remove(old_save_filename)
                old_save_filename = savefilename

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    return _run.result, val_loss