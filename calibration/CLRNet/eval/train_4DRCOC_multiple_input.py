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

from CLRNet.datasets.DatasetCLR import DatasetCRVoD
from CLRNet.utils.losses import DRCoCLoss
from models.FourDRCOC import TestNet

from CLRNet.utils.utils import merge_inputs, quaternion_distance, bcolors, read_config
from CLRNet.utils.preprocess import preprocess_sample_CR

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("CameraRadarCalib")
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
    seed = seed + worker_id + EPOCH * 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@ex.capture
def train(model, optimizer, rgb_img, radar_depth, rgb_depth_input, target_transl_CR, target_rot_CR, loss_fn, radar):
    model.train()
    optimizer.zero_grad()
    transl_CR, rot_CR, pred_depth = model(rgb_img, radar_depth)
    losses = loss_fn(radar, target_transl_CR, target_rot_CR, transl_CR, rot_CR, pred_depth, radar_depth)
    losses['total_loss'].backward()
    optimizer.step()
    return losses

@ex.capture
def val(model, rgb_imgs, radar_depths, rgb_depth_inputs, target_transl_CR, target_rot_CR, loss_fn, radar):
    model.eval()
    with torch.no_grad():
        transl_CR, rot_CR = model(rgb_imgs, radar_depths, True)
    pred_depth = None
    losses = loss_fn(radar, target_transl_CR, target_rot_CR, transl_CR, rot_CR, pred_depth, radar_depths)
    
    total_transl_error_CR = torch.tensor(0.0).cuda()
    total_rot_error_CR = quaternion_distance(target_rot_CR, rot_CR, target_rot_CR.device)
    total_rot_error_CR = total_rot_error_CR * 180. / math.pi

    for j in range(rgb_imgs.shape[0]):
        total_transl_error_CR += torch.norm(target_transl_CR[j] - transl_CR[j]) * 100.

    return losses, total_transl_error_CR.item(), total_rot_error_CR.sum().item()

@ex.automain
def main(_run, seed):
    global EPOCH
    _config = read_config('/home/appuser/src/calibration/CLRNet/configs/VOD_4DRCOC_config.yaml')

    dataset_class = DatasetCRVoD

    dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='train', rgb_depth_input=False)
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='val', rgb_depth_input=False)

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

    model = TestNet(image_size = (_config['input_size'][0], _config['input_size'][1]), use_feat_from=1, md=4, attention=False,
                    dropout=_config['dropout'], res_num=18)
    
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu', weights_only=True)
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

    try:
        model = nn.DataParallel(model)
        model = model.cuda()
    except RuntimeError as e:
        print("CUDA Error:", e)


    loss_fn = DRCoCLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])

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
        epoch_start_time = time.time()
        total_train_loss = 0

        local_loss = 0.

        _run.log_scalar("LR", scheduler.get_last_lr()[0])

        ## Training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            
            rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR(sample, _config)
            if _config['use_depth'] is False:
                rgb_depth_input = None

            loss = train(model, optimizer, rgb_input, radar_input, rgb_depth_input,
                                                   sample['tr_error'], sample['rot_error'],
                                                   loss_fn, sample['radar'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            local_loss += loss['total_loss'].item()

            if batch_idx % 10 == 0 and batch_idx != 0:
                _run.log_scalar("Loss", local_loss/10, train_iter)
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print('Epoch %d, total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train))
              + ', time = %.1f min' % ((time.time() - epoch_start_time) / 60))
        if epoch % 5 == 0:
            ## Validation ##
            total_val_loss = 0.
            total_val_t = 0.
            total_val_r = 0.

            local_loss = 0.0
            for batch_idx, sample in enumerate(ValImgLoader):
                rgb_input, radar_input, rgb_depth_input = preprocess_sample_CR(sample, _config)

                if _config['use_depth'] is False:
                    rgb_depth_input = None

                loss, transl_e_CR, rot_e_CR = val(model, rgb_input, radar_input, rgb_depth_input,
                                                  sample['tr_error'], sample['rot_error'],
                                                  loss_fn, sample['radar'])
                
                for key in loss.keys():
                    if loss[key].item() != loss[key].item():
                        raise ValueError("Loss {} is NaN".format(key))
                
                local_loss += loss['total_loss'].item()
                total_val_loss += loss['total_loss'].item() * len(sample['rgb'])
                total_val_t += transl_e_CR
                total_val_r += rot_e_CR
                val_iter += 1

            print('Total validation loss = %.3f' % (total_val_loss / len(dataset_val)))
            print('Total validation transl error = %.1f cm' % (total_val_t / len(dataset_val)))
            print('Total validation rot error = %.2f degree' % (total_val_r / len(dataset_val)))

            # SAVE
            val_loss = total_val_loss / len(dataset_val)
            if val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = val_loss
                #_run.result = BEST_VAL_LOSS
                if _config['rescale_transl'] > 0:
                    _run.result = total_val_t / len(dataset_val)
                else:
                    _run.result = total_val_r / len(dataset_val)
                savefilename = f'{model_savepath}/checkpoint_4DRCOC_r{_config["max_r"]:.1f}_t{_config["max_t"]:.2f}_e{epoch}_{val_loss:.3f}.tar'
                torch.save({
                    'config': _config,
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