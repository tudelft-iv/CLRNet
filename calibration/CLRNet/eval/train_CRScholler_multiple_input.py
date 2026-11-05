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

from CLRNet.datasets.DatasetCLR import DatasetCRVoD
from CLRNet.utils.losses import QuaternionLoss
from models.CLRNet import CRSchollerNet

from CLRNet.utils.utils import quaternion_distance,  merge_inputs, bcolors, read_config
from CLRNet.utils.preprocess import preprocess_sample_CRScholler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

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

def train(model, optimizer, rgb_img, radar_depth, target_rot_CR, loss_fn):
    model.train()
    optimizer.zero_grad()
    rot_CR = model(rgb_img, radar_depth)
    loss = loss_fn(target_rot_CR, rot_CR)
    loss.backward()
    optimizer.step()
    return loss.item()

def val(model, rgb_imgs, radar_depths, target_rot_CR, loss_fn):
    model.eval()
    with torch.no_grad():
        rot_CR = model(rgb_imgs, radar_depths)
    loss = loss_fn(target_rot_CR, rot_CR)
    
    total_rot_error_CR = quaternion_distance(target_rot_CR, rot_CR, target_rot_CR.device)
    total_rot_error_CR = total_rot_error_CR * 180. / math.pi

    return loss.item(), total_rot_error_CR.sum().item()

def main(seed):
    global EPOCH
    _config = read_config('/home/appuser/src/calibration/CLRNet/configs/VOD_CRScholler_config.yaml')

    dataset_class = DatasetCRVoD

    dataset_train = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='train')
    dataset_val = dataset_class(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'], split='val')

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

    model = CRSchollerNet()
    
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)
    model = model.cuda()

    loss_fn = QuaternionLoss()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
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

        ## Training ##
        for batch_idx, sample in enumerate(TrainImgLoader):
            
            rgb_input, radar_input = preprocess_sample_CRScholler(sample, _config)

            loss = train(model, optimizer, rgb_input, radar_input, sample['rot_error'], loss_fn)

            if loss != loss:
                raise ValueError("Loss is NaN")

            local_loss += loss

            if batch_idx % 10 == 0 and batch_idx != 0:
                local_loss = 0.
            total_train_loss += loss * len(sample['rgb'])
            train_iter += 1

        print('Epoch %d, total training loss = %.3f' % (epoch, total_train_loss / len(dataset_train))
              + ', time = %.1f min' % ((time.time() - epoch_start_time) / 60))
        if epoch % 5 == 0:
            ## Validation ##
            total_val_loss = 0.
            total_val_r = 0.

            local_loss = 0.0
            for batch_idx, sample in enumerate(ValImgLoader):
                rgb_input, radar_input = preprocess_sample_CRScholler(sample, _config)
                
                loss, rot_error_CR = val(model, rgb_input, radar_input, sample['rot_error'], loss_fn)
                if loss != loss:
                    raise ValueError("Loss is NaN")
                local_loss += loss

                total_val_loss += loss * len(sample['rgb'])
                total_val_r += rot_error_CR

            avg_val_loss = total_val_loss / len(dataset_val)
            avg_rot_error_CR = total_val_r / len(dataset_val)

            print('Validation loss = %.3f, mean rotation error CR = %.3f degree' %
                  (avg_val_loss, avg_rot_error_CR))

            ## Save best model ##
            if avg_val_loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = avg_val_loss
                save_filename = os.path.join(model_savepath, "epoch_%03d_iter_%08d" % (epoch, train_iter) + ".pth")
                torch.save({
                    'config': _config,
                    'epoch': epoch,
                    # 'state_dict': model.state_dict(), # single gpu
                    'state_dict': model.module.state_dict(), # multi gpu
                    'optimizer': optimizer.state_dict(),
                    'train_loss': total_train_loss / len(dataset_train),
                    'val_loss': total_val_loss / len(dataset_val),
                }, save_filename)
                print("Model saved as %s" % save_filename)
                if old_save_filename is not None:
                    if os.path.exists(old_save_filename):
                        os.remove(old_save_filename)
                old_save_filename = save_filename

        scheduler.step()

    print("Total training time: %.1f hour" % ((time.time() - start_full_time) / 3600))
    print('Best validation loss = %.3f' % BEST_VAL_LOSS)

if __name__ == '__main__':   
    main(0)