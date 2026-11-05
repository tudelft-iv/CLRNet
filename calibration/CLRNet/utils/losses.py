import torch
from torch import nn as nn
import numpy as np

from CLRNet.utils.utils import quat2mat, rotate_forward, tvector2mat, quat2mat, compose_poses, quaternion_from_matrix, invert_translation, invert_quaternion, quaternion_distance

class QuaternionLoss(nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, target_quat, predicted_quat):
        """
        Quaternion Distance Loss
        Args:
            target_quat: target quaternions
            predicted_quat: predicted quaternions

        Returns:
            The mean quaternion distance loss
        """
        loss = quaternion_distance(predicted_quat, target_quat, predicted_quat.device).mean()
        return loss

class DRCoCLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(DRCoCLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err, predict_depth_image, refl_img):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame [(4,N1), (4, N2)] 齐次坐标
            target_transl: groundtruth of the translations  (120, 3)
            target_rot: groundtruth of the rotations        (120, 4)
            transl_err: network estimate of the translations    (120, 3)
            rot_err: network estimate of the rotations          (120, 4)

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()     # 标量
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()  # 标量 w x y z
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)  # (4, N)
            point_cloud_out = point_clouds[i].clone()               # (4, N)

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)  # (4, 4)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)  # (N,)
            error.clamp(100.)   # (N,)
            point_clouds_loss += error.mean()

        if predict_depth_image is not None:
            depth_loss = self.mse_loss(predict_depth_image, refl_img.squeeze(1))
        else:
            depth_loss = 0
        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = (1 - self.weight_point_cloud) * pose_loss +self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0])+depth_loss

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]
        return self.loss

class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error = torch.clamp(error, min=-100.0, max=100.0)
            if error.mean() > 0.0:
                point_clouds_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = (1 - self.weight_point_cloud) * pose_loss +\
                     self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]

        return self.loss
    
class CLRLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CLRLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, lidar_pc, target_transl_CL, target_rot_CL, target_transl_LR, target_rot_LR, target_transl_CR, target_rot_CR, transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR, epoch_norm = None):

        loss_transl_CL = 0.
        loss_transl_CR = 0.
        loss_transl_LR = 0.
        loss_rot_CL = 0.
        loss_rot_CR = 0.
        loss_rot_LR = 0.
        if self.rescale_trans != 0.:
            loss_transl_CL = self.transl_loss(transl_CL, target_transl_CL).sum(1).mean()
            loss_transl_CR = self.transl_loss(transl_CR, target_transl_CR).sum(1).mean()
            loss_transl_LR = self.transl_loss(transl_LR, target_transl_LR).sum(1).mean()
        if self.rescale_rot != 0.:
            loss_rot_CL = quaternion_distance(rot_CL, target_rot_CL, rot_CL.device).mean()
            loss_rot_CR = quaternion_distance(rot_CR, target_rot_CR, rot_CR.device).mean()
            loss_rot_LR = quaternion_distance(rot_LR, target_rot_LR, rot_LR.device).mean()

        pose_loss_CL = self.rescale_trans*loss_transl_CL + self.rescale_rot*loss_rot_CL
        pose_loss_CR = self.rescale_trans*loss_transl_CR + self.rescale_rot*loss_rot_CR
        pose_loss_LR = self.rescale_trans*loss_transl_LR + self.rescale_rot*loss_rot_LR

        point_clouds_loss_CL = torch.tensor([0.0]).to(transl_CL.device)
        point_clouds_loss_CR = torch.tensor([0.0]).to(transl_CR.device)
        point_clouds_loss_LR = torch.tensor([0.0]).to(transl_LR.device)

        for i in range(len(lidar_pc)):
            point_cloud_gt = lidar_pc[i].to(transl_CL.device)
            point_cloud_out_CL = lidar_pc[i].clone()
            point_cloud_out_CR = lidar_pc[i].clone()
            point_cloud_out_LR = lidar_pc[i].clone()

            R_target_CL = quat2mat(target_rot_CL[i])
            T_target_CL = tvector2mat(target_transl_CL[i])
            RT_target_CL = torch.mm(T_target_CL, R_target_CL)

            R_target_CR = quat2mat(target_rot_CR[i])
            T_target_CR = tvector2mat(target_transl_CR[i])
            RT_target_CR = torch.mm(T_target_CR, R_target_CR)

            R_target_LR = quat2mat(target_rot_LR[i])
            T_target_LR = tvector2mat(target_transl_LR[i])
            RT_target_LR = torch.mm(T_target_LR, R_target_LR)

            R_predicted_CL = quat2mat(rot_CL[i])
            T_predicted_CL = tvector2mat(transl_CL[i])
            RT_predicted_CL = torch.mm(T_predicted_CL, R_predicted_CL)

            R_predicted_CR = quat2mat(rot_CR[i])
            T_predicted_CR = tvector2mat(transl_CR[i])
            RT_predicted_CR = torch.mm(T_predicted_CR, R_predicted_CR)

            R_predicted_LR = quat2mat(rot_LR[i])
            T_predicted_LR = tvector2mat(transl_LR[i])
            RT_predicted_LR = torch.mm(T_predicted_LR, R_predicted_LR)

            RT_total_CL = torch.mm(RT_target_CL.inverse(), RT_predicted_CL)
            RT_total_CR = torch.mm(RT_target_CR.inverse(), RT_predicted_CR)
            RT_total_LR = torch.mm(RT_target_LR.inverse(), RT_predicted_LR)

            point_cloud_out_CL = rotate_forward(point_cloud_out_CL, RT_total_CL)
            point_cloud_out_CR = rotate_forward(point_cloud_out_CR, RT_total_CR)
            point_cloud_out_LR = rotate_forward(point_cloud_out_LR, RT_total_LR)

            error_CL = (point_cloud_out_CL - point_cloud_gt).norm(dim=0)
            error_CR = (point_cloud_out_CR - point_cloud_gt).norm(dim=0)
            error_LR = (point_cloud_out_LR - point_cloud_gt).norm(dim=0)
            error_CL = torch.clamp(error_CL, min=-100.0, max=100.0)
            error_CR = torch.clamp(error_CR, min=-100.0, max=100.0)
            error_LR = torch.clamp(error_LR, min=-100.0, max=100.0)
            point_clouds_loss_CL += error_CL.mean()
            point_clouds_loss_CR += error_CR.mean()
            point_clouds_loss_LR += error_LR.mean()

        total_loss_CL = (1 - self.weight_point_cloud) * pose_loss_CL +\
                     self.weight_point_cloud * (point_clouds_loss_CL/target_transl_CL.shape[0])
        total_loss_CR = (1 - self.weight_point_cloud) * pose_loss_CR +\
                        self.weight_point_cloud * (point_clouds_loss_CR/target_transl_CR.shape[0])
        total_loss_LR = (1 - self.weight_point_cloud) * pose_loss_LR +\
                        self.weight_point_cloud * (point_clouds_loss_LR/target_transl_LR.shape[0])
        total_loss = total_loss_CL + total_loss_CR + total_loss_LR
        trans_loss = loss_transl_CL + loss_transl_CR + loss_transl_LR
        rot_loss = loss_rot_CL + loss_rot_CR + loss_rot_LR
        
        self.loss['total_loss_CL'] = total_loss_CL
        self.loss['transl_loss_CL'] = loss_transl_CL
        self.loss['rot_loss_CL'] = loss_rot_CL
        self.loss['point_clouds_loss_CL'] = point_clouds_loss_CL/target_transl_CL.shape[0]

        self.loss['total_loss_CR'] = total_loss_CR
        self.loss['transl_loss_CR'] = loss_transl_CR
        self.loss['rot_loss_CR'] = loss_rot_CR
        self.loss['point_clouds_loss_CR'] = point_clouds_loss_CR/target_transl_CR.shape[0]

        self.loss['total_loss_LR'] = total_loss_LR
        self.loss['transl_loss_LR'] = loss_transl_LR
        self.loss['rot_loss_LR'] = loss_rot_LR
        self.loss['point_clouds_loss_LR'] = point_clouds_loss_LR/target_transl_LR.shape[0]

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = trans_loss
        self.loss['rot_loss'] = rot_loss
        self.loss['point_clouds_loss'] = point_clouds_loss_CL/target_transl_CL.shape[0] +  point_clouds_loss_CR/target_transl_CR.shape[0] + point_clouds_loss_LR/target_transl_LR.shape[0]

        return self.loss

class CLRLoopLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud, lambda_loop=0.5):
        super(CLRLoopLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}
        self.lambda_loop = lambda_loop

    def loop_closure_loss(self, translations, rotations):
        rot = []
        transl = []
        target_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(rotations), device='cuda')
        target_transl = torch.tensor([[0.0, 0.0, 0.0]] * len(translations), device='cuda')
        for i in range(len(translations)):
            final_transformation = compose_poses(translations[i], rotations[i])
            rot.append(quaternion_from_matrix(final_transformation))
            transl.append(final_transformation[:3, 3])

        rot = torch.stack(rot)
        transl = torch.stack(transl)

        loss_transl = 0.
        loss_rot = 0.

        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl, target_transl).sum(1).mean()

        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot, target_rot, rot.device).mean()

        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        return pose_loss

    def forward(self, lidar_pc, target_transl_CL, target_rot_CL, target_transl_LR, target_rot_LR, target_transl_CR, target_rot_CR, transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR, epoch_norm):
        loss_transl_CL = 0.
        loss_transl_CR = 0.
        loss_transl_LR = 0.
        loss_rot_CL = 0.
        loss_rot_CR = 0.
        loss_rot_LR = 0.
        if self.rescale_trans != 0.:
            loss_transl_CL = self.transl_loss(transl_CL, target_transl_CL).sum(1).mean()
            loss_transl_CR = self.transl_loss(transl_CR, target_transl_CR).sum(1).mean()
            loss_transl_LR = self.transl_loss(transl_LR, target_transl_LR).sum(1).mean()
        if self.rescale_rot != 0.:
            loss_rot_CL = quaternion_distance(rot_CL, target_rot_CL, rot_CL.device).mean()
            loss_rot_CR = quaternion_distance(rot_CR, target_rot_CR, rot_CR.device).mean()
            loss_rot_LR = quaternion_distance(rot_LR, target_rot_LR, rot_LR.device).mean()

        pose_loss_CL = self.rescale_trans*loss_transl_CL + self.rescale_rot*loss_rot_CL
        pose_loss_CR = self.rescale_trans*loss_transl_CR + self.rescale_rot*loss_rot_CR
        pose_loss_LR = self.rescale_trans*loss_transl_LR + self.rescale_rot*loss_rot_LR

        point_clouds_loss_CL = torch.tensor([0.0]).to(transl_CL.device)
        point_clouds_loss_CR = torch.tensor([0.0]).to(transl_CR.device)
        point_clouds_loss_LR = torch.tensor([0.0]).to(transl_LR.device)
        point_clouds_loss_CLRC = torch.tensor([0.0]).to(transl_CL.device)

        for i in range(len(lidar_pc)):
            point_cloud_gt = lidar_pc[i].to(transl_CL.device)
            point_cloud_out_CL = lidar_pc[i].clone()
            point_cloud_out_CR = lidar_pc[i].clone()
            point_cloud_out_LR = lidar_pc[i].clone()
            point_cloud_out_CLRC = lidar_pc[i].clone()

            R_target_CL = quat2mat(target_rot_CL[i])
            T_target_CL = tvector2mat(target_transl_CL[i])
            RT_target_CL = torch.mm(T_target_CL, R_target_CL)

            R_target_CR = quat2mat(target_rot_CR[i])
            T_target_CR = tvector2mat(target_transl_CR[i])
            RT_target_CR = torch.mm(T_target_CR, R_target_CR)

            R_target_LR = quat2mat(target_rot_LR[i])
            T_target_LR = tvector2mat(target_transl_LR[i])
            RT_target_LR = torch.mm(T_target_LR, R_target_LR)

            R_predicted_CL = quat2mat(rot_CL[i])
            T_predicted_CL = tvector2mat(transl_CL[i])
            RT_predicted_CL = torch.mm(T_predicted_CL, R_predicted_CL)

            R_predicted_CR = quat2mat(rot_CR[i])
            T_predicted_CR = tvector2mat(transl_CR[i])
            RT_predicted_CR = torch.mm(T_predicted_CR, R_predicted_CR)

            R_predicted_LR = quat2mat(rot_LR[i])
            T_predicted_LR = tvector2mat(transl_LR[i])
            RT_predicted_LR = torch.mm(T_predicted_LR, R_predicted_LR)

            RT_total_CL = torch.mm(RT_target_CL.inverse(), RT_predicted_CL)
            RT_total_CR = torch.mm(RT_target_CR.inverse(), RT_predicted_CR)
            RT_total_LR = torch.mm(RT_target_LR.inverse(), RT_predicted_LR)

            #loop
            RT_predicted_CLRC = torch.mm(torch.mm(RT_predicted_CL, RT_predicted_LR), RT_predicted_CR.inverse())
            RT_target_CLRC = torch.mm(torch.mm(RT_target_CL, RT_target_LR), RT_target_CR.inverse())
            RT_total_CLRC = torch.mm(RT_target_CLRC.inverse(), RT_predicted_CLRC)

            point_cloud_out_CL = rotate_forward(point_cloud_out_CL, RT_total_CL)
            point_cloud_out_CR = rotate_forward(point_cloud_out_CR, RT_total_CR)
            point_cloud_out_LR = rotate_forward(point_cloud_out_LR, RT_total_LR)
            point_cloud_out_CLRC = rotate_forward(point_cloud_out_CLRC, RT_total_CLRC)

            error_CL = (point_cloud_out_CL - point_cloud_gt).norm(dim=0)
            error_CR = (point_cloud_out_CR - point_cloud_gt).norm(dim=0)
            error_LR = (point_cloud_out_LR - point_cloud_gt).norm(dim=0)
            error_CLRC = (point_cloud_out_CLRC - point_cloud_gt).norm(dim=0)
            error_CL = torch.clamp(error_CL, min=-100.0, max=100.0)
            error_CR = torch.clamp(error_CR, min=-100.0, max=100.0)
            error_LR = torch.clamp(error_LR, min=-100.0, max=100.0)
            error_CLRC = torch.clamp(error_CLRC, min=-100.0, max=100.0)
            point_clouds_loss_CL += error_CL.mean()
            point_clouds_loss_CR += error_CR.mean()
            point_clouds_loss_LR += error_LR.mean()
            point_clouds_loss_CLRC += error_CLRC.mean()

        total_loss_CL = (1 - self.weight_point_cloud) * pose_loss_CL +\
                     self.weight_point_cloud * (point_clouds_loss_CL/target_transl_CL.shape[0])
        total_loss_CR = (1 - self.weight_point_cloud) * pose_loss_CR +\
                        self.weight_point_cloud * (point_clouds_loss_CR/target_transl_CR.shape[0])
        total_loss_LR = (1 - self.weight_point_cloud) * pose_loss_LR +\
                        self.weight_point_cloud * (point_clouds_loss_LR/target_transl_LR.shape[0])
        
        #loop closure loss
        transl_RC = [invert_translation(t) for t in transl_CR]
        rot_RC = [invert_quaternion(r) for r in rot_CR]

        translations = list(zip(transl_CL, transl_LR, transl_RC))
        translations = [list(item) for item in translations]
        rotations = list(zip(rot_CL, rot_LR, rot_RC))
        rotations = [list(item) for item in rotations]
        loop_loss =  self.loop_closure_loss(translations, rotations)

        total_loop_loss = (1 - self.weight_point_cloud) * loop_loss +\
                        self.weight_point_cloud * (point_clouds_loss_CLRC/target_transl_LR.shape[0])

        total_loss = (1 - (self.lambda_loop * epoch_norm)) * (total_loss_CL + total_loss_CR + total_loss_LR) + (self.lambda_loop * epoch_norm) * total_loop_loss

        self.loss['loop_closure_loss'] = loop_loss
        
        self.loss['total_loss_CL'] = total_loss_CL
        self.loss['transl_loss_CL'] = loss_transl_CL
        self.loss['rot_loss_CL'] = loss_rot_CL
        self.loss['point_clouds_loss_CL'] = point_clouds_loss_CL/target_transl_CL.shape[0]

        self.loss['total_loss_CR'] = total_loss_CR
        self.loss['transl_loss_CR'] = loss_transl_CR
        self.loss['rot_loss_CR'] = loss_rot_CR
        self.loss['point_clouds_loss_CR'] = point_clouds_loss_CR/target_transl_CR.shape[0]

        self.loss['total_loss_LR'] = total_loss_LR
        self.loss['transl_loss_LR'] = loss_transl_LR
        self.loss['rot_loss_LR'] = loss_rot_LR
        self.loss['point_clouds_loss_LR'] = point_clouds_loss_LR/target_transl_LR.shape[0]

        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl_CL + loss_transl_CR + loss_transl_LR
        self.loss['rot_loss'] = loss_rot_CL + loss_rot_CR + loss_rot_LR
        self.loss['point_clouds_loss'] = point_clouds_loss_CL/target_transl_CL.shape[0] +  point_clouds_loss_CR/target_transl_CR.shape[0] + point_clouds_loss_LR/target_transl_LR.shape[0]

        return self.loss