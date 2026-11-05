import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from models.correlation_package.correlation import Correlation

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        weights = 'ResNet' + str(num_layers) + '_Weights.DEFAULT'
        if pretrained:
            self.encoder = resnets[num_layers](weights=weights)
        else:
            self.encoder = resnets[num_layers](weights=None)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class LRNet(nn.Module):
    def __init__(self, image_size, input_dims=(512, 1024), use_feat_from=1, md=4, use_reflectance_RADAR=False, use_reflectance_LIDAR=False, use_velocity_RADAR=False, use_time_RADAR = False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):

        super(LRNet, self).__init__()

        self.input_dims = input_dims
        input_lidar = 1
        input_radar = 1
        self.res_num = res_num
        self.use_feat_from = use_feat_from
        if use_reflectance_LIDAR: input_lidar += 1
        if use_reflectance_RADAR: input_radar += 1
        if use_velocity_RADAR: input_radar += 1
        if use_time_RADAR: input_radar += 1

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64

        block1 = BasicBlock
        block2 = BasicBlock

        # lidar
        self.conv1_lidar1 = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.leakyRELU_lidar1 = nn.LeakyReLU(0.1)
        self.maxpool_lidar1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar1 = self._make_layer(block1, 64, 2)
        self.layer2_lidar1 = self._make_layer(block1, 128, 2, stride=2)
        self.layer3_lidar1 = self._make_layer(block1, 256, 2, stride=2)
        self.layer4_lidar1 = self._make_layer(block1, 512, 2, stride=2)

        # radar
        self.inplanes = 64
        self.conv1_lidar2 = nn.Conv2d(input_radar, 64, kernel_size=7, stride=2, padding=3)
        self.leakyRELU_lidar2 = nn.LeakyReLU(0.1)
        self.maxpool_lidar2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar2 = self._make_layer(block2, 64, 2)
        self.layer2_lidar2 = self._make_layer(block2, 128, 2, stride=2)
        self.layer3_lidar2 = self._make_layer(block2, 256, 2, stride=2)
        self.layer4_lidar2 = self._make_layer(block2, 512, 2, stride=2)


        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        fc_size = od + dd[4]
        downsample = 128 // (2**use_feat_from)
        fc_size *= ((input_dims[0] // downsample) * (input_dims[1] // downsample))
        self.fc1 = nn.Linear(fc_size * 4, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, lidar1, lidar2):
        #encoder
        # lidar_image1
        x1 = self.conv1_lidar1(lidar1)
        c12 = self.leakyRELU_lidar1(x1)  # 2
        c13 = self.layer1_lidar1(self.maxpool_lidar1(c12))  # 4
        c14 = self.layer2_lidar1(c13)  # 8
        c15 = self.layer3_lidar1(c14)  # 16
        c16 = self.layer4_lidar1(c15)  # 32

        # lidar_image2
        x2 = self.conv1_lidar2(lidar2)
        c22 = self.leakyRELU_lidar2(x2)  # 2
        c23 = self.layer1_lidar2(self.maxpool_lidar2(c22))  # 4
        c24 = self.layer2_lidar2(c23)  # 8
        c25 = self.layer3_lidar2(c24)  # 16
        c26 = self.layer4_lidar2(c25)  # 32

        # correlation
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot

class LLNet(nn.Module):
    def __init__(self, image_size,  input_dims=(512, 1024), use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        super(LLNet, self).__init__()
        input_lidar = 1
        self.res_num = res_num
        self.use_feat_from = use_feat_from
        if use_reflectance:
            input_lidar = 2

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64

        block1 = BasicBlock
        block2 = BasicBlock

        # lidar1
        self.conv1_lidar1 = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.leakyRELU_lidar1 = nn.LeakyReLU(0.1)
        self.maxpool_lidar1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar1 = self._make_layer(block1, 64, 2)
        self.layer2_lidar1 = self._make_layer(block1, 128, 2, stride=2)
        self.layer3_lidar1 = self._make_layer(block1, 256, 2, stride=2)
        self.layer4_lidar1 = self._make_layer(block1, 512, 2, stride=2)

        # lidar2
        self.inplanes = 64
        self.conv1_lidar2 = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.leakyRELU_lidar2 = nn.LeakyReLU(0.1)
        self.maxpool_lidar2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar2 = self._make_layer(block2, 64, 2)
        self.layer2_lidar2 = self._make_layer(block2, 128, 2, stride=2)
        self.layer3_lidar2 = self._make_layer(block2, 256, 2, stride=2)
        self.layer4_lidar2 = self._make_layer(block2, 512, 2, stride=2)


        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        fc_size = od + dd[4]
        downsample = 128 // (2**use_feat_from)
        fc_size *= ((input_dims[0] // downsample) * (input_dims[1] // downsample))
        self.fc1 = nn.Linear(fc_size * 4, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, lidar1, lidar2):
        #encoder
        # lidar_image1
        x1 = self.conv1_lidar1(lidar1)
        c12 = self.leakyRELU_lidar1(x1)  # 2
        c13 = self.layer1_lidar1(self.maxpool_lidar1(c12))  # 4
        c14 = self.layer2_lidar1(c13)  # 8
        c15 = self.layer3_lidar1(c14)  # 16
        c16 = self.layer4_lidar1(c15)  # 32

        # lidar_image2
        x2 = self.conv1_lidar2(lidar2)
        c22 = self.leakyRELU_lidar2(x2)  # 2
        c23 = self.layer1_lidar2(self.maxpool_lidar2(c22))  # 4
        c24 = self.layer2_lidar2(c23)  # 8
        c25 = self.layer3_lidar2(c24)  # 16
        c26 = self.layer4_lidar2(c25)  # 32

        # correlation
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot

class Encoder(nn.Module):
    def __init__(self, input_channels, block, layers):
        super(Encoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class FCLayer(nn.Module):
    def __init__(self, fc_size=128, dropout=0.5, dim=1):
        super(FCLayer, self).__init__()
        
        self.fc1 = nn.Linear(fc_size * 4 * dim, 512)
        self.fc1_transl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_transl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)
        self.dropout = nn.Dropout(dropout)
        self.leakyRELU = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))
        
        transl = self.leakyRELU(self.fc1_transl(x))
        transl = self.fc2_transl(transl)
        
        rot = self.leakyRELU(self.fc1_rot(x))
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)
        
        return transl, rot
    
class NaiveTransformationRegressor(nn.Module):
    def __init__(self, input_dim):
        super(NaiveTransformationRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 21)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CLRNet(nn.Module):
    def __init__(self, input_dims=(512, 1024), use_feat_from=1, md=4,
                 use_reflectance_LIDAR=False, use_reflectance_RADAR=False, use_velocity_RADAR=False,
                 use_time_RADAR=False, use_depth=False, dropout=0.0, res_num=18, shared_fc=True):
        super(CLRNet, self).__init__()
        self.use_feat_from = use_feat_from
        self.shared_fc = shared_fc
        input_lidar = 2 if use_reflectance_LIDAR else 1
        input_radar = 1 + use_reflectance_RADAR + use_velocity_RADAR + use_time_RADAR

        self.res_num = res_num
        self.pretrained_encoder = True
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        if use_depth:
            self.depth_encoder = Encoder(1, BasicBlock, [2, 2, 2, 2])
            self.conv1_concat = nn.Conv2d(1024, 512, kernel_size=1)

        self.lidar_encoder = Encoder(input_lidar, BasicBlock, [2, 2, 2, 2])
        self.radar_encoder = Encoder(input_radar, BasicBlock, [2, 2, 2, 2])

        self.corr_CL = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.corr_LR = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.corr_CR = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)

        self.leakyRELU_CL = nn.LeakyReLU(0.1)
        self.leakyRELU_LR = nn.LeakyReLU(0.1)
        self.leakyRELU_CR = nn.LeakyReLU(0.1)

        self.corr_conv_CL = self._make_corr_conv_layers(md)
        self.corr_conv_LR = self._make_corr_conv_layers(md)
        self.corr_conv_CR = self._make_corr_conv_layers(md)

        fc_size = (2 * md + 1) ** 2 + np.cumsum([128, 128, 96, 64, 32])[-1]
        downsample = 128 // (2 ** use_feat_from)
        fc_size *= ((input_dims[0] // downsample) * (input_dims[1] // downsample))

        if shared_fc:
            self.fc_CL = FCLayer(fc_size, dropout, dim=3)
            self.fc_LR = FCLayer(fc_size, dropout, dim=3)
            self.fc_CR = FCLayer(fc_size, dropout, dim=3)
        else:
            self.fc_CL = FCLayer(fc_size, dropout, dim=1)
            self.fc_LR = FCLayer(fc_size, dropout, dim=1)
            self.fc_CR = FCLayer(fc_size, dropout, dim=1)

        self._initialize_weights()

    def _make_corr_conv_layers(self, md):
        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        layers = [
            myconv(nd, 128, kernel_size=3, stride=1),
            myconv(nd + dd[0], 128, kernel_size=3, stride=1),
            myconv(nd + dd[1], 96, kernel_size=3, stride=1),
            myconv(nd + dd[2], 64, kernel_size=3, stride=1),
            myconv(nd + dd[3], 32, kernel_size=3, stride=1),
        ]
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb_list, lidar_list, radar_list, rgb_depth_list=None):
        rgb_feature_list = [None] * rgb_list.shape[1]
        lidar_feature_list = [None] * lidar_list.shape[1]
        radar_feature_list = [None] * radar_list.shape[1]

        for i in range(rgb_list.shape[1]):
            rgb = rgb_list[:, i, :, :, :]
            lidar = lidar_list[:, i, :, :, :]
            radar = radar_list[:, i, :, :, :]
            rgb_feature_list[i] = self.net_encoder(rgb)[5]
            lidar_feature_list[i] = self.lidar_encoder(lidar)
            radar_feature_list[i] = self.radar_encoder(radar)
            if rgb_depth_list is not None:
                rgb_depth = rgb_depth_list[:, i, :, :, :]
                depth_feature = self.depth_encoder(rgb_depth)
                rgb_feature_list[i] = self.conv1_concat(torch.cat((rgb_feature_list[i], depth_feature), dim=1))

        concatenated_rgb_features = torch.cat(rgb_feature_list, dim=1)
        concatenated_lidar_features = torch.cat(lidar_feature_list, dim=1)
        concatenated_radar_features = torch.cat(radar_feature_list, dim=1)

        x_CL = self._correlate(self.corr_CL, self.leakyRELU_CL, self.corr_conv_CL, concatenated_rgb_features, concatenated_lidar_features)
        x_LR = self._correlate(self.corr_LR, self.leakyRELU_LR, self.corr_conv_LR, concatenated_lidar_features, concatenated_radar_features)
        x_CR = self._correlate(self.corr_CR, self.leakyRELU_CR, self.corr_conv_CR, concatenated_rgb_features, concatenated_radar_features)

        concatenated_x = torch.cat((x_CL, x_LR, x_CR), dim=1)

        if self.shared_fc:
            transl_CL, rot_CL = self.fc_CL(concatenated_x)
            transl_LR, rot_LR = self.fc_LR(concatenated_x)
            transl_CR, rot_CR = self.fc_CR(concatenated_x)
        else:
            transl_CL, rot_CL = self.fc_CL(x_CL)
            transl_LR, rot_LR = self.fc_LR(x_LR)
            transl_CR, rot_CR = self.fc_CR(x_CR)

        return transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR

    def _correlate(self, corr_layer, leakyrelu_layer, corr_conv_layers, feat1, feat2):
        corr = corr_layer(feat1, feat2)
        corr = leakyrelu_layer(corr)
        x = torch.cat((corr_conv_layers[0](corr), corr), 1)
        x = torch.cat((corr_conv_layers[1](x), x), 1)
        x = torch.cat((corr_conv_layers[2](x), x), 1)
        x = torch.cat((corr_conv_layers[3](x), x), 1)
        x = torch.cat((corr_conv_layers[4](x), x), 1)

        x = x.view(x.size(0), -1)
        return x

class CLRHayoun(nn.Module):
    def __init__(self):
        super(CLRHayoun, self).__init__()
        self.rgbencoder = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=1)
        self.radarencoder = Encoder(1, BasicBlock, [2, 2, 2, 2])
        self.lidarencoder = Encoder(1, BasicBlock, [2, 2, 2, 2])
        self.regressor = NaiveTransformationRegressor(512 * 512 * 3)

    def forward(self, rgb_list, lidar_list, radar_list, rgb_depth_list=None):
        # Prepare lists to store features from multiple inputs
        rgb_feature_list = [None] * rgb_list.shape[1]
        lidar_feature_list = [None] * lidar_list.shape[1]
        radar_feature_list = [None] * radar_list.shape[1]

        for i in range(rgb_list.shape[1]):  # Iterate over the number of input sequences
            rgb = rgb_list[:, i, :, :, :]  # Get i-th RGB input
            lidar = lidar_list[:, i, :, :, :]  # Get i-th lidar input
            radar = radar_list[:, i, :, :, :]  # Get i-th radar input

            # Extract features from each modality
            rgb_feature_list[i] = self.rgbencoder(rgb)[5]  # Extract RGB features (assuming feature map at [5])
            lidar_feature_list[i] = self.lidarencoder(lidar)  # Extract lidar features
            radar_feature_list[i] = self.radarencoder(radar)  # Extract radar features

        # Concatenate features across multiple inputs
        concatenated_rgb_features = torch.cat(rgb_feature_list, dim=1)
        concatenated_lidar_features = torch.cat(lidar_feature_list, dim=1)
        concatenated_radar_features = torch.cat(radar_feature_list, dim=1)

        # Combine the concatenated features from all modalities
        x = torch.cat((concatenated_rgb_features, concatenated_lidar_features, concatenated_radar_features), dim=1)
        x = x.view(x.size(0), -1)  # Flatten the combined features for regression

        
        # Pass through regressor to predict transformations
        transformations = self.regressor(x)

        # Split transformations into the required sensor pairs
        transl_CL = transformations[:, 0:3]
        rot_CL = transformations[:, 3:7]
        transl_LR = transformations[:, 7:10]
        rot_LR = transformations[:, 10:14]
        transl_CR = transformations[:, 14:17]
        rot_CR = transformations[:, 17:21]
        
        return transl_CL, rot_CL, transl_LR, rot_LR, transl_CR, rot_CR

class CRSchollerNet(nn.Module):
    def __init__(self, dropout=0.0):
        super(CRSchollerNet, self).__init__()

        # RGB stream using MobileNet cropped after the third depthwise convolution layer
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        mobilenet = mobilenet_v2(weights=weights)
        self.rgb_encoder = nn.Sequential(*list(mobilenet.features)[:7])  
        
        # MlpConv layers for the RGB stream
        self.mlpconv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1)
        )
        self.mlpconv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=1)
        )

        # Radar stream
        self.radar_encoder = nn.Sequential(
            nn.MaxPool2d(2)
        )

        # Fully connected layers for latent vectors
        self.fc_rgb = nn.Linear(16 * 19 * 30, 50)  # Updated for input image size (150, 240)
        self.fc_radar = nn.Linear(1 * 75 * 120, 50)  # Sparse matrix size (150, 240) after pooling
        
        # Regression block
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        
        # Activations
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, rgb_input, radar_input):
        # Process RGB stream
        rgb_feature = self.rgb_encoder(rgb_input)
        rgb_feature = self.mlpconv1(rgb_feature)
        rgb_feature = self.mlpconv2(rgb_feature)
        rgb_feature = rgb_feature.view(rgb_feature.size(0), -1)
        rgb_latent = self.fc_rgb(rgb_feature)

        # Process radar stream
        radar_feature = self.radar_encoder(radar_input)
        radar_feature = radar_feature.view(radar_feature.size(0), -1)
        radar_latent = self.fc_radar(radar_feature)

        # Concatenate latent vectors
        latent_vector = torch.cat((rgb_latent, radar_latent), dim=1)

        # Regression block
        x = self.prelu(self.fc1(latent_vector))
        x = self.dropout(x)
        x = self.prelu(self.fc2(x))
        quaternion = self.fc3(x)  # Output quaternion

        return quaternion

class CRNet(nn.Module):
    def __init__(self, input_dims=(512, 1024), use_feat_from=1, md=4,
                 use_reflectance_RADAR=False, use_velocity_RADAR=False, use_time_RADAR=False,
                 use_depth=False, dropout=0.0, res_num=18):
        super(CRNet, self).__init__()
        self.use_feat_from = use_feat_from
        input_radar = 1 + use_reflectance_RADAR + use_velocity_RADAR + use_time_RADAR

        self.res_num = res_num
        self.pretrained_encoder = True
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        if use_depth:
            self.depth_encoder = Encoder(1, BasicBlock, [2, 2, 2, 2])
            self.conv1_concat = nn.Conv2d(1024, 512, kernel_size=1)

        self.radar_encoder = Encoder(input_radar, BasicBlock, [2, 2, 2, 2])

        self.corr_CR = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU_CR = nn.LeakyReLU(0.1)
        self.corr_conv_CR = self._make_corr_conv_layers(md)

        fc_size = (2 * md + 1) ** 2 + np.cumsum([128, 128, 96, 64, 32])[-1]
        downsample = 128 // (2 ** use_feat_from)
        fc_size *= ((input_dims[0] // downsample) * (input_dims[1] // downsample))

        self.fc_CR = FCLayer(fc_size, dropout)

        self._initialize_weights()

    def _make_corr_conv_layers(self, md):
        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        layers = [
            myconv(nd, 128, kernel_size=3, stride=1),
            myconv(nd + dd[0], 128, kernel_size=3, stride=1),
            myconv(nd + dd[1], 96, kernel_size=3, stride=1),
            myconv(nd + dd[2], 64, kernel_size=3, stride=1),
            myconv(nd + dd[3], 32, kernel_size=3, stride=1),
        ]
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb_list, radar_list, rgb_depth_list=None):
        rgb_feature_list = [None] * rgb_list.shape[1]
        radar_feature_list = [None] * radar_list.shape[1]

        for i in range(rgb_list.shape[1]):
            rgb = rgb_list[:, i, :, :, :]
            radar = radar_list[:, i, :, :, :]
            rgb_feature_list[i] = self.net_encoder(rgb)[5]
            radar_feature_list[i] = self.radar_encoder(radar)
            if rgb_depth_list is not None:
                rgb_depth = rgb_depth_list[:, i, :, :, :]
                depth_feature = self.depth_encoder(rgb_depth)
                rgb_feature_list[i] = self.conv1_concat(torch.cat((rgb_feature_list[i], depth_feature), dim=1))

        concatenated_rgb_features = torch.cat(rgb_feature_list, dim=1)
        concatenated_radar_features = torch.cat(radar_feature_list, dim=1)

        x_CR = self._correlate(self.corr_CR, self.leakyRELU_CR, self.corr_conv_CR, concatenated_rgb_features, concatenated_radar_features)

        transl_CR, rot_CR = self.fc_CR(x_CR)

        return transl_CR, rot_CR

    def _correlate(self, corr_layer, leakyrelu_layer, corr_conv_layers, feat1, feat2):
        corr = corr_layer(feat1, feat2)
        corr = leakyrelu_layer(corr)
        x = torch.cat((corr_conv_layers[0](corr), corr), 1)
        x = torch.cat((corr_conv_layers[1](x), x), 1)
        x = torch.cat((corr_conv_layers[2](x), x), 1)
        x = torch.cat((corr_conv_layers[3](x), x), 1)
        x = torch.cat((corr_conv_layers[4](x), x), 1)

        x = x.view(x.size(0), -1)
        return x
    
class CLNet(nn.Module):
    def __init__(self, input_dims=(512, 1024), use_feat_from=1, md=4,
                 use_reflectance_LIDAR=True,
                 use_depth=False, dropout=0.0, res_num=18):
        super(CLNet, self).__init__()
        self.use_feat_from = use_feat_from
        input_radar = 1 + use_reflectance_LIDAR

        self.res_num = res_num
        self.pretrained_encoder = True
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        if use_depth:
            self.depth_encoder = Encoder(1, BasicBlock, [2, 2, 2, 2])
            self.conv1_concat = nn.Conv2d(1024, 512, kernel_size=1)

        self.radar_encoder = Encoder(input_radar, BasicBlock, [2, 2, 2, 2])

        self.corr_CR = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU_CR = nn.LeakyReLU(0.1)
        self.corr_conv_CR = self._make_corr_conv_layers(md)

        fc_size = (2 * md + 1) ** 2 + np.cumsum([128, 128, 96, 64, 32])[-1]
        downsample = 128 // (2 ** use_feat_from)
        fc_size *= ((input_dims[0] // downsample) * (input_dims[1] // downsample))

        self.fc_CR = FCLayer(fc_size, dropout)

        self._initialize_weights()

    def _make_corr_conv_layers(self, md):
        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        layers = [
            myconv(nd, 128, kernel_size=3, stride=1),
            myconv(nd + dd[0], 128, kernel_size=3, stride=1),
            myconv(nd + dd[1], 96, kernel_size=3, stride=1),
            myconv(nd + dd[2], 64, kernel_size=3, stride=1),
            myconv(nd + dd[3], 32, kernel_size=3, stride=1),
        ]
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb_list, radar_list, rgb_depth_list=None):
        rgb_feature_list = [None] * rgb_list.shape[1]
        radar_feature_list = [None] * radar_list.shape[1]

        for i in range(rgb_list.shape[1]):
            rgb = rgb_list[:, i, :, :, :]
            radar = radar_list[:, i, :, :, :]
            rgb_feature_list[i] = self.net_encoder(rgb)[5]
            radar_feature_list[i] = self.radar_encoder(radar)
            if rgb_depth_list is not None:
                rgb_depth = rgb_depth_list[:, i, :, :, :]
                depth_feature = self.depth_encoder(rgb_depth)
                rgb_feature_list[i] = self.conv1_concat(torch.cat((rgb_feature_list[i], depth_feature), dim=1))

        concatenated_rgb_features = torch.cat(rgb_feature_list, dim=1)
        concatenated_radar_features = torch.cat(radar_feature_list, dim=1)

        x_CR = self._correlate(self.corr_CR, self.leakyRELU_CR, self.corr_conv_CR, concatenated_rgb_features, concatenated_radar_features)

        transl_CR, rot_CR = self.fc_CR(x_CR)

        return transl_CR, rot_CR

    def _correlate(self, corr_layer, leakyrelu_layer, corr_conv_layers, feat1, feat2):
        corr = corr_layer(feat1, feat2)
        corr = leakyrelu_layer(corr)
        x = torch.cat((corr_conv_layers[0](corr), corr), 1)
        x = torch.cat((corr_conv_layers[1](x), x), 1)
        x = torch.cat((corr_conv_layers[2](x), x), 1)
        x = torch.cat((corr_conv_layers[3](x), x), 1)
        x = torch.cat((corr_conv_layers[4](x), x), 1)

        x = x.view(x.size(0), -1)
        return x

class LCCNet(nn.Module):
    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        super(LCCNet, self).__init__()
        input_lidar = 1
        self.res_num = res_num
        self.use_feat_from = use_feat_from

        # original resnet
        self.pretrained_encoder = True
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64
        if self.res_num == 50:
            layers = [3, 4, 6, 3]
        elif self.res_num == 18:
            layers = [2, 2, 2, 2]

        block = BasicBlock

        # rgb_image
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu_rgb = nn.ELU()
        self.leakyRELU_rgb = nn.LeakyReLU(0.1)
        self.maxpool_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_rgb = self._make_layer(block, 64, layers[0])
        self.layer2_rgb = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_rgb = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_rgb = self._make_layer(block, 512, layers[3], stride=2)

        # lidar_image
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        fc_size = od + dd[4]
        downsample = 128 // (2**use_feat_from)
        if image_size[0] % downsample == 0:
            fc_size *= image_size[0] // downsample
        else:
            fc_size *= (image_size[0] // downsample)+1
        if image_size[1] % downsample == 0:
            fc_size *= image_size[1] // downsample
        else:
            fc_size *= (image_size[1] // downsample)+1
        self.fc1 = nn.Linear(fc_size * 4, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, rgb, lidar):
        H, W = rgb.shape[2:4]

        #encoder
        if self.pretrained_encoder:
            # rgb_image
            features1 = self.net_encoder(rgb)
            c12 = features1[0]  # 2
            c13 = features1[2]  # 4
            c14 = features1[3]  # 8
            c15 = features1[4]  # 16
            c16 = features1[5]  # 32
            # lidar_image
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c22 = self.elu_lidar(x2)  # 2
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4
            c24 = self.layer2_lidar(c23)  # 8
            c25 = self.layer3_lidar(c24)  # 16
            c26 = self.layer4_lidar(c25)  # 32

        else:
            x1 = self.conv1_rgb(rgb)
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c12 = self.leakyRELU_rgb(x1)  # 2
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c12 = self.elu_rgb(x1)  # 2
                c22 = self.elu_lidar(x2)  # 2
            c13 = self.layer1_rgb(self.maxpool_rgb(c12))  # 4
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4
            c14 = self.layer2_rgb(c13)  # 8
            c24 = self.layer2_lidar(c23)  # 8
            c15 = self.layer3_rgb(c14)  # 16
            c25 = self.layer3_lidar(c24)  # 16
            c16 = self.layer4_rgb(c15)  # 32
            c26 = self.layer4_lidar(c25)  # 32

        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        return transl, rot