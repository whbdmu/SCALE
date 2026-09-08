from collections import OrderedDict
from copy import deepcopy

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from torchvision import datasets, models, transforms
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pdb
from spcl.models.dsbn import DSBN2d, DSBN1d
from utils import add_module_after_block
WEIGHT = 0.2
class AdaptiveFilter(nn.Module):
    def __init__(self, channel, gap_size, level=2):
        super().__init__()
        self.avg_gap = nn.AdaptiveAvgPool2d(gap_size)
        self.max_gap = nn.AdaptiveMaxPool2d(gap_size)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.Sigmoid(),
        )

        self.epsilon = 1e-16
        self.level = level
        self.alpha = nn.Parameter(torch.ones(1, channel, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1), requires_grad=True)

    def compute_threshold(self, x_abs, is_source=None):
        x_avg = self.avg_gap(x_abs)
        x_avg = torch.flatten(x_avg, 1)
        x_max = self.max_gap(x_abs)
        x_max = torch.flatten(x_max, 1)

        x_max_scores_1 = self.mlp(x_max) if is_source is None else self._process_with_DSBN(x_max, is_source, self.mlp)
        channel_threshold = (x_avg * x_max_scores_1).unsqueeze(-1).unsqueeze(-1)

        return channel_threshold

    def _process_with_DSBN(self, features, is_source, body):
        for module in body:
            if isinstance(module, DSBN1d):
                features = module(features, is_source)
            else:
                features = module(features)
        return features

    def apply_filter(self, x_abs, threshold):
        up_mask = (x_abs > threshold).float()  # Mask tensor
        up_sub = x_abs * up_mask  # Raw hard threshold filter
        sqrt_p = torch.pow(
            F.relu(torch.pow(up_sub, self.level) - torch.pow(threshold, self.level)) + self.epsilon,
            1.0 / self.level,
        )  # High-Order Soft Threshold filter
        return (self.alpha * F.relu(up_sub - threshold) + self.beta * (sqrt_p - self.epsilon)) / (
            self.alpha + self.beta
        )

    def forward(self, x, is_source=None):
        x = x.float()
        x_raw = x
        x_abs = torch.abs(x)

        # Channel threshold
        channel_threshold = self.compute_threshold(x_abs, is_source)
        # Apply adaptive channel filter
        up_sub = self.apply_filter(x_abs, channel_threshold)

        return torch.mul(torch.sign(x_raw), up_sub)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()

        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 3x3 convolution
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

        # 5x5 convolution
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=in_channels)

        # 7x7 convolution
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)

        # pointwise convolution
        self.pw_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

        # batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_source=None):
        # 1x1 convolution
        x1 = self.conv1x1(x)
        x1 = F.relu(x1)

        # 3x3 convolution
        x3 = self.conv3x3(x)
        x3 = F.relu(x3)

        # 5x5 convolution
        x5 = self.conv5x5(x)
        x5 = F.relu(x5)

        # 7x7 convolution
        x7 = self.conv7x7(x)
        x7 = F.relu(x7)

        # concatenate feature maps
        x_cat = torch.cat((x1, x3, x5, x7), dim=1)

        # pointwise convolution
        out = self.pw_conv(x_cat)
        # batch normalization
        if is_source is None :
            out = self.bn(out)
        else:
            out = self.bn(out,is_source)

        # relu activation
        out = F.relu(out)

        return out


class SCR(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SCR, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "scr"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x,is_source=None):
        if is_source is None:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x,is_source)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # (B, 1, H, W)
        return torch.cat([max_pool, avg_pool], dim=1)     # (B, 2, H, W)


class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, padding=(kernel_size-1)//2, relu=False, bn=True)
    def forward(self, x, is_source=None):
        if is_source is None:
            x_cp = self.compress(x)
            x_out = self.conv(x_cp)
            scale = torch.sigmoid(x_out)
            # 注意力权重已包含 BN
            return x * scale
        else:
            x_cp = self.compress(x)
            x_out = self.conv(x_cp,is_source)
            scale = torch.sigmoid(x_out)
            # 注意力权重已包含 BN
            return x * scale
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x, is_source=None):
        # branch channel-width
        x1 = x.permute(0,2,1,3).contiguous()
        y1 = self.cw(x1, is_source).permute(0,2,1,3).contiguous()
        # branch height-channel
        x2 = x.permute(0,3,2,1).contiguous()
        y2 = self.hc(x2, is_source).permute(0,3,2,1).contiguous()
        # optional spatial-only
        if not self.no_spatial:
            y3 = self.hw(x, is_source)
            return (y1 + y2 + y3) / 3
        else:
            return (y1 + y2) / 2




class SIA(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.trip_spatial = TripletAttention(no_spatial=False)
        self.trip_channel = TripletAttention(no_spatial=True)
        self.spatial_msc = MultiScaleConv(num_features, num_features)
        self.channel_msc = MultiScaleConv(num_features, num_features)

    def forward(self, x, is_source=None):
        residual = x

        # Spatial branch
        x_spatial = self.trip_spatial(x, is_source)
        x_spatial = self.spatial_msc(x_spatial, is_source)

        # Channel branch
        x_channel = self.trip_channel(x, is_source)
        x_channel = self.channel_msc(x_channel, is_source)

        # Output
        out = WEIGHT*x_spatial +WEIGHT*x_channel + residual
        return out

        


class SDH(nn.Module):
    def __init__(
            self,
            in_channels=256,
            num_features=256,
    ):
        super().__init__()

        self.sia = SIA(num_features)
        self.scr = SCR()

    def forward(self, x,is_source=None):
        if is_source is None:
            residual = x
            embedding = self.sia(x)#256 256 14 14->256 256 14 14
            embedding_final =WEIGHT*embedding +WEIGHT* self.scr(x) + residual
            return embedding_final
        residual = x
        embedding = self.sia(x,is_source)
        embedding_final = WEIGHT*embedding +WEIGHT*self.scr(x) + residual
        return embedding_final

       



class Backbone(nn.Module):
    def __init__(self, resnet, use_filter):
        super().__init__()

        # resnet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        # Add filter after first block
        if use_filter:
            self.layer1 = add_module_after_block(resnet.layer1, 1, AdaptiveFilter(256, gap_size=(1, 1)))
            self.layer2 = add_module_after_block(resnet.layer2, 1, AdaptiveFilter(512, gap_size=(1, 1)))
            self.layer3 = add_module_after_block(resnet.layer3, 1, AdaptiveFilter(1024, gap_size=(1, 1)))
        else:
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3

        self.out_channels = 1024

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if torch.isnan(x).int().sum() > 0:
            print(torch.isnan(x).int().sum())
        return x

    def forward(self, x):
        feat = self._forward(x)
        return OrderedDict([["feat_res4", feat]])

class BoxHead(nn.Module):
    def __init__(self, resnet):
        super(BoxHead, self).__init__()
        self.layer4 = deepcopy(resnet.layer4)
        self.out_channels = [1024, 2048]

        self.SDH_model = SDH(256)

        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1)
    def forward(self, x,is_source=None):
        qconv1 = self.qconv1(x)#256 1024 14 14 --->256 256 14 14
        x_sc_mlp_feat = self.SDH_model(qconv1,is_source)
        qconv2 = self.qconv2(x_sc_mlp_feat)

        layer5_feat = self.layer4(qconv2)

        x_feat = F.adaptive_max_pool2d(qconv2, 1)

        feat = F.adaptive_max_pool2d(layer5_feat, 1)

        return OrderedDict([["feat_res4", x_feat], ["feat_res5", feat]])

class Res5Head(nn.Module):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()
        self.layer4 = resnet.layer4
        self.out_channels = [1024, 2048]

        self.SDH_model = SDH(256)

        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1)

    def bottleneck_forward(self, bottleneck, x, is_source):
        identity = x

        out = bottleneck.conv1(x)
        out = bottleneck.bn1(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv2(out)
        out = bottleneck.bn2(out, is_source)
        out = bottleneck.relu(out)
        out = bottleneck.conv3(out)
        out = bottleneck.bn3(out, is_source)
        if bottleneck.downsample is not None:
            for module in bottleneck.downsample:
                if not isinstance(module, DSBN2d):
                    identity = module(x)
                else:
                    identity = module(identity, is_source)
        out += identity
        out = bottleneck.relu(out)
        return out

    def forward(self, x,is_source=True):
        qconv1 = self.qconv1(x)
        x_sc_mlp_feat = self.SDH_model(qconv1,is_source)
        qconv2 = self.qconv2(x_sc_mlp_feat)

        # layer5_feat = self.layer4(qconv2)
        layer5_feat = qconv2
        for module in self.layer4:
            layer5_feat = self.bottleneck_forward(module,layer5_feat, is_source)
        x_feat = F.adaptive_max_pool2d(qconv2, 1)

        feat = F.adaptive_max_pool2d(layer5_feat, 1)

        return OrderedDict([["feat_res4", x_feat], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet,use_filter=True), BoxHead(resnet),Res5Head(resnet)

# def build_resnet(name="resnet50", pretrained=True):
#     from torchvision.models import resnet
#     resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
#     resnet_model = resnet.resnet50(pretrained=True)
#
#     # freeze layers
#     resnet_model.conv1.weight.requires_grad_(False)
#     resnet_model.bn1.weight.requires_grad_(False)
#      resnet_model.bn1.bias.requires_grad_(False)
#
#     return Backbone(resnet_model), Res5Head(
#         resnet_model)
