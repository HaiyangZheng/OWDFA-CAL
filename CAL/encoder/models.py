import warnings

import numpy as np
import timm
import wandb
from loguru import logger
from torch_dct import dct_2d, idct_2d

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BinaryClassifier']

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)

warnings.filterwarnings("ignore")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    if isinstance(m, torch.nn.Parameter) or isinstance(m, torch.Tensor):
        nn.init.normal_(m, std=0.001)

# Custom implementation of Adaptive Average Pooling
class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(
            np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - \
            (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size),
                           stride=list(stride_size))
        x = avg(x)
        return x

class CAL_Classifier(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='no',
                 pretrained=False,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck

        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)

        self.fft_attn_module = DCTAttention(in_channels=3)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        self.pool = AdaptiveAvgPool2dCustom((1, 1))

        self.dropout = nn.Dropout(drop_rate)

        if self.neck == 'bnneck':
            logger.info('Using BNNeck')
            self.bottleneck = nn.BatchNorm1d(self.num_features)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

            self.fc2 = nn.Parameter(torch.empty(num_classes, self.num_features))
            self.valid_proto_num = num_classes
            weights_init_classifier(self.fc2)

        else:
            self.fc2 = nn.Linear(self.num_features, self.num_classes)

    def forward_featuremaps(self, x):
        featuremap = self.encoder.forward_features(x)
        return featuremap

    def forward_features(self, x):
        featuremap = self.encoder.forward_features(x)  # [B, C, H, W]

        fft_attn = self.fft_attn_module(x)  # [B, 1, H, W]
        if fft_attn.shape[-2:] != featuremap.shape[-2:]:
            fft_attn = F.interpolate(fft_attn, size=featuremap.shape[-2:], mode='bilinear', align_corners=False)
        featuremap = featuremap * fft_attn  # 加权融合

        feature = self.pool(featuremap).flatten(1)

        if self.neck == 'bnneck':
            feature = self.bottleneck(feature)

        return feature

    def forward(self, x):
        feature = self.forward_features(x)
        x = self.dropout(feature)

        method = x @ self.fc2[:self.valid_proto_num].t()

        if self.is_feat:
            return method, feature
        return method

class DCTAttention(nn.Module):
    def __init__(self, in_channels, target_size=(8,8)):
        """
        Frequency-Guided Feature Enhancement (FFE) based on DCT
        """
        super().__init__()
        # Lightweight convolutional network operating in the frequency domain
        self.freq_attn_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.pool = AdaptiveAvgPool2dCustom(target_size)

    def forward(self, x_input):
        # Transform input features into frequency domain
        dct_feat = dct_2d(x_input)
        freq_attn = self.freq_attn_net(dct_feat)
        dct_mod = dct_feat * freq_attn
        attn_spatial = idct_2d(dct_mod)
        attn = torch.sigmoid(attn_spatial.mean(dim=1, keepdim=True))  # [B, 1, H, W]

        attn = self.pool(attn)
        return attn