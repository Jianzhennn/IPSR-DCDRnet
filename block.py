# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

class SAB(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(SAB, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        # self.conv_f = conv(f, f, kernel_size=1)
        # self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        # self.conv3 = conv(f, f, kernel_size=3, padding=1)
        
        self.dw1 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0, groups=f)
        self.dw2 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=0, groups=f)
        self.pw = nn.Conv2d(f, f, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.dw1(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.dw2(v_max)
        c3 = self.pw(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        # cf = self.conv_f(c1_)
        # c4 = self.conv4(c3 + cf)
        c4 = self.conv4(c3)
        m = self.sigmoid(c4)
        return x * m

# CAB
class ChannelAggregationFFN(nn.Module): 
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels, # embed_dims的4倍效果较好
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d( # 对通道进行升维处理，不涉及空间位置的变化
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.GELU() # GELU
        self.fc2 = nn.Conv2d(   # 用于将通道数从 feedforward_channels 降回 embed_dims，完成特征图的降维
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d( # 用于将 feedforward_channels 通道数压缩到 1。该操作是通道聚合CA的一部分
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale( # 乘 比率因子
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.GELU() # GELU
    
    # CA模块实现
    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        input_x = x
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + input_x
        
class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale
    
    
class convBlock(nn.Module):
    """
    
    """
    def __init__(self, in_channels, out_channels):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.dw = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(inplace=True)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.dw(x))
        output = self.lrelu(x + x1 + x2)
        return output
        

class RLFB(nn.Module):
    def __init__(self, in_channels):
        """
        实验42 RLFB 模块实现：4个DW+关于原点对称激活函数
        - in_channels: 输入通道数（保持输入输出通道数一致）
        """
        super(RLFB, self).__init__()
        
        # Pointwise 卷积 (1x1 卷积) 混合通道
        self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        # 深度可分离卷积 + ReLU 激活
        self.dw3_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels // 4 + (1 if i < in_channels % 4 else 0),  # 动态处理通道
                    in_channels // 4 + (1 if i < in_channels % 4 else 0),
                    kernel_size=3, stride=1, padding=1, groups=in_channels // 4 + (1 if i < in_channels % 4 else 0)
                ),
                nn.ReLU()
            ) for i in range(4)
        ])
        
        # 1x1 卷积用于通道融合，恢复到 in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Pointwise 卷积（通道混合）
        out = self.pw(x)
        
        # 动态分割通道：处理通道数不能被4整除的情况
        split_sizes = [out.shape[1] // 4 + (1 if i < out.shape[1] % 4 else 0) for i in range(4)]
        splits = torch.split(out, split_sizes, dim=1)  # 使用 torch.split 分割
        
        # 每个通道块通过 DW-3 卷积块
        outputs = [self.dw3_blocks[i](splits[i]) for i in range(4)]
        
        # 拼接所有输出
        concat_out = torch.cat(outputs, dim=1)
        
        # 1x1 卷积融合通道，恢复到 in_channels
        out = self.conv1(concat_out)
        
        # 关于原点对称的激活函数
        alpha_out = torch.sigmoid(out) - 0.5
        
        # 残差连接：输入与输出相加 相乘
        out = (out + x) * alpha_out
        return out
        
# class RLFB(nn.Module):
#     def __init__(self, in_channels):
#         """
#         实验44 RLFB 模块实现：
#         - in_channels: 输入通道数（保持输入输出通道数一致）
#         """
#         super(RLFB, self).__init__()
        
#         # 深度可分离卷积 + ReLU 激活
#         self.convBlock = convBlock(in_channels, in_channels)
        
#         self.SAB = SAB(esa_channels=96, n_feats=in_channels, conv=nn.Conv2d)
#         self.CAB = ChannelAggregationFFN(in_channels, in_channels*4)

#     def forward(self, x):
#         input_x = x
#         x = self.convBlock(input_x)
#         x = self.SAB(x) + input_x
        
#         out = self.convBlock(x)
#         out = self.CAB(out) + x
#         return out

# # ------------------------------------------------------------RLFB修改12---------------------------------------------------------
# class RLFB43(nn.Module):
#     def __init__(self, in_channels):
#         """
#         实验43 RLFB 模块实现：
#         - in_channels: 输入通道数（保持输入输出通道数一致）
#         """
#         super(RLFB43, self).__init__()
        
#         # Pointwise 卷积 (1x1 卷积) 混合通道
#         # self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
#         # 深度可分离卷积 + ReLU 激活
#         self.convBlock = nn.ModuleList([
#             nn.Sequential(
#                 convBlock(
#                     in_channels // 4 + (1 if i < in_channels % 4 else 0),  # 动态处理通道
#                     in_channels // 4 + (1 if i < in_channels % 4 else 0),
#                     ),
#             ) for i in range(4)
#         ])
        
#         # 1x1 卷积用于通道融合，恢复到 in_channels
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.SAB = SAB(esa_channels=96, n_feats=in_channels, conv=nn.Conv2d)

#     def forward(self, x):
#         # Pointwise 卷积（通道混合）
#         # out = self.pw(x)
#         out = x
        
#         # 动态分割通道：处理通道数不能被4整除的情况
#         split_sizes = [out.shape[1] // 4 + (1 if i < out.shape[1] % 4 else 0) for i in range(4)]
#         splits = torch.split(out, split_sizes, dim=1)  # 使用 torch.split 分割
        
#         # 每个通道块通过 卷积块
#         outputs = [self.convBlock[i](splits[i]) for i in range(4)]
        
#         # 拼接所有输出
#         concat_out = torch.cat(outputs, dim=1)
        
#         # 1x1 卷积融合通道，恢复到 in_channels
#         out = self.conv1(concat_out)
        
#         out = self.SAB(out)
        
#         # 关于原点对称的激活函数
#         alpha_out = torch.sigmoid(out) - 0.5
        
#         # 残差连接：输入与输出相加 相乘
#         out = (out + x) * alpha_out
        
#         return out
# # ------------------------------------------------------------RLFB修改12---------------------------------------------------------