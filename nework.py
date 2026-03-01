# 第三阶段改为RLFN
import torch.nn as nn
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
# RLFN
from rlfn_ntire import RLFN_Prune


class TxSNetsmallnew(nn.Module): # JDD network architecture
    '''JDD network architecture'''
    def __init__(self):
        super(TxSNetsmallnew, self).__init__()
        self.scale = 1
        self.outC = 16
        self.msfa_size = 2
        # WB_Conv 这种配置通常用于模型中的特征分离，尤其是在多通道图像处理或深度可分离卷积操作中，用于提高计算效率
        self.WB_Conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False, groups=4)

        self.input_channels = 4
        self.base_channels = 64
        self.num_residuals = 3
        
        # The first stage: denoising
        modules_dn_head = [default_conv(self.input_channels, self.base_channels, kernel_size=3), nn.ReLU(inplace=True)]
        modules_dn_body = [RRDB46(nc=self.base_channels, gc=64) for _ in range(1)]
        modules_dn_tail = [default_conv(self.base_channels, 4, kernel_size=3)]
        # The second stage
        modules_head_1 = [default_conv(5, self.base_channels, kernel_size=3), nn.ReLU(inplace=True)]
        modules_body_1 = [RRDB(nc=self.base_channels, gc=64) for _ in range(1)]
        modules_tail_1 = [default_conv(self.base_channels, 4, kernel_size=3)]
        # The third stage
        # modules_head_2 = [default_conv(5, self.base_channels, kernel_size=3), nn.ReLU(inplace=True)]
        # modules_body_2 = [RCAB(default_conv, self.base_channels, kernel_size=3, reduction=16) for _ in range(10)]
        # modules_tail_2 = [default_conv(self.base_channels, 4, kernel_size=3)]

        self.dn_head = nn.Sequential(*modules_dn_head)
        self.dn_body = nn.Sequential(*modules_dn_body)
        self.dn_tail = nn.Sequential(*modules_dn_tail)

        self.head_1 = nn.Sequential(*modules_head_1)
        self.body_1 = nn.Sequential(*modules_body_1)
        self.tail_1 = nn.Sequential(*modules_tail_1)

        # self.head_2 = nn.Sequential(*modules_head_2)
        # self.body_2 = nn.Sequential(*modules_body_2)
        self.body_2 = RLFN_Prune(in_channels=5, out_channels=4, feature_channels=48, mid_channels=48, upscale=1)
        # self.tail_2 = nn.Sequential(*modules_tail_2)
        # ?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.groups == self.msfa_size**2:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter(h, self.msfa_size)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, data):

        x, y = data # noisy DoT_LR, noisy DoFP
        # -------------The first stage: denoising-----------------------
        y = self.space_to_depth(y, 2)
        y = self.dn_head(y)
        y = self.dn_body(y)
        y = self.dn_tail(y) # 4通道
        y = self.depth_to_space(y, 2) # 1通道
        dn = y # 得到去噪的DoFP图像
       
        # -------------The second stage-----------------------------------------
        x = self.masaick_input(y) # DoFP->DoT(单通道->四通道)  所以传进来的noisy DoT_LR根本没用上！又重新算了一遍
        WB_norelu = self.WB_Conv(x) # 分4组卷积
        edge1 = self.edge_guidance(WB_norelu) # 边缘引导图
        stage_one_input = torch.cat((WB_norelu, edge1), dim=1) # 加入边缘引导图
        
        y = self.head_1(stage_one_input)
        y = self.body_1(y)
        HR_4x_1 = self.tail_1(y) 
        first_stage = torch.add(HR_4x_1, WB_norelu) # 第二阶段前后图像相加 得到第二阶段结果
        
        # -------------The third stage------------------------------------------
        edge2 = self.edge_guidance(first_stage) # 边缘引导图
        stage_two_input = torch.cat((first_stage, edge2), dim=1) # 加入边缘引导图

        # y = self.head_2(stage_two_input)
        # y = self.body_2(y)
        HR_4x_2 = self.body_2(stage_two_input)
        # HR_4x_2 = self.tail_2(y) 
        second_stage = torch.add(first_stage, HR_4x_2) # 得到第三阶段结果
        
        # 返回：第一阶段去噪结果、第三阶段重建结果、第二阶段重建结果、第二阶段处理的图像
        return dn, second_stage, first_stage, WB_norelu

    def masaick_input(self, y, msfa_size=2): # DoFP->DoT(单通道->四通道)
        '''
        对输入张量 y 进行处理 DoFP->DoT(单通道->四通道)
        msfa_size：马赛克模式的大小，这里默认值为 2，表示使用 2x2 的模式对图像进行马赛克分割。
        '''
        N, C, H, W = y.size()
        mask = torch.zeros((N, C*4, H, W), dtype=torch.float32).cuda() # DoT图像的尺寸
        input_image = torch.clone(mask).cuda()
        
        # 采样的位置设1，不采样的设0，DoFP与DoT每个通道相乘即可得到DoT
        for k in range(0, N):
            for i in range(0, msfa_size):
                for j in range(0, msfa_size):
                    mask[k, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1

        input_image[:, 0, :, :] = mask[:, 0, :, :] * y[:, 0, :, :]
        input_image[:, 1, :, :] = mask[:, 1, :, :] * y[:, 0, :, :]
        input_image[:, 2, :, :] = mask[:, 2, :, :] * y[:, 0, :, :]
        input_image[:, 3, :, :] = mask[:, 3, :, :] * y[:, 0, :, :]

        return input_image

    def edge_guidance(self, y): # 生成边缘引导图
        ''''''
        s = torch.sum(y, dim=1) / 2.
        s = s.unsqueeze(1)
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()
        grad_x = F.conv2d(s[:, 0:1, :, :], kernel_x, padding=1)
        grad_y = F.conv2d(s[:, 0:1, :, :], kernel_y, padding=1)
        g = torch.abs(grad_x) + torch.abs(grad_y)
        # tt0 = g[:, 0, :, :].detach().cpu().numpy()
        zero = torch.zeros_like(g)
        one = torch.ones_like(g)
        g = torch.where(g > 0.8, one, zero)
        # tt1 = g[:, 0, :, :].detach().cpu().numpy()
        return g

    def space_to_depth(self, x, block_size):
        '''将空间信息压缩到深度维度来改变张量结构,通道数变成 c * block_size ** 2'''
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

    def depth_to_space(self, x, block_size):
        '''将深度（通道）信息重新分布到空间维度'''
        return torch.nn.functional.pixel_shuffle(x, block_size)

#================================================================================


class CALayer(nn.Module): # RCAB的一部分
    '''Channel Attention (CA) Layer'''
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module): # 第三阶段用的RCAB 10次
    '''Residual Channel Attention Block (RCAB)'''
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class RCAB_residual(nn.Module):
    def __init__(
        self, conv, channels, k_size, reduc):

        super(RCAB_residual, self).__init__()
        modules_body1 = [RCAB(conv, channels, kernel_size=k_size, reduction=reduc)]
        modules_body2 = [RCAB(conv, channels, kernel_size=k_size, reduction=reduc)]
        modules_body3 = [RCAB(conv, channels, kernel_size=k_size, reduction=reduc)]
        self.body1 = nn.Sequential(*modules_body1)
        self.body2 = nn.Sequential(*modules_body2)
        self.body3 = nn.Sequential(*modules_body3)

    def forward(self, x):
        res1 = self.body1(x)
        res1 += x
        res2 = self.body2(res1)
        res2 += res1
        res3 = self.body3(res2)
        res3 += res2
        #res = self.body(x).mul(self.res_scale)
        res3 += x
        return res3

#================================================================================

def get_WB_filter(size, msfa_size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(msfa_size**2)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (msfa_size**2))
    filter0 = np.reshape(BilinearFilter, (size, size))
    filter1 = torch.from_numpy(filter0).float()
    return torch.from_numpy(filter0).float()

def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)

#====================================================================================
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer

def norm_layer(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channelss, out_channels, kernel_size=3, stride=1, bias=False,
            norm_type=False, act_type='relu'):

        m = [default_conv(in_channelss, out_channels, kernel_size, stride=stride, bias=bias)]
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, out_channels) if norm_type else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock, self).__init__(*m)

class ResidualDenseBlock5(nn.Module):
    """
    原Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(ResidualDenseBlock5, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x
class RRDB(nn.Module): # 第一阶段用的RRDB
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB2 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB3 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x

class FourierUnit(nn.Module):
    '''频域特征融合'''
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2, 
            out_channels=out_channels * 2,
            kernel_size=1, 
            stride=1, 
            padding=0, 
            groups=self.groups, 
            bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        batch, c, h, w = x.size()
        
        ffted = torch.fft.rfft2(x, norm='ortho') # 计算输入图像的二维快速傅里叶变换（2D FFT）。rfft2 是“快速傅里叶变换的实数版本”，适用于实数输入图像数据，它返回一个包含实部和虚部的复数频域表示。norm='ortho'是正交规范化选项，确保傅里叶变换具有能量守恒的性质
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1) # 提取复数结果的实部
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1) # 提取复数结果的虚部
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1) # 给实部和虚部增加一个新的维度，使其形状符合后续卷积操作的要求。将实部和虚部拼接在一起，形成一个大小为 (batch, c, h, w/2+1, 2) 的张量，其中 2 代表实部和虚部
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous() # 调整维度顺序 (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:]) # 将张量展平为合适的形状，卷积层的输入形状为 (batch, c*2, h, w/2+1)
        
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        # (batch, c*2, h, w/2+1) --> (batch, c, 2, h, w/2+1) --> (batch, c, h, w/2+1, 2) 
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous() 
        ffted = torch.view_as_complex(ffted) # 将 ffted 张量从实部和虚部的表示转换为复数张量
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho') # 对频域数据进行逆傅里叶变换（Inverse 2D FFT），将频域信号转换回时域信号
        return output



class ResidualDenseBlock46(nn.Module):
    """
    Residual Dense Block修改2
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(ResidualDenseBlock46, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.pw = nn.Conv2d(nc+3*gc, nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.fft = FourierUnit(nc, nc) 
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x3 = torch.cat((x, x1, x2, x3), 1)
        x3 = self.pw(x3.mul(self.res_scale))
        x4 = self.fft(x3)
        return x4+x3+x
        


class RRDB46(nn.Module): # 第二阶段用的RRDB
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(RRDB46, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock46(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB2 = ResidualDenseBlock46(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB3 = ResidualDenseBlock46(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x


    

