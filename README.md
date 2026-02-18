# IPSR-DCDRnet
这是一个用于红外偏振联合去噪去马赛克的深度学习方法。

数据集：IR-DoT数据集由83个不同的户外场景组成，每组包含四个不同方向的偏振图像，每个图像的高分辨率为640×480像素。IR-DoT用于生成合成的DoFP并作为真实值。IRDoFP数据集由200个不同的户外场景组成，是需要去马赛克的640×512像素单通道图像。

52_Best_demosaick_TxSnew_epoch4107.pth 是best model

本方法基于https://ieeexplore.ieee.org/document/10304085，是本文的一个对比方法。
