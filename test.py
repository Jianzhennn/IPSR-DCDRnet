# test only 加载模型 测试
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from network import TxSNetsmallnew
from tqdm import tqdm
import pandas as pd
from dataset_msimaker.data import get_test_set_realnoise, get_noisytest_set_realnoise
from torch.utils.data import DataLoader
from common import calculate_psnr, calculate_aolp_psnr, calculate_stokes, calculate_stokes_tensor, mask_input_tensor, l1_aolp, ssim_2D
import numpy as np
from PIL import Image
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--nEpochs", type=int, default=1, help="number of epochs to train for")

def main():

    # cudnn.benchmark = True
    opt = parser.parse_args()
    print(opt)

    print("===> Loading test_set")
 
    test_set = get_test_set_realnoise()
    testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    
    print("===> Building model")
    model = TxSNetsmallnew()


    print("===> Loading test_model")
    test_model = "checkpoint/Best_demosaick_TxSnew_epoch4017.pth"
    
    if os.path.isfile(test_model):
        print("=> loading model '{}'".format(test_model))
        weights = torch.load(test_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(test_model))

    print("===> Setting GPU")
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    print("===> Testing")
    results = {'s0_psnr': [],'s0_ssim': [],'dolp_psnr': [], 'dolp_ssim': [], 'aolp_psnr': [],  'aolp_ssim': [],'model':[]}
          
    test_results = test(testing_data_loader, model, 1, opt)


    results['model'] = test_model
    results['s0_psnr'] = test_results['s0_psnr']/20
    results['dolp_psnr'] = test_results['dolp_psnr']/20
    results['aolp_psnr'] = test_results['aolp_psnr']/20
    results['s0_ssim'] = test_results['s0_ssim']/20
    results['dolp_ssim'] = test_results['dolp_ssim']/20
    results['aolp_ssim'] = test_results['aolp_ssim']/20

    


    # 保存定量结果
    if results:

        statistics_folder = "result/"
        if not os.path.exists(statistics_folder):
            os.makedirs(statistics_folder)
        data_frame = pd.DataFrame(data=[results], index=[1])
        data_frame.to_csv('result/test_results.csv', mode='a', header=False, index_label='Epoch')
    else:
        print("Error: 'results' is empty. No data to save.")

 

def test(testing_data_loader, model, epoch, opt):

    test_bar = tqdm(testing_data_loader)
    test_results = {'batch_sizes': 0, 's0_psnr': 0, 's0_mse': 0, 'dolp_psnr': 0, 'dolp_mse': 0, 'aolp_psnr': 0, 'aolp_mse': 0,
                    'i0_psnr': 0, 'i0_mse': 0, 'i45_psnr': 0, 'i45_mse': 0, 'i90_psnr': 0, 'i90_mse': 0, 'i135_psnr': 0, 'i135_mse': 0,
                    's0_ssim': 0, 'dolp_ssim': 0, 'aolp_ssim': 0,
                    'S0': [], 'DOLP': [], 'AOLP': [],
                    's0_psnr_list': []}
    model.eval()

    with torch.no_grad():
        for batch in test_bar:
            input_raw, input, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(
            batch[2], requires_grad=False)

            N, C, H, W = batch[0].size()

            test_results['batch_sizes'] += N

            if torch.cuda.is_available():
                input = input.cuda()
                input_raw = input_raw.cuda()
                label_x4 = label_x4.cuda()


            dn, HR_4x, HR_4x_stage_1, BI = model([input, input_raw])

            i0_output = HR_4x[:, 0, :, :].detach().cpu().numpy() 
            i45_output = HR_4x[:, 1, :, :].detach().cpu().numpy()
            i90_output = HR_4x[:, 3, :, :].detach().cpu().numpy()
            i135_output = HR_4x[:, 2, :, :].detach().cpu().numpy()

            i0_gt = label_x4[:, 0, :, :].detach().cpu().numpy()
            i45_gt = label_x4[:, 1, :, :].detach().cpu().numpy()
            i90_gt = label_x4[:, 3, :, :].detach().cpu().numpy()
            i135_gt = label_x4[:, 2, :, :].detach().cpu().numpy()

            stokes_output = calculate_stokes(i0_output, i45_output, i90_output, i135_output) # stokes_output['S0'].shape:(1,480,640)
            stokes_gt = calculate_stokes(i0_gt, i45_gt, i90_gt, i135_gt)

            i0_mse, i0_psnr = calculate_psnr(i0_output, i0_gt)
            i45_mse, i45_psnr = calculate_psnr(i45_output, i45_gt)
            i90_mse, i90_psnr = calculate_psnr(i90_output, i90_gt)
            i135_mse, i135_psnr = calculate_psnr(i135_output, i135_gt)

            s0_mse, s0_psnr = calculate_psnr(stokes_output['S0'], stokes_gt['S0'])
            dolp_mse, dolp_psnr = calculate_psnr(stokes_output['DOLP'], stokes_gt['DOLP'])
            aolp_mse, aolp_psnr = calculate_aolp_psnr(stokes_output['AOLP'], stokes_gt['AOLP'])
            s0_ssim = ssim_2D(stokes_gt['S0'], stokes_output['S0'])
            dolp_ssim = ssim_2D(stokes_gt['DOLP'], stokes_output['DOLP'])
            aolp_ssim = ssim_2D(stokes_gt['AOLP'], stokes_output['AOLP'])

            # 最终是所有批次的加和
            test_results['i0_mse'] += i0_mse * N
            test_results['i0_psnr'] += i0_psnr * N
            test_results['i45_mse'] += i45_mse * N
            test_results['i45_psnr'] += i45_psnr * N
            test_results['i90_mse'] += i90_mse * N
            test_results['i90_psnr'] += i90_psnr * N
            test_results['i135_mse'] += i135_mse * N
            test_results['i135_psnr'] += i135_psnr * N

            test_results['s0_mse'] += s0_mse * N
            test_results['s0_psnr'] += s0_psnr * N           
            test_results['dolp_mse'] += dolp_mse * N
            test_results['dolp_psnr'] += dolp_psnr * N            
            test_results['aolp_mse'] += aolp_mse * N
            test_results['aolp_psnr'] += aolp_psnr * N
            test_results['s0_ssim'] += s0_ssim * N
            test_results['dolp_ssim'] += dolp_ssim * N
            test_results['aolp_ssim'] += aolp_ssim * N
            # 保存每个批次的 s0_psnr
            test_results['s0_psnr_list'].append(s0_psnr)

            # 保存预测出的图像
            test_results['S0'].append(stokes_output['S0'])
            test_results['DOLP'].append(stokes_output['DOLP'])
            test_results['AOLP'].append(stokes_output['AOLP'])


            test_bar.set_description(desc='[%d/%d] i0_psnr: %.4f, i45_psnr: %.4f, i90_psnr: %.4f, i135_psnr: %.4f,'
                                          's0_psnr: %.4f, dolp_psnr: %.4f, aolp_psnr: %.4f '
                                          's0_ssim: %.4f, dolp_ssim: %.4f, aolp_ssim: %.4f ' % (
                epoch, opt.nEpochs, 
                test_results['i0_psnr'] / test_results['batch_sizes'], 
                test_results['i45_psnr'] / test_results['batch_sizes'],
                test_results['i90_psnr'] / test_results['batch_sizes'],
                test_results['i135_psnr'] / test_results['batch_sizes'],

                test_results['s0_psnr'] / test_results['batch_sizes'],
                test_results['dolp_psnr'] / test_results['batch_sizes'],
                test_results['aolp_psnr'] / test_results['batch_sizes'],

                test_results['s0_ssim'] / test_results['batch_sizes'],
                test_results['dolp_ssim'] / test_results['batch_sizes'],
                test_results['aolp_ssim'] / test_results['batch_sizes']))
            
    return test_results

def save_result_s0(images, file_path, img_type):
    # 确保输出目录存在
    os.makedirs(file_path, exist_ok=True)
    for i in range(0, len(images)):
        img = []
        img = (255 * (images[i] - images[i].min()) / (images[i].max() - images[i].min())).astype('uint8') 


        if img.ndim == 2:
            pass  # 灰度图像 (H, W)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)  
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0)) 

        file_name = "{}/{}_output_{}.png".format(file_path, i+1, img_type)
        
        success = cv2.imwrite(file_name, img)
        if success:
            print(f"Saved as PNG to {file_name}")
        else:
            print(f"Failed to save image to {file_name}")
            print(f"Image dtype: {img.dtype}, shape: {img.shape}")
    

# 将 DoLP 数据转换为伪彩色图像并保存为 PNG 格式
def save_result_DoLP(images, output_path, img_type):
  
    dolp_data = images[7][0]

    plt.imshow(dolp_data, cmap='Blues')
    plt.savefig('DoLP_visualization.png', dpi=300)
    

# 将 AoLP 数据转换为伪彩色图像并保存为 PNG 格式 
def save_result_AoLP(images, output_path, img_type):
    for i in range(0, len(images)):
        aop_data = []
        aop_data = images[i]
        
        hue = (aop_data[0] * 179).astype(np.uint8)  

        saturation = np.full_like(hue, 255)  
        value = np.full_like(hue, 255)       

        hsv_image = cv2.merge([hue, saturation, value])
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        file_name = "{}/{}_output_{}.png".format(output_path, i+1, img_type)
        success = cv2.imwrite(file_name, rgb_image)
        if success:
            print(f"Saved image as PNG to {file_name}")
        else:
            print(f"Failed to save image to {file_name}")

if __name__ == "__main__":
    for i in range(0,100):# 循环测试100次
        main()