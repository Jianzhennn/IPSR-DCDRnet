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
from dataset_msimaker.data import get_training_set_realnoise, get_test_set_realnoise
from torch.utils.data import DataLoader
from common import calculate_psnr, calculate_aolp_psnr, calculate_stokes, calculate_stokes_tensor, mask_input_tensor, l1_aolp, ssim_2D
import numpy as np
from PIL import Image
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--trainBatchSize", type=int, default=16, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=5000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=2000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--drop_rate", type=float, default=0.1, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--common_aug", action="store_false", help="Use common augmentation")
parser.add_argument("--mosaic_aug", action="store_false", help="Use mosaic augmentation")
parser.add_argument("--use_bi", action="store_false", help="Use mosaic augmentation")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--msfa_size', '-uf',  type=int, default=2, help="the size of square msfa")
parser.add_argument('--test_epoch',  type=int, default=1, help="the epoch of test") # 多少个epoch测试一次 test data
# Loss权重
parser.add_argument('--dn_loss_lambda', default=2.5,  type=float, help="the lambda of stokes loss")# L1=2.5
parser.add_argument('--stage1_loss_lambda', default=0.5,  type=float, help="the lambda of stokes loss")# L2=0.5
parser.add_argument('--img_loss_lambda', default=1.5,  type=float, help="the lambda of stokes loss") # L3w1=1.5
parser.add_argument('--dolp_loss_lambda', default=1.64,  type=float, help="the lambda of stokes loss")# L3w2=1.64
parser.add_argument('--aolp_loss_lambda', default=0.2,  type=float, help="the lambda of stokes loss")# L3w3=0.2
parser.add_argument('--polar_loss_lambda', default=0.5,  type=float, help="the lambda of stokes loss")# L3w4=0.5



def main() -> object:
    global opt, model, PATTERN, PAD

    PAD = 16
    PATTERN = np.array([[1, 2, 3, 4], [1, 3, 2, 4],
                        [2, 1, 4, 3], [2, 4, 1, 3],
                        [3, 1, 4, 2], [3, 4, 1, 2],
                        [4, 2, 3, 1], [4, 3, 2, 1]])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    opt = parser.parse_args()
    print(opt)
    '''例子：
    Namespace(trainBatchSize=16, testBatchSize=1, nEpochs=5000, lr=0.0003, step=2000, drop_rate=0.1, 
    cuda=False, common_aug=True, mosaic_aug=True, use_bi=True, resume='checkpoint/Demosaick_epoch_pre_TxSnew_750.pth', 
    start_epoch=750, threads=0, momentum=0.9, weight_decay=0.0001, pretrained='', msfa_size=2, test_epoch=20, img_loss_lambda=1.5, dn_loss_lambda=2.5, dolp_loss_lambda=1.64, aolp_loss_lambda=0.2, polar_loss_lambda=0.5, stage1_loss_lambda=0.5)'''

    opt.common_aug = True
    opt.mosaic_aug = False
    cuda = True
    opt.cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # opt.seed = random.randint(123)
    opt.seed = 123
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # opt.norm_flag = False
    # opt.augment_flag = False
    # opt.add_blur = False
    train_set = get_training_set_realnoise(opt)
    test_set = get_test_set_realnoise()
    # noisytest_set = get_noisytest_set_realnoise()
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.trainBatchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)

    print("===> Building model")
    model = TxSNetsmallnew()

    criterion1 = nn.L1Loss()
    criterion_aop = l1_aolp()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion_aop = criterion_aop.cuda()
        criterion1 = criterion1.cuda()
    else:
        model = model.cpu()
    # optionally resume from a checkpoint

    num_threads = 1
    torch.set_num_threads(num_threads)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    save_opt(opt)
    print("===> Training")
    best_psnr = 0
    best_s0psnr = 0
    best_dolppsnr = 0
    best_aoppsnr = 0
    results = {'im_loss': [], 'dolp_loss': [], 'dn_loss': [], 'all_loss': [], 
               's0_psnr': [],'dolp_psnr': [], 'aolp_psnr': [], 's0_ssim': [],'dolp_ssim': [], 'aolp_ssim': []}
    
    if opt.resume: # 将之前的result放到新的文件中
        if not os.path.exists('checkpoint/1_train_results.csv'):
            print("CSV file does not exist.")
        print("======================加============================")
        import pandas as pd
        # 读取CSV文件，跳过第一行和重新开始epoch之后
        df = pd.read_csv('checkpoint/1_train_results.csv', nrows=opt.start_epoch-1)
        # 去掉 'Epoch' 列
        df = df.drop(columns=['Epoch'])
        # 将 DataFrame 转换为字典，按行分配到相应的键上
        for column in df.columns:
            if column in results:  # 确保列名存在于 results 字典中
                results[column].extend(df[column].tolist())  # 使用 extend 添加数据到对应键
    # 打印结果查看
    # print(results)
    print(len(results['s0_psnr']),'=================================================')

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        running_results = train(training_data_loader, optimizer, model, criterion1, epoch, opt, criterion_aop)
        results['im_loss'].append(running_results['im_loss'] / running_results['batch_sizes'])
        results['dolp_loss'].append(running_results['dolp_loss'] / running_results['batch_sizes'])
        results['dn_loss'].append(running_results['dn_loss'] / running_results['batch_sizes'])
        results['all_loss'].append(running_results['all_loss'] / running_results['batch_sizes'])

        # if epoch % opt.test_epoch != 0: # 不是20的倍数 不测试下列指标
        #     results['s0_psnr'].append(0)
        #     results['dolp_psnr'].append(0)
        #     results['aolp_psnr'].append(0)
        #     results['s0_ssim'].append(0)
        #     results['dolp_ssim'].append(0)
        #     results['aolp_ssim'].append(0)

        if epoch % 250 == 0: # 每250个epoch保存一次模型
            save_checkpoint(model, epoch)

        if epoch % opt.test_epoch == 0: # opt.test_epoch=20,每20个epoch测试一次
            test_results = test(testing_data_loader, model, epoch, opt)
            results['s0_psnr'].append(test_results['s0_psnr'] / test_results['batch_sizes'])
            results['dolp_psnr'].append(test_results['dolp_psnr'] / test_results['batch_sizes'])
            results['aolp_psnr'].append(test_results['aolp_psnr'] / test_results['batch_sizes'])
            results['s0_ssim'].append(test_results['s0_ssim'] / test_results['batch_sizes'])
            results['dolp_ssim'].append(test_results['dolp_ssim'] / test_results['batch_sizes'])
            results['aolp_ssim'].append(test_results['aolp_ssim'] / test_results['batch_sizes'])
            
            if test_results['s0_psnr'] + test_results['dolp_psnr'] + test_results['aolp_psnr'] > best_psnr:                
                # 如果已经有一个之前的最佳模型文件，删除它
                if epoch >= 2:
                    # 设置目标文件夹路径（可以是当前目录，也可以是其他路径）
                    folder_path = "./checkpoint"  # 根据你的实际路径修改
                    import glob
                    # 使用glob查找文件名以"Best_demosaick_TxSnew_epoch"开头的文件
                    files_to_delete = glob.glob(os.path.join(folder_path, "Best_demosaick_TxSnew_epoch*"))
                    # 删除找到的文件
                    for file in files_to_delete:
                        try:
                            os.remove(file)  # 删除文件
                            print(f"已删除文件: {file}")
                        except Exception as e:
                            print(f"删除文件 {file} 时发生错误: {e}")
                save_best_checkpoint(model, epoch)
                best_psnr = test_results['s0_psnr'] + test_results['dolp_psnr'] + test_results['aolp_psnr']
            
            # 保存s0_psnr最高的权重 
            if test_results['s0_psnr'] > best_s0psnr:                
                # 如果已经有一个之前的最佳模型文件，删除它
                if epoch >= 2:
                    # 设置目标文件夹路径（可以是当前目录，也可以是其他路径）
                    folder_path = "./checkpoint"  # 根据你的实际路径修改
                    import glob
                    # 使用glob查找文件名以"Best_demosaick_TxSnew_epoch"开头的文件
                    files_to_delete = glob.glob(os.path.join(folder_path, "Best_s0_demosaick_TxSnew_epoch*"))
                    # 删除找到的文件
                    for file in files_to_delete:
                        try:
                            os.remove(file)  # 删除文件
                            print(f"已删除文件: {file}")
                        except Exception as e:
                            print(f"删除文件 {file} 时发生错误: {e}")
                save_best_s0_checkpoint(model, epoch)
                best_s0psnr = test_results['s0_psnr']              
                
            # 保存dolp_psnr最高的权重 
            if test_results['dolp_psnr'] > best_dolppsnr:                
                # 如果已经有一个之前的最佳模型文件，删除它
                if epoch >= 2:
                    # 设置目标文件夹路径（可以是当前目录，也可以是其他路径）
                    folder_path = "./checkpoint"  # 根据你的实际路径修改
                    import glob
                    # 使用glob查找文件名以"Best_demosaick_TxSnew_epoch"开头的文件
                    files_to_delete = glob.glob(os.path.join(folder_path, "Best_dolp_demosaick_TxSnew_epoch*"))
                    # 删除找到的文件
                    for file in files_to_delete:
                        try:
                            os.remove(file)  # 删除文件
                            print(f"已删除文件: {file}")
                        except Exception as e:
                            print(f"删除文件 {file} 时发生错误: {e}")
                save_best_dolp_checkpoint(model, epoch)
                best_dolppsnr = test_results['dolp_psnr']               
                
            # 保存aop_psnr最高的权重 
            if test_results['aolp_psnr'] > best_aoppsnr:                
                # 如果已经有一个之前的最佳模型文件，删除它
                if epoch >= 2:
                    # 设置目标文件夹路径（可以是当前目录，也可以是其他路径）
                    folder_path = "./checkpoint"  # 根据你的实际路径修改
                    import glob
                    # 使用glob查找文件名以"Best_demosaick_TxSnew_epoch"开头的文件
                    files_to_delete = glob.glob(os.path.join(folder_path, "Best_aop_demosaick_TxSnew_epoch*"))
                    # 删除找到的文件
                    for file in files_to_delete:
                        try:
                            os.remove(file)  # 删除文件
                            print(f"已删除文件: {file}")
                        except Exception as e:
                            print(f"删除文件 {file} 时发生错误: {e}")
                save_best_aop_checkpoint(model, epoch)
                best_aoppsnr = test_results['aolp_psnr']                                     
                
            save_statistics(opt, results, epoch)
            update_plot_psnr(results, epoch, opt)
            update_plot_loss(results, epoch, opt)
            update_plot_ssim(results, epoch, opt)
            
            

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.drop_rate ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion1, epoch, opt, criterion_aop):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'im_loss': 0, 'im_loss1': 0, 'dolp_loss': 0, 'aolp_loss': 0, 'dn_loss': 0,
                       'all_loss': 0, 'polar_loss': 0}
    model.train()

    for batch in train_bar:

        if opt.mosaic_aug:
            pattern = PATTERN[np.random.randint(low=0, high=8), :]
        else:
            pattern = np.array([1, 2, 3, 4])

        # raw_noise, input_image, target
        # torch.Size([1, 480, 640]), torch.Size([4, 480, 640]),torch.Size([4, 480, 640])
        # noisy DoFP, noisy DoT_LR, DoT_HR
        input_raw, input_image, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)

        N, C, H, W = batch[0].size()

        running_results['batch_sizes'] += N

        if opt.cuda:
            input_image = input_image.cuda()
            input_raw = input_raw.cuda()
            label_x4 = label_x4.cuda()

        # input_image, input_raw, label_x4 = mask_input_tensor(label_x4, msfa_size=2, pattern=pattern)
        dn, HR_4x, HR_4x_stage_1, BI = model([input_image, input_raw])
        # dn, HR_4x, HR_4x_stage_1 = model([input_image, input_raw])
        input_i, input_label, label_x4 = mask_input_tensor(label_x4, msfa_size=2, pattern=pattern)
        # HR_4x = model(input_raw)

        # I0_target = label_x4[:, int(np.argwhere(pattern == 1)), :, :]
        # I45_target = label_x4[:, int(np.argwhere(pattern == 2)), :, :]
        # I90_target = label_x4[:, int(np.argwhere(pattern == 4)), :, :]
        # I135_target = label_x4[:, int(np.argwhere(pattern == 3)), :, :]
        I0_target = label_x4[:, int(np.argwhere(pattern == 1).item()), :, :]
        I45_target = label_x4[:, int(np.argwhere(pattern == 2).item()), :, :]
        I90_target = label_x4[:, int(np.argwhere(pattern == 4).item()), :, :]
        I135_target = label_x4[:, int(np.argwhere(pattern == 3).item()), :, :]

        stokes_target = calculate_stokes_tensor(I0_target, I45_target, I90_target, I135_target)
        polar_target = ((I0_target - I45_target + I90_target - I135_target) + 2) / 4

        # I0_model = HR_4x[:, int(np.argwhere(pattern == 1)), :, :]
        # I45_model = HR_4x[:, int(np.argwhere(pattern == 2)), :, :]
        # I90_model = HR_4x[:, int(np.argwhere(pattern == 4)), :, :]
        # I135_model = HR_4x[:, int(np.argwhere(pattern == 3)), :, :]
        I0_model = HR_4x[:, int(np.argwhere(pattern == 1).item()), :, :]
        I45_model = HR_4x[:, int(np.argwhere(pattern == 2).item()), :, :]
        I90_model = HR_4x[:, int(np.argwhere(pattern == 4).item()), :, :]
        I135_model = HR_4x[:, int(np.argwhere(pattern == 3).item()), :, :]

        stokes_model = calculate_stokes_tensor(I0_model, I45_model, I90_model, I135_model)
        polar_model = ((I0_model - I45_model + I90_model - I135_model) + 2)/4
        
        # 损失函数
        loss_dn = opt.dn_loss_lambda * criterion1(dn, input_label) #1.8
        loss_x4_stage_1 = opt.stage1_loss_lambda * criterion1(HR_4x_stage_1, label_x4)
        loss_x4 = opt.img_loss_lambda * criterion1(HR_4x, label_x4)
        loss_dolp = opt.dolp_loss_lambda * criterion1(stokes_model['DOLP'], stokes_target['DOLP']) #1
        loss_aolp = opt.aolp_loss_lambda * criterion_aop(stokes_model['AOLP'], stokes_target['AOLP'])  # 1
        loss_polar = opt.polar_loss_lambda * criterion1(polar_model, polar_target) 
        # 总损失
        loss = loss_dn + loss_x4_stage_1 + loss_x4 + loss_dolp + loss_aolp + loss_polar

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_results['im_loss'] += loss_x4.item()
        running_results['im_loss1'] += loss_x4_stage_1.item()
        running_results['dolp_loss'] += loss_dolp.item()
        running_results['aolp_loss'] += loss_aolp.item()
        running_results['dn_loss'] += loss_dn.item()
        running_results['polar_loss'] += loss_polar.item()
        running_results['all_loss'] += loss.item()
        train_bar.set_description(
            desc='[%d/%d] dn: %.1f im1: %.1f im2: %.1f dop: %.1f aop: %.1f polar: %.1f all: %.1f' % (
                epoch, 
                opt.nEpochs, 
                10000000 * running_results['dn_loss'] / running_results['batch_sizes'],
                10000000 * running_results['im_loss1'] / running_results['batch_sizes'],
                10000000 * running_results['im_loss'] / running_results['batch_sizes'],
                10000000 * running_results['dolp_loss'] / running_results['batch_sizes'],
                10000000 * running_results['aolp_loss'] / running_results['batch_sizes'],
                10000000 * running_results['polar_loss'] / running_results['batch_sizes'],
                10000000 * running_results['all_loss'] / running_results['batch_sizes']))
    
    return running_results


def test(testing_data_loader, model, epoch, opt):

    test_bar = tqdm(testing_data_loader)
    test_results = {'batch_sizes': 0, 's0_psnr': 0, 's0_mse': 0, 'dolp_psnr': 0, 'dolp_mse': 0, 'aolp_psnr': 0, 'aolp_mse': 0,
                    'i0_psnr': 0, 'i0_mse': 0, 'i45_psnr': 0, 'i45_mse': 0, 'i90_psnr': 0, 'i90_mse': 0, 'i135_psnr': 0, 'i135_mse': 0,
                    's0_ssim': 0, 'dolp_ssim': 0, 'aolp_ssim': 0,
                    'S0': 0, 'DOLP': 0, 'AOLP': 0}
    model.eval()

    with torch.no_grad():
        for batch in test_bar:
            input_raw, input, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(
            batch[2], requires_grad=False)

            N, C, H, W = batch[0].size()

            test_results['batch_sizes'] += N


            if opt.cuda:
                input = input.cuda()
                input_raw = input_raw.cuda()
                label_x4 = label_x4.cuda()

            # HR_4x = model(input_raw)

            dn, HR_4x, HR_4x_stage_1, BI = model([input, input_raw])
            # dn, HR_4x, HR_4x_stage_1 = model([input, input_raw])
            # calculate s0,dolp psnr from model and bilinear

            i0_output = HR_4x[:, 0, :, :].detach().cpu().numpy() # (1,480,640)
            i45_output = HR_4x[:, 1, :, :].detach().cpu().numpy()
            i90_output = HR_4x[:, 3, :, :].detach().cpu().numpy()
            i135_output = HR_4x[:, 2, :, :].detach().cpu().numpy()

            i0_gt = label_x4[:, 0, :, :].detach().cpu().numpy() # (1,480,640)
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


def save_best_checkpoint(model, epoch):

    model_folder = "checkpoint/"

    model_out_path = model_folder + "Best_demosaick_TxSnew_epoch{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_best_s0_checkpoint(model, epoch):

    model_folder = "checkpoint/"

    model_out_path = model_folder + "Best_s0_demosaick_TxSnew_epoch{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_best_dolp_checkpoint(model, epoch):

    model_folder = "checkpoint/"

    model_out_path = model_folder + "Best_dolp_demosaick_TxSnew_epoch{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
                            
def save_best_aop_checkpoint(model, epoch):

    model_folder = "checkpoint/"

    model_out_path = model_folder + "Best_aop_demosaick_TxSnew_epoch{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_checkpoint(model, epoch):

    model_folder = "checkpoint/"
    model_out_path = model_folder + "Demosaick_epoch_pre_TxSnew_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_opt(opt):

    statistics_folder = "checkpoint/"
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=vars(opt), index=range(1, 2))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_opt.csv', index_label='Epoch')
    print("save--opt")


def save_statistics(opt, results, epoch):

    statistics_folder = "checkpoint/"
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=results, index=range(1, epoch + 1))

    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_train_results.csv', index_label='Epoch')

# 保存训练过程的折线图
def update_plot_psnr(results, epoch, opt):
    """
    更新图像并保存。
    
    参数:
        s0: 当前epoch对应的s0值
        dolp: 当前epoch对应的dolp值
        aop: 当前epoch对应的aop值
        epoch: 当前的epoch值
    """
    # print(len(results['s0_psnr']), epoch)
    # 初始化保存的图像文件路径
    plot_filename = 'checkpoint/{}_PSNR.pdf'.format(opt.start_epoch)
    

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Epoch vs PSNR (s0, dolp, aop)')
    plt.grid(True)

    # 绘制三条折线
    plt.plot(range(1, epoch + 1), results['s0_psnr'], label='s0', color='blue')
    plt.plot(range(1, epoch + 1), results['dolp_psnr'], label='dolp', color='green')
    plt.plot(range(1, epoch + 1), results['aolp_psnr'], label='aop', color='red')

    # 添加图例
    plt.legend()

    # 保存图像到 checkpoint 目录
    plt.savefig(plot_filename)
    # plt.savefig(self.get_path('test_{}.pdf'.format(d)))
    plt.close()

def update_plot_loss(results, epoch, opt):
    """
    更新图像并保存。
    
    参数:
        s0: 当前epoch对应的s0值
        dolp: 当前epoch对应的dolp值
        aop: 当前epoch对应的aop值
        epoch: 当前的epoch值
    """
    # 初始化保存的图像文件路径
    plot_filename = 'checkpoint/{}_Loss.pdf'.format(opt.start_epoch)
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.grid(True)

    # 绘制三条折线
    plt.plot(range(1, epoch + 1), results['im_loss'], label='im_loss', color='blue')
    plt.plot(range(1, epoch + 1), results['dolp_loss'], label='dolp_loss', color='green')
    plt.plot(range(1, epoch + 1), results['dn_loss'], label='dn_loss', color='red')
    plt.plot(range(1, epoch + 1), results['all_loss'], label='all_loss', color='yellow')


    plt.legend()

    # 保存图像到 checkpoint 目录
    plt.savefig(plot_filename)
    plt.close()

def update_plot_ssim(results, epoch, opt):
    """
    更新图像并保存。
    
    参数:
        s0: 当前epoch对应的s0值
        dolp: 当前epoch对应的dolp值
        aop: 当前epoch对应的aop值
        epoch: 当前的epoch值
    """
    # 初始化保存的图像文件路径
    plot_filename = 'checkpoint/{}_SSIM.pdf'.format(opt.start_epoch)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Epoch vs SSIM (s0, dolp, aop)')
    plt.grid(True)

    # 绘制三条折线
    plt.plot(range(1, epoch + 1), results['s0_ssim'], label='s0', color='blue')
    plt.plot(range(1, epoch + 1), results['dolp_ssim'], label='dolp', color='green')
    plt.plot(range(1, epoch + 1), results['aolp_ssim'], label='aop', color='red')

    plt.legend()

    # 保存图像到 checkpoint 目录
    plt.savefig(plot_filename)
    plt.close()


if __name__ == "__main__":
    main()
