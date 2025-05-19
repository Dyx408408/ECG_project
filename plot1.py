import torch
from torch import nn

import os
import torch
import argparse
import torch.nn as nn
from Unet import UNet1D
from dataset import ECGDataset
from torch.utils.data import DataLoader
from utils import cal_f1s, cal_aucs, split_data
import numpy as np
from tqdm import tqdm

from Unet import UNet1D
from SRA18 import ResNet18_1D

import physionet_challenge_utility_script as pc
import ecg_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=17, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    # parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', help='Path to saved model')
    return parser.parse_args()

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: 已训练的模型，例如 ResNet18_1D 实例
            target_layer: 用于生成 Grad-CAM 的目标层，本例选择最后 BasicBlock 中的 conv2 层
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        # 注册前向 hook 获取特征图
        self.hook_handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        # 使用 register_full_backward_hook 代替 register_backward_hook
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )
    
    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output 为一个 tuple，取第一个即可
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class=None):
        """
        Args:
            input_tensor: 输入信号，形状 (B, in_channels, L)
            target_class: 指定类别索引（若为 None，则取预测得分最高的类别）
        Returns:
            cam: Grad-CAM 热力图，形状 (B, L)，数值归一化到 [0,1]
        """
        # 前向传播
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        # 选择目标类别得分
        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # 获取捕获到的梯度和特征图，形状 (B, C, L_feat)
        gradients = self.gradients
        activations = self.activations
        
        # 计算权重：对梯度在时序维度 L_feat 求平均，形状 (B, C, 1)
        weights = gradients.mean(dim=2, keepdim=True)
        # 加权求和得到热力图，形状 (B, L_feat)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)
        # 上采样热力图到输入信号的长度
        cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-1],
                            mode='linear', align_corners=False)
        cam = cam.squeeze(1)
        # 归一化到 [0,1]
        for i in range(cam.shape[0]):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            cam[i] = (cam[i] - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

class CFGAN(nn.Module):
    def __init__(self, unet, self_attention_resnet18):
        super(CFGAN, self).__init__()
        self.unet = unet
        self.self_attention_resnet18 = self_attention_resnet18
        self.freeze_sar18()
        self.return_exp = False

    def freeze_sar18(self):
        for param in self.self_attention_resnet18.parameters():
            param.requires_grad = False
    def forward(self, inputs):
        _, attention_map = self.self_attention_resnet18(inputs)
        explainable_ecg = self.unet(inputs, attention_map)
        cf_ecg = inputs + explainable_ecg
        outputs, _ = self.self_attention_resnet18(cf_ecg)
        if self.return_exp:
            return explainable_ecg
        else:
            return outputs

def denoise(data):

    denoised_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        # 小波变换
        coeffs = pywt.wavedec(data=data[i], wavelet='db5', level=9)
        cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

        # 阈值去噪
        threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
        cD1.fill(0)
        cD2.fill(0)
        #cD3.fill(0)
        for j in range(1, len(coeffs) - 2):
            coeffs[j] = pywt.threshold(coeffs[j], threshold)

        # 小波反变换,获取去噪后的信号
        denoised_data[i] = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return denoised_data

import scipy.signal as signal



if __name__ == '__main__':
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    # if not args.model_path:
    #     args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)

    label_csv = os.path.join(data_dir, 'labels.csv')
    # label_csv = 'dyx/labels.csv'
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    SAR = ResNet18_1D(num_classes=6, in_channels=12)
    SAR = SAR.to(device)
    UNet = UNet1D(in_channels=12, out_channels=12, noise_strength=0.0)
    UNet = UNet.to(device)

    cfgan = CFGAN(UNet, SAR)
    cfgan.load_state_dict(torch.load('models/cfgan_ratio_0.8296943231441049.pth'),strict=True)
    # cfgan.load_state_dict(torch.load('models/best_model.pth'),strict=True)
    
    cfgan.to(device)
    cfgan.eval()
    cfgan.return_exp = True

    for id, (patient_id, data, labels) in enumerate(tqdm(val_loader)):
        # print(len(patient_id))
        # print(type(patient_id), type())
        data, labels = data.to(device), labels.to(device)
        explainable_ecg = cfgan(data)
        cf_ecg = explainable_ecg + data
        predict_labels, att = cfgan.self_attention_resnet18(data)
        predict_labels_cf, att_cf = cfgan.self_attention_resnet18(cf_ecg)

        # predict_x, attn = SAR(data)
        # predict_x = predict_x.detach().cpu().numpy()

        # 转换为numpy
        data = data.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        explainable_ecg = explainable_ecg.detach().cpu().numpy()
        cf_ecg = cf_ecg.detach().cpu().numpy()
        predict_labels = predict_labels.detach().cpu().numpy()
        predict_labels_cf = predict_labels_cf.detach().cpu().numpy()
        att = att.detach().cpu().numpy()
        att_cf = att_cf.detach().cpu().numpy()
        # for i in range(10):
        #     print(np.argmax(labels[i]), np.argmax(predict_labels[i]), np.argmax(predict_x[i]))

        for i in range(len(data)):
            if np.argmax(labels[i]) == np.argmax(predict_labels[i]): #and np.argmax(predict_labels[i]) == 0:
                ecg_plot.plot(denoise(data[i][:,1000:2000]), sample_rate=500, title='Original ECG')
                # ecg_plot.plot(data[i][:,1000:2000], sample_rate=500, title='Original ECG')
                ecg_plot.save_as_png(f'new_image/原始心电图{patient_id[i]}_{np.argmax(labels[i])}_{np.argmax(predict_labels[i])}_{np.argmax(predict_labels_cf[i])}',dpi=400)

                ecg_plot.plot(denoise(cf_ecg[i][:,1000:2000]), sample_rate=500, title='Counter-Factual ECG')
                ecg_plot.save_as_png(f'new_image/反事实心电图{patient_id[i]}_{np.argmax(labels[i])}_{np.argmax(predict_labels[i])}_{np.argmax(predict_labels_cf[i])}',dpi=400)

                ecg_plot.plot(denoise(explainable_ecg[i][:,1000:2000]), sample_rate=500, title='Expainable ECG')
                ecg_plot.save_as_png(f'new_image/反事实解释心电图{patient_id[i]}_{np.argmax(labels[i])}_{np.argmax(predict_labels[i])}_{np.argmax(predict_labels_cf[i])}',dpi=400)
                

                # plt.imshow(att[i])
                # plt.title('Attention Map')
                # plt.savefig(f'new_image/原始注意力热图{patient_id[i]}_{np.argmax(labels[i])}_{np.argmax(predict_labels[i])}_{np.argmax(predict_labels_cf[i])}',dpi=400)

                # plt.imshow(att_cf[i])
                # plt.title('Attention Map')
                # plt.savefig(f'new_image/反事实注意力热图{patient_id[i]}_{np.argmax(labels[i])}_{np.argmax(predict_labels[i])}_{np.argmax(predict_labels_cf[i])}',dpi=400)
   


# import torch
# from torch import nn
# import torch.nn.functional as F
# import os
# import argparse
# import numpy as np
# from tqdm import tqdm

# from Unet import UNet1D
# from dataset import ECGDataset
# from torch.utils.data import DataLoader
# from utils import cal_f1s, cal_aucs, split_data
# from SRA18 import ResNet18_1D

# import pywt
# import scipy.signal as signal
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import pandas as pd

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', type=str, default='CPSC', help='Directory for data dir')
#     parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
#     parser.add_argument('--seed', type=int, default=17, help='Seed to split data')
#     parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
#     parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
#     parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
#     parser.add_argument('--num-workers', type=int, default=0, help='Num of workers to load data')
#     parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
#     parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
#     parser.add_argument('--resume', default=False, action='store_true', help='Resume')
#     parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
#     parser.add_argument('--model-path', type=str, default='models/best_model.pth', help='Path to saved model')
#     return parser.parse_args()

# class GradCAM:
#     def __init__(self, model, target_layer):
#         """
#         Args:
#             model: 已训练的模型，例如 ResNet18_1D 实例
#             target_layer: 用于生成 Grad-CAM 的目标层，本例选择最后 BasicBlock 中的 conv2 层
#         """
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self.hook_handles = []
#         # 注册前向 hook 获取特征图
#         self.hook_handles.append(
#             self.target_layer.register_forward_hook(self._save_activation)
#         )
#         # 使用 register_full_backward_hook 代替 register_backward_hook
#         self.hook_handles.append(
#             self.target_layer.register_full_backward_hook(self._save_gradient)
#         )
    
#     def _save_activation(self, module, input, output):
#         self.activations = output

#     def _save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def __call__(self, input_tensor, target_class=None):
#         output = self.model(input_tensor)
#         if isinstance(output, tuple):
#             logits = output[0]
#         else:
#             logits = output
#         if target_class is None:
#             target_class = logits.argmax(dim=1).item()
#         score = logits[0, target_class]
#         self.model.zero_grad()
#         score.backward(retain_graph=True)
        
#         gradients = self.gradients
#         activations = self.activations
#         weights = gradients.mean(dim=2, keepdim=True)
#         cam = (weights * activations).sum(dim=1)
#         cam = F.relu(cam)
#         cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-1],
#                             mode='linear', align_corners=False)
#         cam = cam.squeeze(1)
#         for i in range(cam.shape[0]):
#             cam_min = cam[i].min()
#             cam_max = cam[i].max()
#             cam[i] = (cam[i] - cam_min) / (cam_max - cam_min + 1e-8)
#         return cam

#     def remove_hooks(self):
#         for handle in self.hook_handles:
#             handle.remove()

# class CFGAN(nn.Module):
#     def __init__(self, unet, self_attention_resnet18):
#         super(CFGAN, self).__init__()
#         self.unet = unet
#         self.self_attention_resnet18 = self_attention_resnet18
#         self.freeze_sar18()
#         self.return_exp = False

#     def freeze_sar18(self):
#         for param in self.self_attention_resnet18.parameters():
#             param.requires_grad = False

#     def forward(self, inputs):
#         _, attention_map = self.self_attention_resnet18(inputs)
#         explainable_ecg = self.unet(inputs, attention_map)
#         cf_ecg = inputs + explainable_ecg
#         outputs, _ = self.self_attention_resnet18(cf_ecg)
#         if self.return_exp:
#             return explainable_ecg
#         else:
#             return outputs

# def denoise(data):
#     denoised_data = np.zeros_like(data)
#     for i in range(data.shape[0]):
#         coeffs = pywt.wavedec(data=data[i], wavelet='db5', level=9)
#         cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
#         threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
#         cD1.fill(0)
#         cD2.fill(0)
#         for j in range(1, len(coeffs) - 2):
#             coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
#         denoised_data[i] = pywt.waverec(coeffs=coeffs, wavelet='db5')
#     return denoised_data

# def plot_subplots(x1, x2, main_ax, lead_label):
#     """在给定的主坐标轴中创建上方信号图和下方热力图"""
#     main_ax.remove()
#     gs = gridspec.GridSpecFromSubplotSpec(
#         2, 1, 
#         subplot_spec=main_ax.get_subplotspec(),
#         height_ratios=[5, 1],
#         hspace=0.05
#     )
#     ax_top = plt.subplot(gs[0])
#     ax_top.plot(x1, color='red', linewidth=0.6, label='Original ECG')
#     ax_top.plot(x2, color='black', linewidth=0.6, label='CF ECG')
#     ax_top.text(0.02, 0.95, lead_label, 
#                 transform=ax_top.transAxes,
#                 fontsize=9, weight='bold',
#                 va='top', ha='left')
#     ax_top.tick_params(axis='both', labelsize=7)
#     ax_top.set_ylim(-0.2, 1.2)
#     ax_top.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax_top.set_xticks([])
#     ax_top.legend(loc='upper right', fontsize=3)
    
#     ax_bottom = plt.subplot(gs[1])
#     diff = np.abs(x1 - x2)
#     ax_bottom.imshow(diff[np.newaxis, :], aspect='auto', cmap='jet',
#                      interpolation='nearest', vmin=0, vmax=1)
#     ax_bottom.set_yticks([])
#     ax_bottom.set_xticks([])
#     ax_bottom.xaxis.set_ticks_position('bottom')

# def nice_ecg_plots(x1, x2, save_path):
#     """
#     绘制12导联心电图，每个导联包括上方信号图与下方热力图，
#     左侧显示标准导联，右侧显示胸导联。
    
#     参数：
#         x1: 原始信号，形状 (12, N)
#         x2: 反事实信号，形状 (12, N)
#         save_path: 若不为 False，则保存图片；否则调用 plt.show()
#     """
#     labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
#               'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#     fig = plt.figure(figsize=(10, 15), dpi=120)
#     main_axes = fig.subplots(6, 2, gridspec_kw={'hspace': 0.4, 'wspace': 0.15})
#     for row in range(6):
#         lead_idx = row
#         plot_subplots(x1[lead_idx], x2[lead_idx], main_axes[row, 0], labels[lead_idx])
#         chest_idx = row + 6
#         plot_subplots(x1[chest_idx], x2[chest_idx], main_axes[row, 1], labels[chest_idx])
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#     else:
#         plt.show()

# if __name__ == '__main__':
#     args = parse_args()
#     args.best_metric = 0
#     data_dir = os.path.normpath(args.data_dir)
#     database = os.path.basename(data_dir)
    
#     if args.use_gpu and torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     else:
#         device = 'cpu'
    
#     if args.leads == 'all':
#         leads = 'all'
#         nleads = 12
#     else:
#         leads = args.leads.split(',')
#         nleads = len(leads)
    
#     label_csv = os.path.join(data_dir, 'labels.csv')
#     train_folds, val_folds, test_folds = split_data(seed=args.seed)
#     train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
#     val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#     test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
#     # 构建模型
#     SAR = ResNet18_1D(num_classes=6, in_channels=12).to(device)
#     UNet = UNet1D(in_channels=12, out_channels=12, noise_strength=0.0).to(device)
#     cfgan = CFGAN(UNet, SAR)
#     cfgan.load_state_dict(torch.load('models/cfgan_ratio_0.8296943231441049.pth'), strict=True)
#     cfgan.to(device)
#     cfgan.eval()
#     cfgan.return_exp = True
    
#     for patient_id, data, labels in tqdm(val_loader):
#         data, labels = data.to(device), labels.to(device)
#         explainable_ecg = cfgan(data)
#         cf_ecg = explainable_ecg + data
#         predict_labels, att = cfgan.self_attention_resnet18(data)
#         predict_labels_cf, att_cf = cfgan.self_attention_resnet18(cf_ecg)
    
#         # 转换为 numpy 数组
#         data = data.detach().cpu().numpy()
#         cf_ecg = cf_ecg.detach().cpu().numpy()
#         labels = labels.detach().cpu().numpy()
#         predict_labels = predict_labels.detach().cpu().numpy()
#         predict_labels_cf = predict_labels_cf.detach().cpu().numpy()
    
#         for i in range(len(data)):
#             # 只对预测标签正确的样本进行绘图
#             if np.argmax(labels[i]) == np.argmax(predict_labels[i]):
#                 # 计算x1和x2：对原始与反事实信号选取时间段1000:2000后去噪
#                 x1 = denoise(data[i][:, 1000:2000])
#                 x2 = denoise(cf_ecg[i][:, 1000:2000])
#                 # 构造保存路径，可根据 patient_id 及标签信息命名
#                 save_path = f'new_image/ecg_{patient_id[i]}_{np.argmax(labels[i])}_{np.argmax(predict_labels[i])}_{np.argmax(predict_labels_cf[i])}.png'
#                 nice_ecg_plots(x1, x2, save_path=save_path)
