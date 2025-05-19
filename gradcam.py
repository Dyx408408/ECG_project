# import argparse
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# from SRA18 import ResNet18_1D
# from dataset import ECGDataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from utils import cal_f1s, cal_aucs, split_data
# import physionet_challenge_utility_script as pc

# #############################################
# # Grad-CAM 实现（使用 register_full_backward_hook）
# #############################################
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
#         # grad_output 为一个 tuple，取第一个即可
#         self.gradients = grad_output[0]

#     def __call__(self, input_tensor, target_class=None):
#         """
#         Args:
#             input_tensor: 输入信号，形状 (B, in_channels, L)
#             target_class: 指定类别索引（若为 None，则取预测得分最高的类别）
#         Returns:
#             cam: Grad-CAM 热力图，形状 (B, L)，数值归一化到 [0,1]
#         """
#         # 前向传播
#         output = self.model(input_tensor)
#         if isinstance(output, tuple):
#             logits = output[0]
#         else:
#             logits = output
#         if target_class is None:
#             target_class = logits.argmax(dim=1).item()
#         # 选择目标类别得分
#         score = logits[0, target_class]
#         self.model.zero_grad()
#         score.backward(retain_graph=True)
        
#         # 获取捕获到的梯度和特征图，形状 (B, C, L_feat)
#         gradients = self.gradients
#         activations = self.activations
        
#         # 计算权重：对梯度在时序维度 L_feat 求平均，形状 (B, C, 1)
#         weights = gradients.mean(dim=2, keepdim=True)
#         # 加权求和得到热力图，形状 (B, L_feat)
#         cam = (weights * activations).sum(dim=1)
#         cam = F.relu(cam)
#         # 上采样热力图到输入信号的长度
#         cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-1],
#                             mode='linear', align_corners=False)
#         cam = cam.squeeze(1)
#         # 归一化到 [0,1]
#         for i in range(cam.shape[0]):
#             cam_min = cam[i].min()
#             cam_max = cam[i].max()
#             cam[i] = (cam[i] - cam_min) / (cam_max - cam_min + 1e-8)
#         return cam

#     def remove_hooks(self):
#         for handle in self.hook_handles:
#             handle.remove()

# #############################################
# # 绘图辅助函数
# #############################################
# # def plot_gradcam(ecg_signal, cam, idx, channel=0):
# #     """
# #     绘制某个导联的 ECG 信号及其 Grad-CAM 热力图
    
# #     Args:
# #         ecg_signal: numpy 数组，形状 (L,) 表示单通道 ECG 信号
# #         cam: numpy 数组，形状 (L,) 对应时序热力图（数值在 [0,1]）
# #         channel: 指示绘制哪个导联（仅用于标题说明）
# #     """
# #     # print(ecg_signal.shape)
# #     L = ecg_signal.shape[0]
# #     x_axis = np.arange(L)
    
# #     plt.figure(figsize=(12, 4))
# #     plt.plot(x_axis, ecg_signal, label='ECG Signal (Channel {})'.format(channel), color='black')
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.imshow(cam[np.newaxis, :], aspect='auto', cmap='jet', 
# #                alpha=0.5, extent=[0, L, ecg_signal.min(), ecg_signal.max()])
# #     # plt.colorbar(label='Grad-CAM')
# #     # plt.title('Grad-CAM Visualization on ECG Signal (Channel {})'.format(channel))
# #     # plt.xlabel('Time')
# #     # plt.ylabel('Amplitude')
# #     # plt.legend()
# #     # plt.tight_layout()
# #     if not os.path.exists(os.path.join('grad_cam1', str(idx))):
# #         os.mkdir(os.path.join('grad_cam1', str(idx)))
# #     plt.savefig(os.path.join('grad_cam1', str(idx), str(channel)))
# def plot_gradcam(ecg_signal, cam, idx, channel=0):
#     """
#     绘制某个导联的 ECG 信号，并将 Grad-CAM 热力图数值沿信号路径上色
    
#     Args:
#         ecg_signal: numpy 数组，形状 (L,) 表示单通道 ECG 信号
#         cam: numpy 数组，形状 (L,) 对应时序热力图（数值在 [0,1]）
#         channel: 指示绘制哪个导联（仅用于标题说明）
#     """
#     lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#     L = ecg_signal.shape[0]
#     x_axis = np.arange(L)
    
#     # 构造分段线条，每个线段根据 Grad-CAM 的数值上色
#     points = np.array([x_axis, ecg_signal]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
#     from matplotlib.collections import LineCollection
#     lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, 1))
#     # 将 Grad-CAM 数值赋值到线条上（颜色将依据此数值映射）
#     lc.set_array(cam)
#     lc.set_linewidth(2)
    
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.add_collection(lc)
#     ax.set_xlim(x_axis.min(), x_axis.max())
#     ax.set_ylim(ecg_signal.min(), ecg_signal.max())
    
#     ax.set_title('Grad-CAM Visualization on ECG Signal (Channel {})'.format(lead_labels[channel]))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # 若需要显示颜色条，可以取消下一行注释
#     # fig.colorbar(lc, ax=ax, label='Grad-CAM Intensity')
    
#     # 保存图片
#     save_dir = os.path.join('grad_cam1', str(idx))
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     plt.savefig(os.path.join(save_dir, lead_labels[channel] + '.png'))
#     plt.close(fig)
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', type=str, default='/home/dyx/CFGAN/dyx1/CPSC', help='Directory for data dir')
#     parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
#     parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
#     parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
#     parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
#     parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
#     parser.add_argument('--num-workers', type=int, default=0, help='Num of workers to load data')
#     parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
#     parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
#     parser.add_argument('--resume', default=False, action='store_true', help='Resume')
#     parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
#     parser.add_argument('--model-path', type=str, default='/home/dyx/CFGAN/dyx1/models/best_model.pth', help='Path to saved model')
#     return parser.parse_args()

# if __name__ == '__main__':
#     args = parse_args()
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
#     # label_csv = 'dyx/labels.csv'
#     train_folds, val_folds, test_folds = split_data(seed=args.seed)
#     train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
#     val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#     test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


#     model = ResNet18_1D(num_classes=6, in_channels=12)
#     model.eval()  # 设置为评估模式
#     model.load_state_dict(torch.load('/home/dyx/CFGAN/dyx1/models/best_model.pth', map_location=device))
#     # 选取目标层：这里选择 layer4 中最后一个 BasicBlock 的 conv2 层
#     target_layer = model.layer4[-1].conv2
    
#     idx='A0114'
#     # idx = 'A0018'
#     # idx = 'A0070'
#     # idx = 'A0069'
#     # norm data
#     # idx = 'A6870'
#     # idx = 'A6853'
#     # idx = 'A6801'
#     # idx = 'A6114'
#     # idx = 'A5373'
#     # 读取对应的ECG数据
#     ecg_data = pc.load_challenge_data(f"/home/dyx/CFGAN/dyx1/CPSC/{idx}.mat")
#     ecg_data = torch.from_numpy(ecg_data[0][:,-5000:]).float().unsqueeze(0)
    
#     gradcam = GradCAM(model, target_layer)
#     # 计算 Grad-CAM 热力图
#     cam = gradcam(ecg_data)  # 默认为预测类别的得分
#     cam = cam[0].detach().cpu().numpy()  # 取 batch 中第一个样本
#     cam = cam[1000:2000]

#     for n in range(12):
#         # 绘制第 n 个导联的 ECG 信号与热力图叠加
#         ecg_signal = ecg_data[0, n].detach().cpu().numpy()
#         plot_gradcam(ecg_signal[1000:2000], cam, idx, channel=n)
#         # 移除 hook
#         gradcam.remove_hooks()

#     # 实例化 GradCAM 类
#     # for idx, (data, labels) in enumerate(tqdm(test_loader)):
#     #     gradcam = GradCAM(model, target_layer)
#     #      # 计算 Grad-CAM 热力图
#     #     cam = gradcam(data)  # 默认为预测类别的得分
#     #     cam = cam[0].detach().cpu().numpy()  # 取 batch 中第一个样本
#     #     cam = cam[1000:2000]
#     #     for n in range(12):
#     #         # 绘制第 n 个导联的 ECG 信号与热力图叠加
#     #         ecg_signal = data[0, n].detach().cpu().numpy()
#     #         plot_gradcam(ecg_signal[1000:2000], cam, idx, channel=n)
#     #         # 移除 hook
#     #         gradcam.remove_hooks()

        

   

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from SRA18 import ResNet18_1D
from dataset import ECGDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import cal_f1s, cal_aucs, split_data
import physionet_challenge_utility_script as pc

#############################################
# Grad-CAM 实现（使用 register_full_backward_hook）
#############################################
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

#############################################
# 绘图辅助函数
#############################################
def plot_gradcam(ecg_signal, cam, idx, channel=0):
    """
    绘制某个导联的 ECG 信号，同时将 Grad-CAM 热力图以全图形式显示，
    图中 ECG 信号以固定颜色绘制，不再根据热力图进行上色。

    Args:
        ecg_signal: numpy 数组，形状 (L,) 表示单通道 ECG 信号
        cam: numpy 数组，形状 (L,) 对应时序热力图（数值在 [0,1]）
        channel: 指示绘制哪个导联（仅用于标题说明）
    """
    # ECG 导联标签列表
    lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    L = ecg_signal.shape[0]
    x_axis = np.arange(L)
    
    # 构造2D热力图：将1D的 cam 重复多次，形成一个矩阵
    heatmap = np.tile(cam, (100, 1))  # 例如重复100行
    
    fig, ax = plt.subplots(figsize=(12, 4))
    # 显示热力图，填满整个图片
    im = ax.imshow(heatmap, aspect='auto', cmap='jet', 
                   extent=[0, L, ecg_signal.min(), ecg_signal.max()],
                   origin='lower')
    # 叠加绘制 ECG 信号（固定颜色，如黑色）
    ax.plot(x_axis, ecg_signal, color='black', linewidth=2)
    
    ax.set_title('Grad-CAM Heatmap on ECG Signal (Lead {})'.format(lead_labels[channel]))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label='Grad-CAM Intensity')
    
    # 保存图片
    save_dir = os.path.join('grad_cam1', str(idx))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, lead_labels[channel] + '.png'))
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/dyx/CFGAN/dyx1/CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='/home/dyx/CFGAN/dyx1/models/best_model.pth', help='Path to saved model')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

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
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ResNet18_1D(num_classes=6, in_channels=12)
    model.eval()  # 设置为评估模式
    model.load_state_dict(torch.load('/home/dyx/CFGAN/dyx1/models/best_model.pth', map_location=device))
    # 选取目标层：这里选择 layer4 中最后一个 BasicBlock 的 conv2 层
    target_layer = model.layer4[-1].conv2
    
    # idx = 'A0114'
    # idx = 'A3651'
    # idx = 'A1440'
    idx = 'A0401'
    # idx = 'A0114'
    
    # 读取对应的ECG数据
    ecg_data = pc.load_challenge_data(f"/home/dyx/CFGAN/dyx1/CPSC/{idx}.mat")
    ecg_data = torch.from_numpy(ecg_data[0][:, -5000:]).float().unsqueeze(0)
    
    gradcam = GradCAM(model, target_layer)
    # 计算 Grad-CAM 热力图
    cam = gradcam(ecg_data)  # 默认为预测类别的得分
    cam = cam[0].detach().cpu().numpy()  # 取 batch 中第一个样本
    cam = cam[1000:2000]

    for n in range(12):
        # 绘制第 n 个导联的 ECG 信号与热力图叠加
        ecg_signal = ecg_data[0, n].detach().cpu().numpy()
        plot_gradcam(ecg_signal[1000:2000], cam, idx, channel=n)
    
    # 在所有导联绘制完后移除 hook
    gradcam.remove_hooks()
