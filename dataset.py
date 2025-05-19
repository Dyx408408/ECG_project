import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig


class ECGDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(ECGDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv('labels.csv')
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        # self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'STD']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-5000:, self.use_leads]
        result = np.zeros((5000, self.nleads)) # 10 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id) is not None and self.label_dict.get(patient_id).any():
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return patient_id, torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)


# import os
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import numpy as np
# import wfdb
# import pywt

# def scaling(X, sigma=0.1):
#     scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
#     myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
#     return X * myNoise

# def shift(sig, interval=20):
#     for col in range(sig.shape[1]):
#         offset = np.random.choice(range(-interval, interval))
#         sig[:, col] += offset / 1000 
#     return sig

# def transform(sig, train=False):
#     if train:
#         if np.random.randn() > 0.5: 
#             sig = scaling(sig)
#         if np.random.randn() > 0.5: 
#             sig = shift(sig)
#     return sig

# def denoise_db8(data):
#     """
#     使用 db8 小波对 ECG 信号进行去噪处理。
    
#     参数：
#         data: numpy 数组，形状 (num_channels, num_samples)
#     返回：
#         denoised_data: 去噪后的信号，形状与 data 相同
#     """
#     denoised_data = np.zeros_like(data)
#     # 对每个导联单独处理
#     for i in range(data.shape[0]):
#         # 小波分解：使用 db8 小波，分解层数设为 9
#         coeffs = pywt.wavedec(data[i], wavelet='db8', level=9)
#         # 计算阈值：基于最高频细节系数
#         threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * np.sqrt(2 * np.log(len(coeffs[-1])))
#         # 对各细节系数应用软阈值处理
#         for j in range(1, len(coeffs)):
#             coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
#         # 小波重构
#         rec = pywt.waverec(coeffs, wavelet='db8')
#         # 保证重构后的信号长度与原始一致（截断或填充零）
#         if rec.shape[0] > data.shape[1]:
#             rec = rec[:data.shape[1]]
#         elif rec.shape[0] < data.shape[1]:
#             rec = np.pad(rec, (0, data.shape[1] - rec.shape[0]), 'constant')
#         denoised_data[i] = rec
#     return denoised_data

# class ECGDataset(Dataset):
#     def __init__(self, phase, data_dir, label_csv, folds, leads):
#         super(ECGDataset, self).__init__()
#         self.phase = phase
#         df = pd.read_csv('labels.csv')
#         df = df[df['fold'].isin(folds)]
#         self.data_dir = data_dir
#         self.labels = df
#         self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
#                       'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#         if leads == 'all':
#             self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
#         else:
#             self.use_leads = np.where(np.in1d(self.leads, leads))[0]
#         self.nleads = len(self.use_leads)
#         # 使用的分类标签，可根据实际情况调整
#         self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'STD']
#         self.n_classes = len(self.classes)
#         self.data_dict = {}
#         self.label_dict = {}

#     def __getitem__(self, index: int):
#         row = self.labels.iloc[index]
#         patient_id = row['patient_id']
#         # 读取信号
#         ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
#         ecg_data = transform(ecg_data, self.phase == 'train')
#         nsteps, _ = ecg_data.shape
#         # 取最后5000个样本，并选取需要的导联
#         ecg_data = ecg_data[-5000:, self.use_leads]
#         result = np.zeros((5000, self.nleads))  # 10 s, 500 Hz
#         result[-nsteps:, :] = ecg_data
#         # 对结果进行去噪：先转置为 (导联数, 数据点数)，方便逐导联去噪
#         result = result.transpose()  # 形状 (nleads, 5000)
#         result = denoise_db8(result)
#         # 返回的 tensor 形状为 (导联数, 数据点数)
#         return patient_id, torch.from_numpy(result).float(), torch.from_numpy(row[self.classes].to_numpy(dtype=np.float32)).float()

#     def __len__(self):
#         return len(self.labels)
