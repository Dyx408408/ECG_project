import torch
from torch import nn
import os
import argparse
import numpy as np
from tqdm import tqdm

from Unet import UNet1D
from dataset import ECGDataset
from torch.utils.data import DataLoader
from utils import cal_f1s, cal_aucs, split_data
from SRA18 import ResNet18_1D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=9, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='checkpoint_3.9/best_model.pth', help='Path to saved model')
    return parser.parse_args()

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
        # 首先通过自注意力 ResNet18 得到注意力图
        _, attention_map = self.self_attention_resnet18(inputs)
        # 利用 U-Net 和注意力图生成可解释的ECG残差
        explainable_ecg = self.unet(inputs, attention_map)
        # 反事实ECG由原始输入和残差相加得到
        cf_ecg = inputs + explainable_ecg
        outputs, _ = self.self_attention_resnet18(cf_ecg)
        if self.return_exp:
            return explainable_ecg
        else:
            return outputs, cf_ecg

# 新增的1D鉴别器，用于判断输入的ECG信号是真实的还是生成的
class Discriminator(nn.Module):
    def __init__(self, in_channels=12):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        out = self.model(x)
        # 输出形状(batch, 1, 1)转换为(batch,)
        return out.view(-1)

def train(dataloader, net, disc, args, criterion, adv_loss_fn, epoch, scheduler, optimizer_g, optimizer_d, device):
    mseloss = nn.MSELoss()
    print('Training epoch %d:' % epoch)
    net.train()
    net.return_exp = False
    running_loss_g = 0
    running_loss_d = 0
    for _, (_, data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        # 构造虚假标签（用于分类部分）
        fake_labels_cls = torch.zeros(labels.shape).to(device)
        fake_labels_cls[:, 0] = 1.
        
        # -------------------------------
        # 1. 更新鉴别器 Discriminator
        # -------------------------------
        # 利用生成器得到反事实ECG，注意detach防止梯度传递到生成器
        with torch.no_grad():
            _, cf_ecg = net(data)
        # 真实信号标签为1，生成信号标签为0
        real_labels_adv = torch.ones(data.size(0)).to(device)
        fake_labels_adv = torch.zeros(data.size(0)).to(device)
        
        output_real = disc(data)
        output_fake = disc(cf_ecg.detach())
        loss_d_real = adv_loss_fn(output_real, real_labels_adv)
        loss_d_fake = adv_loss_fn(output_fake, fake_labels_adv)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        running_loss_d += loss_d.item()
        
        # -------------------------------
        # 2. 更新生成器（CFGAN）
        # -------------------------------
        output, cf_ecg = net(data)
        # 分类损失
        loss_cls = 0.5 * criterion(output, fake_labels_cls)
        # 重构（MSE）损失，确保反事实ECG与原始输入接近
        loss_recon = 0.2 * mseloss(data, cf_ecg)
        # 对抗性损失：生成器希望让鉴别器认为生成的cf_ecg是真实的（标签为1）
        output_adv = disc(cf_ecg)
        loss_adv = adv_loss_fn(output_adv, real_labels_adv)
        # 总生成器损失，lambda_adv为对抗性损失的权重（可根据需要调节）
        lambda_adv = 0.3
        loss_g = loss_cls + loss_recon + lambda_adv * loss_adv
        
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        running_loss_g += loss_g.item()
    
    print('Generator Loss: %.4f, Discriminator Loss: %.4f' % (running_loss_g, running_loss_d))
    with open('running_loss_g.txt', 'a+',) as txt:
        txt.write(str(running_loss_g)+'\n')
    
    with open('running_loss_d.txt', 'a+') as txt1:
        txt1.write(str(running_loss_d)+'\n')
    # scheduler.step()  # 如需要可对生成器优化器调整学习率

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    net.return_exp = True
    running_loss = 0
    output_list, labels_list = [], []
    classification_results = []  # 用于记录每个样本的分类情况（1 表示正常，0 表示非正常）

    for _, (_, data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        # 通过生成器获得解释性ECG残差，再加上原始数据得到反事实ECG
        explainable_ecg = net(data)
        cf_ecg = explainable_ecg + data
        # 使用冻结的 ResNet 进行分类预测
        outputs, _ = net.self_attention_resnet18(cf_ecg)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        # 对输出进行 sigmoid 激活
        outputs_prob = torch.sigmoid(outputs)
        output_list.append(outputs_prob.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
        
        # 这里假设正常样本在标签向量的第一位
        # 若该位置的概率大于0.5，则判定为正常样本
        preds = (outputs_prob.data.cpu().numpy()[:, 0] > 0.5).astype(int)
        classification_results.extend(preds.tolist())

    print('Loss: %.4f' % running_loss)
    with open('running_loss.txt', 'a+') as txt2:
        txt2.write(str(running_loss)+'\n')
    # 统计正常样本比例
    normal_ratio = sum(classification_results) / len(classification_results)
    print('Normal classification ratio: %.4f' % normal_ratio)
    
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    
    # 如果验证集损失更优则保存模型（也可以根据需要添加其他评价指标）
    if args.phase == 'train' and running_loss > args.best_metric:
        args.best_metric = running_loss
        torch.save(net.state_dict(), f'checkpoint_3.9/cfgan_loss_{running_loss}.pth')
        print('best loss model saved!')
    if args.phase == 'train' and normal_ratio > args.best_ratio:
        args.best_ratio = normal_ratio
        torch.save(net.state_dict(), f'checkpoint_3.9/cfgan_ratio_{normal_ratio}.pth')
        print('best ratio model saved!')

if __name__ == '__main__':
    args = parse_args()
    args.best_metric = 1.
    args.best_ratio = 0.
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
    
    # 模型加载与构造
    SAR = ResNet18_1D(num_classes=6, in_channels=12)
    SAR.load_state_dict(torch.load('checkpoint_3.9/best_model.pth', map_location=device))
    SAR = SAR.to(device)
    UNet = UNet1D(in_channels=12, out_channels=12, noise_strength=0.1)
    UNet = UNet.to(device)
    
    cfgan = CFGAN(UNet, SAR)
    cfgan = cfgan.to(device)
    
    # 新增鉴别器实例
    discriminator = Discriminator(in_channels=12)
    discriminator = discriminator.to(device)
    
    # 优化器：生成器只更新 U-Net 部分，鉴别器参数单独更新
    optimizer_g = torch.optim.AdamW(cfgan.unet.parameters(), lr=args.lr)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    
    if args.phase == 'train':
        for epoch in range(args.epochs):
            train(train_loader, cfgan, discriminator, args, criterion, adv_loss_fn, epoch, scheduler, optimizer_g, optimizer_d, device)
            evaluate(val_loader, cfgan, args, criterion, device)
            
            if epoch % 10 == 0:
                torch.save(cfgan.state_dict(), f'models/cfgan_{epoch}.pth')
                print(f'{epoch} model saved!')

