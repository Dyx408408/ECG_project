import os
import torch
import argparse
import torch.nn as nn
from SRA18 import ResNet18_1D
from Unet import UNet1D
from Discriminator import ECGDiscriminator
from dataset import ECGDataset
from torch.utils.data import DataLoader
from utils import cal_f1s, cal_aucs, split_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
# from pandas import openyxl
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', help='Path to saved model')
    return parser.parse_args()

def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
 
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        # print(data.shape, labels.shape)
        # print(type(data), type(labels))
        output = net(data)
        loss = criterion(output[0], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output[0].data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    return running_loss
    

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output[0], labels)
        running_loss += loss.item()
        output = torch.sigmoid(output[0])
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)

    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.model_path)
    return running_loss
    # else:
    #     aucs = cal_aucs(y_trues, y_scores)
    #     avg_auc = np.mean(aucs)
    #     print('AUCs:', aucs)
    #     print('Avg AUC: %.4f' % avg_auc)

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
    SRA = ResNet18_1D(num_classes=6, in_channels=12)
    SRA = SRA.to(device)
    UNet = UNet1D(in_channels=12, out_channels=12, noise_strength=0.1)
    Discriminator = ECGDiscriminator(in_channels=12, base_channels=64, signal_length=5000)

    optimizer = torch.optim.Adam(SRA.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    trainloss = []
    valloss = []

    if args.phase == 'train':
        if args.resume:
            SRA.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            loss1 = train(train_loader, SRA, args, criterion, epoch, scheduler, optimizer, device)
            trainloss.append(loss1/len(train_dataset))
            loss2 = evaluate(val_loader, SRA, args, criterion, device)
            valloss.append(loss2/len(val_dataset))
        df = pd.DataFrame({
        'Column1': trainloss,
        'Column2': valloss
        })
        df.to_excel('output.output.xlsx',index=False)
    else:
        SRA.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, SRA, args, criterion, device)

    # data
    # x = torch.randn(1, 12, 5000)  # batch_size=8
    # y, attn = SRA(x)
    
    # Expainable_ECG = UNet(x, attn)
    
    # CF_ECG = Expainable_ECG + x
    # print("输出形状：", CF_ECG.shape)
    
    # outputs = Discriminator(Expainable_ECG + x)

    # print("输出形状：", outputs.shape)

    
    





