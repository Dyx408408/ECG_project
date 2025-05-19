import torch
import torch.nn as nn
import torch.nn.functional as F

# 自注意力层（针对1D特征图），返回输出特征和注意力权重矩阵
class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention1D, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.size()
        proj_query = self.query_conv(x).permute(0, 2, 1)   # (B, L, C//8)
        proj_key = self.key_conv(x)                        # (B, C//8, L)
        energy = torch.bmm(proj_query, proj_key)           # (B, L, L)
        attention = self.softmax(energy)                   # (B, L, L)
        proj_value = self.value_conv(x)                    # (B, C, L)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, L)
        out = self.gamma * out + x
        # 返回经过自注意力变换后的特征图以及注意力权重矩阵
        return out, attention

# 基础残差块（适用于1D卷积）
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ResNet-18 for 1D信号（适用于12导联 ECG），在 layer1 后加入自注意力层
class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=12):
        super(ResNet1D, self).__init__()
        self.in_planes = 64
        # 修改第一个卷积层，使其支持12通道输入
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 第一个 stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 在 layer1 后加入自注意力层，返回特征和注意力权重
        self.self_attention = SelfAttention1D(64 * block.expansion)
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入 x 的形状应为 (batch_size, 12, signal_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 此处获得经过自注意力层后的特征以及注意力权重矩阵
        x, attn = self.self_attention(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)  # 输出形状 (B, C, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # 返回分类输出和自注意力权重矩阵
        return x, attn

def ResNet18_1D(num_classes, in_channels=12):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

# 测试模型
if __name__ == '__main__':
    # 假设分类任务类别数为10，输入通道为12，信号长度为5000
    model = ResNet18_1D(num_classes=9, in_channels=12)
    # print(model)
    x = torch.randn(8, 12, 5000)  # batch_size=8
    y, attn = model(x)
    print("分类输出形状：", y.shape)
    print("注意力权重矩阵形状：", attn.shape)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np

# # 自注意力层（针对1D特征图），返回输出特征和注意力权重矩阵
# class SelfAttention1D(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention1D, self).__init__()
#         self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
#         self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
#         self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, x):
#         # x: (B, C, L)
#         B, C, L = x.size()
#         proj_query = self.query_conv(x).permute(0, 2, 1)   # (B, L, C//8)
#         proj_key = self.key_conv(x)                        # (B, C//8, L)
#         energy = torch.bmm(proj_query, proj_key)           # (B, L, L)
#         attention = self.softmax(energy)                   # (B, L, L)
#         proj_value = self.value_conv(x)                    # (B, C, L)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, L)
#         out = self.gamma * out + x
#         # 返回经过自注意力变换后的特征图以及注意力权重矩阵
#         return out, attention

# # 基础残差块（适用于1D卷积）
# class BasicBlock1D(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super(BasicBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out

# # ResNet-18 for 1D信号（适用于12导联 ECG），在 layer1 后加入自注意力层
# class ResNet1D(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, in_channels=12):
#         super(ResNet1D, self).__init__()
#         self.in_planes = 64
#         # 修改第一个卷积层，使其支持12通道输入
#         self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
#         # 第一个 stage
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         # 在 layer1 后加入自注意力层，返回特征和注意力权重
#         self.self_attention = SelfAttention1D(64 * block.expansion)
        
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
        
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion),
#             )
#         layers = []
#         layers.append(block(self.in_planes, planes, stride, downsample))
#         self.in_planes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_planes, planes))
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         # 输入 x 的形状应为 (batch_size, 12, signal_length)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         # 此处获得经过自注意力层后的特征以及注意力权重矩阵
#         x, attn = self.self_attention(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)  # 输出形状 (B, C, 1)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         # 返回分类输出和自注意力权重矩阵
#         return x, attn

# def ResNet18_1D(num_classes, in_channels=12):
#     return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

# # 测试模型
# if __name__ == '__main__':
#     # 假设分类任务类别数为10，输入通道为12，信号长度为5000
#     model = ResNet18_1D(num_classes=10, in_channels=12)
#     x = torch.randn(8, 12, 5000)  # batch_size=8
#     y, attn = model(x)
#     print("分类输出形状：", y.shape)
#     print("注意力权重矩阵形状：", attn.shape)
    
#     # -------------------------- 新增部分 --------------------------
#     # 为了计算不同疾病（类别）与不同导联之间的相关性，这里采用每个导联的平均信号值
#     # 与经过 softmax 后的预测概率进行相关性计算（Pearson相关系数）
    
#     # 计算每个导联的平均信号值，结果形状为 (batch_size, 12)
#     x_mean = x.mean(dim=2)
#     # 将模型输出转换为概率，形状为 (batch_size, 10)
#     prob = F.softmax(y, dim=1)
    
#     num_diseases = prob.size(1)
#     num_leads = x_mean.size(1)
#     corr_matrix = torch.zeros(num_diseases, num_leads)
    
#     # 逐类别、逐导联计算 Pearson 相关系数
#     for disease in range(num_diseases):
#         for lead in range(num_leads):
#             # 获取所有样本中第 lead 个导联的平均值，与第 disease 类别的预测概率
#             lead_vals = x_mean[:, lead]
#             disease_probs = prob[:, disease]
#             # 计算均值
#             lead_mean = lead_vals.mean()
#             disease_mean = disease_probs.mean()
#             # 计算协方差
#             cov = ((lead_vals - lead_mean) * (disease_probs - disease_mean)).mean()
#             # 计算标准差，加上一个小的常数以防除零
#             std_lead = lead_vals.std()
#             std_disease = disease_probs.std()
#             corr = cov / (std_lead * std_disease + 1e-8)
#             corr_matrix[disease, lead] = corr
    
#     # 将相关性矩阵转换为 numpy 数组以便于绘图
#     corr_np = corr_matrix.detach().cpu().numpy()
    
#     # 绘制热力图
#     plt.figure(figsize=(8, 6))
#     plt.imshow(corr_np, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.xlabel('导联')
#     plt.ylabel('疾病类别')
#     plt.title('疾病与导联之间的相关性热力图')
#     plt.xticks(np.arange(num_leads), labels=[f'导联 {i+1}' for i in range(num_leads)])
#     plt.yticks(np.arange(num_diseases), labels=[f'疾病 {i+1}' for i in range(num_diseases)])
#     plt.show()
#     # -------------------------- 新增部分结束 --------------------------
