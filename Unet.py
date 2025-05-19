import torch
import torch.nn as nn
import torch.nn.functional as F

# 双卷积模块：连续使用两个 3×1 卷积，保持时序长度不变
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# 高斯噪声注入模块，根据注意力权重矩阵的高亮部位注入噪声，
# 并自适应调整注意力矩阵使其与输入特征图的时序维度匹配
class GaussianNoiseInjector(nn.Module):
    def __init__(self, noise_strength=0.1):
        """
        Args:
            noise_strength: 噪声注入强度因子
        """
        super(GaussianNoiseInjector, self).__init__()
        self.noise_strength = noise_strength

    def forward(self, x, attn):
        """
        Args:
            x: bottleneck 后的特征图，形状 (B, C, L_x)
            attn: 注意力矩阵，形状 (B, L_attn, L_attn) 或 (L_attn, L_attn)
                  其中 L_attn 可能与 x 的时序长度 L_x 不同
        Returns:
            注入高斯噪声后的特征图
        """
        # 如果 attn没有 batch 维度，则扩展
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)  # (1, L_attn, L_attn)
        # 计算每个时刻的权重：沿第1维求均值，得到 (B, L_attn)
        noise_weights = attn.mean(dim=1)  # (B, L_attn)
        # 如果注意力权重的时序长度与 x 的不匹配，则自适应调整
        if noise_weights.shape[-1] != x.shape[-1]:
            # noise_weights: (B, 1, L_attn) -> (B, 1, L_x)
            noise_weights = F.interpolate(noise_weights.unsqueeze(1),
                                          size=x.shape[-1],
                                          mode='linear',
                                          align_corners=True).squeeze(1)
        # 调整形状为 (B, 1, L_x) 便于与 x 相乘
        noise_weights = noise_weights.unsqueeze(1)
        # 生成与 x 同形状的高斯噪声
        noise = torch.randn_like(x)
        # 根据噪声权重注入噪声：注意力权重高的位置注入更多噪声
        x = x + self.noise_strength * noise_weights * noise
        return x

# 标准的 1D U-Net（含跳跃连接），适用于12导联ECG，在bottleneck后注入高斯噪声
class UNet1D(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, features=[64, 128, 256, 512], noise_strength=0.1):
        """
        Args:
            in_channels: 输入信号通道数（12导联 ECG 则为12）
            out_channels: 输出通道数（分割任务时类别数，如二分类时可设为1）
            features: 编码器中各层特征通道数（解码器对称使用）
            noise_strength: 高斯噪声注入强度（在 bottleneck 后根据注意力注入噪声）
        """
        super(UNet1D, self).__init__()
        
        # 编码器部分：逐层双卷积+最大池化，下采样时保存跳跃连接特征
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv1D(prev_channels, feature))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_channels = feature
        
        # Bottleneck 层
        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2)
        # 高斯噪声注入模块，在bottleneck后根据注意力注入噪声
        self.noise_injector = GaussianNoiseInjector(noise_strength=noise_strength)
        
        # 解码器部分：每层先上采样，再与对应跳跃连接拼接，最后双卷积融合
        self.ups = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                DoubleConv1D(feature * 2, feature)
            )
        
        # 输出卷积层，将最后特征映射到目标类别数
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x, attn):
        """
        Args:
            x: 输入信号，形状 (B, in_channels, L)
            attn: 注意力矩阵，形状 (B, L_attn, L_attn) 或 (L_attn, L_attn)
        Returns:
            输出特征图，形状 (B, out_channels, L_out)
        """
        skip_connections = []
        # 编码器：逐层双卷积并记录跳跃连接
        for i, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        # 根据自适应调整后的注意力矩阵在 bottleneck 后注入高斯噪声
        x = self.noise_injector(x, attn)
        
        # 解码器：逐层上采样与跳跃连接拼接，再双卷积融合
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            # 如果上采样后的时序长度与跳跃连接不一致，则通过 padding 补齐
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                x = F.pad(x, (0, diff))
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)
        
        x = self.final_conv(x)
        return x

# 测试模型
if __name__ == '__main__':
    # 假设输入信号长度为1024，12导联 ECG，输出1通道（例如分割或重建任务）
    # 假设 bottleneck 输出的时序长度为 64，对应的注意力矩阵尺寸为 (B, 64, 64)
    model = UNet1D(in_channels=12, out_channels=12, noise_strength=0.1)
    print(model)
    x = torch.randn(1, 12, 5000)
    # 模拟一个注意力矩阵（此处尺寸为 (B, 64, 64)）
    attn = torch.rand(1, 1250, 1250)
    y = model(x, attn)
    print("输出形状：", y.shape)
