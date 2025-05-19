import torch
import torch.nn as nn

class ECGDiscriminator(nn.Module):
    def __init__(self, in_channels=12, base_channels=64, signal_length=1024):
        """
        Args:
            in_channels: 输入信号通道数, 12导联ECG则为12
            base_channels: 第一层卷积输出的通道数
            signal_length: 输入ECG信号的长度
        """
        super(ECGDiscriminator, self).__init__()
        # 逐层下采样，kernel_size=4, stride=2, padding=1
        self.model = nn.Sequential(
            # 第一层：输入12通道 -> 输出64通道
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 第二层：64 -> 128通道
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 第三层：128 -> 256通道
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 第四层：256 -> 512通道
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 经过4次下采样（每层stride=2）后，信号长度缩小为 signal_length / (2^4)= signal_length/16
        final_length = signal_length // 16
        # 全连接层，将卷积后的特征映射到一个标量
        self.fc = nn.Linear(base_channels * 8 * final_length, 1)
    
    def forward(self, x):
        """
        Args:
            x: 输入ECG信号, 形状 (B, 12, signal_length)
        Returns:
            输出的判别分数, 形状 (B, 1)
        """
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 测试鉴别器
if __name__ == '__main__':
    # 假设输入信号长度为1024，batch_size为8，12导联ECG
    discriminator = ECGDiscriminator(in_channels=12, base_channels=64, signal_length=5000)
    # print(discriminator)
    x = torch.randn(1, 12, 5000)
    output = discriminator(x)
    print("输出形状：", output.shape)


