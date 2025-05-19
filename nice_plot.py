import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 生成模拟数据（12个导联，每个导联1000个数据点）

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
def plot_subplots(x1, x2, main_ax, lead_label):
    """在给定的坐标轴上创建上下两个子图"""
    # 移除原始坐标轴
    main_ax.remove()
    
    # 创建网格布局（2行1列，高度比5:1）
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, 
        subplot_spec=main_ax.get_subplotspec(),
        height_ratios=[5, 1],
        hspace=0.05
    )
    
    # 创建上方信号图
    ax_top = plt.subplot(gs[0])
    ax_top.plot(x1, color='red', linewidth=0.6, label='Original ECG')
    ax_top.plot(x2, color='black', linewidth=0.6, label='CF ECG')
    
    # 添加导联标识（左上角）
    ax_top.text(0.02, 0.95, lead_label, 
               transform=ax_top.transAxes,
               fontsize=9, weight='bold',
               va='top', ha='left')
    
    # 设置信号图参数
    ax_top.tick_params(axis='both', labelsize=7)
    ax_top.set_ylim(-0.2, 1.2)
    ax_top.set_yticks([0, 0.5, 1.0])
    ax_top.set_xticks([])  # 隐藏x轴刻度
    ax_top.legend(loc='upper right', fontsize=3)
    
    # 创建下方热力图
    ax_bottom = plt.subplot(gs[1])
    diff = np.abs(x1 - x2)
    im = ax_bottom.imshow(diff[np.newaxis, :], 
                        aspect='auto', 
                        cmap='jet',
                        interpolation='nearest',
                        vmin=0, vmax=1)
    
    # 设置热力图参数
    ax_bottom.set_yticks([])
    ax_bottom.set_xticks([])
    #ax_bottom.set_xticks(np.linspace(0, x1.shape[-1], 5))
    #ax_bottom.set_xticklabels([0, 250, 500, 750, 1000], fontsize=6)
    ax_bottom.xaxis.set_ticks_position('bottom')

def nice_ecg_plots(x1, x2, save_path):
    labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # 创建画布和主坐标轴
    fig = plt.figure(figsize=(10, 15), dpi=120)
    main_axes = fig.subplots(6, 2, gridspec_kw={'hspace': 0.4, 'wspace': 0.15})
    
    # 按导联类型分列显示
    for row in range(6):
        # 左侧标准导联（I、II、III、aVR、aVL、aVF）
        lead_idx = row
        plot_subplots(x1[lead_idx], x2[lead_idx], 
                     main_axes[row, 0], 
                     labels[lead_idx])
        
        # 右侧胸导联（V1-V6）
        chest_idx = row + 6
        plot_subplots(x1[chest_idx], x2[chest_idx],
                     main_axes[row, 1],
                     labels[chest_idx])

    # 添加全局标签
    # fig.text(0.5, 0.08, 'Sample Points', ha='center', va='center', fontsize=10)
    # fig.text(0.08, 0.5, 'Amplitude', ha='center', va='center', 
    #         rotation='vertical', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

nice_ecg_plots(x1, x1, False)
x1 = (denoise(data[1][:,1000:2000]))
X2 = (denoise(cf_ecg[1][:,1000:2000]))