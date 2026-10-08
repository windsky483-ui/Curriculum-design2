"""
=============================================================================
基于AI的通信信号调制方式识别系统 — 配置文件
=============================================================================
"""
import numpy as np
FS = 100e3                # 采样频率 (Hz)
FC = 10e3                 # 载波频率 (Hz)
RS = 1e3                  # 符号速率 (Baud)
N_SYMBOLS = 1500           # 每个信号生成的符号数（平衡速度与特征质量）
SPB = int(FS / RS)        # 每个符号的采样点数 (自动计算)
MODULATION_TYPES = [
    '4ASK', '2FSK', 'BPSK', '8PSK'
]

# RadioML 2016.10a 数据集路径
DATASET_PATH = 'test_dataset/RML2016.10a-main/RML2016.10a-main/2016.10a'

# 信号生成时的 SNR 范围 (dB) —— 匹配 RadioML 2016.10a 数据集正SNR范围
SNR_RANGE = (0, 18)        # RadioML 2016.10a 正SNR范围 (0~18dB, 2dB步进)
DEFAULT_SNR = 10           # 默认SNR
# 高阶累积量选择
CUMULANT_ORDERS = [20, 21, 40, 41, 42]  # 只用2阶和4阶累积量（公式准确），6阶/8阶的简化公式高SNR下会失真
SAMPLES_PER_CLASS = 2000   # 训练集每类调制样本数 (多SNR下自动均分)
TEST_RATIO = 0.3           # 测试集比例
RANDOM_SEED = 42           # 随机种子
MODEL_CONFIGS = {
    'SVM': {
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 'scale',
    }
}
PER_SNR_MODEL_PATH = 'data/per_snr_models.pkl'
DB_PATH = 'data/modulation_recognition.db'
GUI_TITLE = '基于AI的通信信号调制方式识别系统'
GUI_SIZE = '1400x900'
PLOT_DPI = 100

