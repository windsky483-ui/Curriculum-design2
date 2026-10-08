"""
生成外部测试集。
用本系统的信号生成器产生标准的测试数据，保存为 .npy 格式，
可直接用于 GUI "导入外部数据" 功能。
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODULATION_TYPES
from signal_generator import generate_signal
from feature_extraction import extract_features


def generate_test_set(output_path='test_dataset/test_set.npy',
                       snr_levels=None,
                       samples_per_type=500):
    """
    生成标准测试集。

    参数:
      output_path: 输出文件路径
      snr_levels: SNR 列表, 默认 [0, 5, 10, 15, 20, 25, 30]
      samples_per_type: 每种调制每 SNR 的样本数

    输出 .npy 文件包含:
      'signals': 复信号数组 (n_samples, signal_length)
      'labels': 调制类型标签
      'snr': SNR 值
    """
    if snr_levels is None:
        snr_levels = [0, 5, 10, 15, 18]

    print(f'生成测试集: {len(MODULATION_TYPES)} 调制 × {len(snr_levels)} SNR × {samples_per_type} 样本')
    total = len(MODULATION_TYPES) * len(snr_levels) * samples_per_type
    print(f'总计: {total} 个样本')

    signals_list, labels_list, snr_list = [], [], []
    count = 0

    for mod in MODULATION_TYPES:
        print(f'  [{mod}]')
        for snr in snr_levels:
            for _ in range(samples_per_type):
                sig, _, _ = generate_signal(mod, snr_db=snr)
                signals_list.append(sig)
                labels_list.append(mod)
                snr_list.append(snr)
                count += 1
            print(f'    SNR={snr}dB 完成')

    print(f'\n保存到: {output_path}')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 转成统一长度（取最短的）
    min_len = min(len(s) for s in signals_list)
    signals_array = np.array([s[:min_len] for s in signals_list], dtype=complex)

    data = {
        'signals': signals_array,
        'labels': np.array(labels_list),
        'snr': np.array(snr_list),
    }
    np.save(output_path, data, allow_pickle=True)
    print(f'完成! 信号形状: {signals_array.shape}')
    print(f'\n使用方法: GUI → "导入外部数据" → 选择 {output_path}')


def generate_test_set_with_features(output_path='test_dataset/test_features.npy',
                                      snr_levels=None,
                                      samples_per_type=500):
    """
    生成带预提取特征的测试集（更快，直接用于评估）。
    输出: X (特征矩阵), y (标签)
    """
    if snr_levels is None:
        snr_levels = [0, 5, 10, 15, 18]

    print(f'生成测试集(含特征): {len(MODULATION_TYPES)} 调制 × {len(snr_levels)} SNR × {samples_per_type} 样本')

    X_list, y_list = [], []
    for mod in MODULATION_TYPES:
        print(f'  [{mod}]')
        for snr in snr_levels:
            for _ in range(samples_per_type):
                sig, _, _ = generate_signal(mod, snr_db=snr)
                fv, _ = extract_features(sig)
                X_list.append(fv)
                y_list.append(mod)

    X = np.array(X_list)
    data = {'X': X, 'y': np.array(y_list)}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, data, allow_pickle=True)
    print(f'完成! 特征矩阵: {X.shape}, 保存到: {output_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='生成测试数据集')
    parser.add_argument('--samples', type=int, default=500, help='每种调制每SNR的样本数')
    parser.add_argument('--with-features', action='store_true', help='同时生成预提取特征的版本')
    args = parser.parse_args()

    generate_test_set(samples_per_type=args.samples)

    if args.with_features:
        generate_test_set_with_features(samples_per_type=args.samples)

