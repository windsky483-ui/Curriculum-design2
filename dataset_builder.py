"""
=============================================================================
数据集构建模块 — 批量生成信号并提取特征，构建训练/测试数据集
=============================================================================
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

from config import (
    MODULATION_TYPES, SAMPLES_PER_CLASS, TEST_RATIO, RANDOM_SEED,
    SNR_RANGE, DEFAULT_SNR
)
from signal_processor import _get_processor
from feature_extraction import extract_features


def build_dataset(
    mod_types=None,
    samples_per_class=SAMPLES_PER_CLASS,
    snr_db=DEFAULT_SNR,
    random_state=RANDOM_SEED,
    show_progress=True
):
    """
    构建特征数据集 (无放回唯一采样 — 消除训练/测试集信号重叠)

    每个唯一信号在数据集中最多出现一次，确保后续 train_test_split
    不会将同一信号的不同副本同时放入训练集和测试集。

    参数:
        mod_types:         调制类型列表，默认全部
        samples_per_class: 每类样本数 (超过可用数量时自动截断)
        snr_db:            信噪比 (dB)
        random_state:      随机种子 (控制哪些唯一信号被选中)
        show_progress:     是否显示进度
    返回:
        X: 特征矩阵 (n_samples, 27)
        y: 标签 (整数)
        feature_names: 特征名称列表
        label_names:   标签名称列表
    """
    if mod_types is None:
        mod_types = MODULATION_TYPES

    sp = _get_processor()
    rng = np.random.RandomState(random_state)

    X_list, y_list = [], []
    feature_names = None

    for label_idx, mod_type in enumerate(mod_types):
        # 获取该(调制, SNR)下的全部唯一信号
        all_sigs = sp.get_all_signals(mod_type, snr_db)
        n_available = len(all_sigs)
        n_take = min(samples_per_class, n_available)

        if show_progress and samples_per_class > n_available:
            print(f'  ⚠ [{mod_type}] 请求{samples_per_class}样本, 仅{n_available}个可用 → 取全部{n_available}个')

        if show_progress:
            print(f'  [{mod_type}] 唯一采样 {n_take}/{n_available} 个信号...')

        # 无放回随机选取 n_take 个唯一信号
        indices = rng.permutation(n_available)[:n_take]

        for idx in indices:
            sig = all_sigs[idx]
            fv, fn = extract_features(sig)

            if feature_names is None:
                feature_names = fn

            X_list.append(fv)
            y_list.append(label_idx)

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)
    label_names = mod_types

    if show_progress:
        print(f'  数据集构建完成: {X.shape[0]} 样本 (全部唯一), '
              f'{X.shape[1]} 特征, {len(mod_types)} 类别')

    return X, y, feature_names, label_names


def build_multi_snr_dataset(
    mod_types=None,
    samples_per_class=SAMPLES_PER_CLASS,
    snr_list=None,
    random_state=RANDOM_SEED,
    show_progress=True
):
    """
    构建多SNR条件下的数据集 (用于鲁棒性测试)
    参数:
        snr_list: SNR列表，如 [0, 5, 10, 15, 20, 25, 30]
    """
    if snr_list is None:
        snr_list = list(range(0, 31, 5))

    X_list, y_list = [], []
    feature_names = None

    for label_idx, mod_type in enumerate(mod_types or MODULATION_TYPES):
        if show_progress:
            print(f'  生成 [{mod_type}] 多SNR样本...')

        for snr in snr_list:
            n = samples_per_class // len(snr_list)
            for _ in range(n):
                sig, _, _ = get_signal(mod_type, snr_db=snr)
                fv, fn = extract_features(sig)
                if feature_names is None:
                    feature_names = fn
                X_list.append(fv)
                y_list.append(label_idx)

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, feature_names, mod_types or MODULATION_TYPES


def split_and_normalize(X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED):
    """
    划分训练/测试集并标准化
    返回:
        X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def save_dataset(X_train, X_test, y_train, y_test, feature_names, label_names,
                 scaler, filepath='data/dataset.pkl'):
    """保存数据集到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'label_names': label_names,
        'scaler': scaler,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f'数据集已保存到: {filepath}')


def load_dataset(filepath='data/dataset.pkl'):
    """从文件加载数据集"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# 第1级: 大类分组 (ASK / FSK / PSK)
MOD_GROUP = {
    '4ASK': 'ASK',
    '2FSK': 'FSK',
    'BPSK': 'PSK',
    '8PSK': 'PSK',
}
LEVEL1_GROUPS = ['ASK', 'FSK', 'PSK']

# 第2级: PSK内部细分
LEVEL2_MODS = ['BPSK', '8PSK']


def map_to_group_labels(y_4class, label_names_4class):
    """将4类标签映射为3组标签 (Level-1).

    参数:
        y_4class: 原始4类整数标签
        label_names_4class: 4类名称列表 (e.g. ['4ASK','2FSK','BPSK','8PSK'])
    返回:
        y_group: 3组整数标签 (0=ASK, 1=FSK, 2=PSK)
        group_names: ['ASK', 'FSK', 'PSK']
    """
    label_to_group = {}
    for i, name in enumerate(label_names_4class):
        label_to_group[i] = LEVEL1_GROUPS.index(MOD_GROUP[name])
    y_group = np.array([label_to_group[label] for label in y_4class], dtype=int)
    return y_group, LEVEL1_GROUPS


def filter_psk_samples(X, y, label_names):
    """从全量数据中筛选PSK类样本 (BPSK + 8PSK), 用于Level-2训练.

    返回:
        X_psk, y_psk: PSK-only 数据
        psk_labels: ['BPSK', '8PSK']
    """
    psk_indices = []
    psk_label_map = {}  # 原始4类标签 → Level-2标签 (0=BPSK, 1=8PSK)
    for i, name in enumerate(label_names):
        if name in LEVEL2_MODS:
            psk_label_map[i] = LEVEL2_MODS.index(name)

    mask = np.isin(y, list(psk_label_map.keys()))
    X_psk = X[mask]
    y_psk = np.array([psk_label_map[label] for label in y[mask]], dtype=int)
    return X_psk, y_psk, LEVEL2_MODS

