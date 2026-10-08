"""
用 RadioML 2016.10a / 2018.01a 数据集训练 SVM 模型。
自动检测格式 (pkl / hdf5)，提取特征后训练。
"""
import numpy as np
import pickle
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import RANDOM_SEED

# RadioML 调制名 → 本系统名 映射 (4类: 4ASK, 2FSK, BPSK, 8PSK)
NAME_MAP = {
    'BPSK': 'BPSK', '8PSK': '8PSK',
    'PAM4': '4ASK',
    'GFSK': '2FSK',
}
# 本系统支持的调制类型
SUPPORTED = {'4ASK', '2FSK', 'BPSK', '8PSK'}


def load_radioml(filepath, min_snr=0, max_samples_per_class=None):
    """加载 RadioML 数据集 (2016 pkl / 2018 hdf5) 并提取特征"""
    print(f'加载: {filepath}')
    from dataset_loader import load_external_signals

    signals, labels, snr_values = load_external_signals(filepath, min_snr=min_snr)

    if labels is None:
        raise ValueError('数据集无调制类型标签，无法训练。请使用带标签的完整数据集。')

    from feature_extraction import extract_features

    X_list, y_list = [], []
    count = 0
    per_class = {}

    for i, (sig, label) in enumerate(zip(signals, labels)):
        mapped = NAME_MAP.get(str(label))
        if mapped is None or mapped not in SUPPORTED:
            continue

        # 限制每类样本数
        if max_samples_per_class:
            cnt = per_class.get(mapped, 0)
            if cnt >= max_samples_per_class:
                continue

        try:
            fv, _ = extract_features(sig)
            X_list.append(fv)
            y_list.append(mapped)
            per_class[mapped] = per_class.get(mapped, 0) + 1
            count += 1
        except Exception:
            pass

    print(f'  有效样本: {count}')
    return np.array(X_list), np.array(y_list)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='test_dataset/dcbd5-main/dcbd5-main/10a/10a/RML2016.10a_dict.pkl')
    parser.add_argument('--max-per-class', type=int, default=2000,
                        help='每类最多样本数(加速训练)')
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # 1. 加载数据 + 提取特征
    print('=' * 60)
    print('  RadioML 数据集训练 SVM')
    print('=' * 60)
    print('\n[1/4] 加载并提取特征...')
    X, y_str = load_radioml(args.dataset, min_snr=0, max_samples_per_class=args.max_per_class)

    # 2. 标签编码
    label_names = sorted(set(y_str))
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    y = np.array([name_to_idx[n] for n in y_str])
    print(f'  类别: {label_names}')
    print(f'  样本: {len(y)}, 特征: {X.shape[1]}')

    # 打印每类样本数
    from collections import Counter
    for mod, cnt in sorted(Counter(y_str).items()):
        print(f'    {mod}: {cnt}')

    # 3. 划分 + 标准化
    print('\n[2/4] 划分训练/测试集...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f'  训练: {len(X_train)}, 测试: {len(X_test)}')

    # 4. GridSearchCV
    print('\n[3/4] SVM 超参数搜索...')
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'C': [1, 10, 100],
        'kernel': ['linear'],
    }
    svm = SVC(probability=False, random_state=RANDOM_SEED)
    gs = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f'  最佳参数: {gs.best_params_}')
    print(f'  交叉验证: {gs.best_score_:.4f}')

    # 5. 训练最终模型
    print('\n[4/4] 训练最终模型...')
    best = CalibratedClassifierCV(gs.best_estimator_, method='sigmoid', cv=3)
    best.fit(X_train, y_train)

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f'\n=== 测试结果 ===')
    print(f'  准确率: {acc:.4f}')
    print(f'  精确率: {prec:.4f}')
    print(f'  召回率: {rec:.4f}')
    print(f'  F1:     {f1:.4f}')

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f'\n  各类别准确率:')
    for i, name in enumerate(label_names):
        if cm[i].sum() > 0:
            print(f'    {name}: {cm[i,i]}/{cm[i].sum()} = {cm[i,i]/cm[i].sum()*100:.1f}%')

    # 6. 保存
    from ai_model import save_model
    save_model(best, scaler, None, label_names, filepath='data/radioml_model.pkl')
    print(f'\nRadioML模型已保存: data/radioml_model.pkl')

    return acc


if __name__ == '__main__':
    main()

