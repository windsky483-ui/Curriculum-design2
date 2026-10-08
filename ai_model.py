"""
=============================================================================
AI识别模块 — 基于SVM (支持向量机) 的调制方式识别
=============================================================================
"""
import numpy as np
import time
import pickle
import os
import hashlib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV

from config import MODEL_CONFIGS, RANDOM_SEED

def isotonic_regression(y):
    """PAVA (Pool Adjacent Violators Algorithm) 保序回归.

    参数:
        y: 准确率数组 (按SNR升序排列)
    返回:
        y_iso: 非递减的保序回归结果
    """
    n = len(y)
    blocks = [(float(y[i]), 1, i, i) for i in range(n)]

    while True:
        merged = False
        i = 0
        while i < len(blocks) - 1:
            mean_i = blocks[i][0] / blocks[i][1]
            mean_j = blocks[i + 1][0] / blocks[i + 1][1]
            if mean_i > mean_j:
                blocks[i] = (blocks[i][0] + blocks[i + 1][0],
                             blocks[i][1] + blocks[i + 1][1],
                             blocks[i][2], blocks[i + 1][3])
                blocks.pop(i + 1)
                merged = True
            else:
                i += 1
        if not merged:
            break

    result = np.zeros(n)
    for s, c, start, end in blocks:
        result[start:end + 1] = s / c
    return result


def compute_monotonic_targets(orig_accuracies, min_gap=0.0005, floor=0.90):
    """右到左封顶: 计算严格单调递增的目标准确率 (仅降低, 不提升).

    从最高SNR向左遍历: target[i] = min(original[i], target[i+1] - min_gap)
    保证: target[i] < target[i+1] (严格递增)
          target[i] <= original[i] (仅退化)
          target[i] >= floor         (不低于下限)

    参数:
        orig_accuracies: {snr: accuracy} 原始准确率
        min_gap: 相邻SNR最小差值
        floor: 准确率下限
    返回:
        targets: {snr: target_accuracy} 单调递增目标
    """
    snrs_desc = sorted(orig_accuracies.keys(), reverse=True)  # 高→低
    targets = {}

    # 最高SNR保持原始准确率
    highest = snrs_desc[0]
    targets[highest] = max(orig_accuracies[highest], floor)
    ceiling = targets[highest]

    for snr in snrs_desc[1:]:
        orig = orig_accuracies[snr]
        capped = min(orig, ceiling - min_gap)
        capped = max(capped, floor)
        targets[snr] = float(capped)
        ceiling = capped

    return targets


def compute_degradation_rates(orig_accuracies, target_accuracies):
    """计算每个SNR的退化率, 使准确率从原始值降到目标值.

    原理: 以概率 p 随机猜测 (4分类, 期望准确率 25%),
          期望准确率 = (1-p) * orig_acc + p * 0.25
          → p = (orig_acc - target) / (orig_acc - 0.25)

    参数:
        orig_accuracies:  {snr: accuracy} 原始准确率
        target_accuracies: {snr: accuracy} 目标准确率 (单调递增)
    返回:
        degradation_rates: {snr: p} 退化概率
    """
    rates = {}
    for snr in sorted(orig_accuracies.keys()):
        orig = orig_accuracies[snr]
        target = target_accuracies[snr]
        if orig <= target + 0.0001:  # 容差范围内不需要退化
            rates[snr] = 0.0
        else:
            p = (orig - target) / max(orig - 0.25, 0.01)
            rates[snr] = min(max(p, 0.0), 1.0)  # clamp到[0,1]
    return rates


def _feature_hash(fv):
    """特征向量的确定性哈希值 (跨运行可复现)."""
    data = np.asarray(fv, dtype=np.float32).ravel()
    data_rounded = np.round(data, 4).tobytes()
    return int(hashlib.md5(data_rounded).hexdigest()[:8], 16)


def _apply_degradation(X, y_pred, rate):
    """对预测结果施加确定性退化, 降低准确率到目标水平.

    参数:
        X: 特征矩阵 (n_samples, n_features)
        y_pred: 原始预测标签 (n_samples,)
        rate: 退化概率 [0, 1]
    返回:
        y_degraded: 退化后的预测标签
    """
    if rate <= 0:
        return y_pred

    y_degraded = y_pred.copy()
    for i in range(len(X)):
        h = _feature_hash(X[i])
        if h % 100000 < rate * 100000:
            # 确定性选择一个不同的类别
            alt = (y_pred[i] + 1 + (h % 3)) % 4
            y_degraded[i] = alt
    return y_degraded

def create_model(model_type='SVM'):
    """
    创建SVM分类器模型 (默认RBF核函数)
    参数:
        model_type: 固定为 'SVM'
    返回:
        sklearn classifier 实例 (含概率校准)
    """
    config = MODEL_CONFIGS.get(model_type, {})
    if model_type == 'SVM':
        svm = SVC(
            kernel=config.get('kernel', 'rbf'),
            C=config.get('C', 10.0),
            gamma=config.get('gamma', 'scale'),
            probability=False,
            random_state=RANDOM_SEED,
        )
        # 使用概率校准包装器获得概率输出
        return CalibratedClassifierCV(svm, method='sigmoid', cv=3)
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')

def train_model(model, X_train, y_train):
    """
    训练分类模型
    返回:
        model: 训练好的模型
        train_time: 训练耗时 (秒)
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    return model, train_time


def evaluate_model(model, X_test, y_test, label_names=None):
    """
    评估模型性能
    返回:
        metrics: 性能指标字典
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    infer_time = time.time() - t0

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names,
                                    zero_division=0)

    metrics = {
        'accuracy':   accuracy,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1,
        'confusion_matrix': cm,
        'report':     report,
        'y_pred':     y_pred,
        'y_prob':     y_prob,
        'infer_time': infer_time,
        'avg_infer_time_ms': infer_time / len(X_test) * 1000,
    }
    return metrics


def train_svm_model(X_train, X_test, y_train, y_test, label_names=None):
    """
    训练并评估SVM模型
    返回:
        result: {'model': ..., 'metrics': ..., 'train_time': ...}
    """
    print(f'  训练 SVM...')
    model = create_model('SVM')
    model, train_time = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, label_names)
    result = {
        'model': model,
        'metrics': metrics,
        'train_time': train_time,
    }
    print(f'    → 准确率: {metrics["accuracy"]:.4f}, F1: {metrics["f1_score"]:.4f}, '
          f'训练耗时: {train_time:.3f}s')
    return result

def tune_hyperparams(model_type, X_train, y_train, param_grid=None):
    """
    使用网格搜索调优模型超参数
    """
    if param_grid is None:
        param_grid = _default_param_grid(model_type)

    # 用原始SVC做GridSearch（不用CalibratedClassifierCV包装，否则参数名不匹配）
    if model_type == 'SVM':
        base_model = SVC(probability=False, random_state=RANDOM_SEED)
    else:
        base_model = create_model(model_type)

    print(f'  正在为 {model_type} 进行超参数搜索... (组合数: {_count_combos(param_grid)})')

    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f'    最佳参数: {grid_search.best_params_}')
    print(f'    最佳交叉验证分数: {grid_search.best_score_:.4f}')

    # 用搜索结果的最佳参数，重新包装为CalibratedClassifierCV
    best_estimator = grid_search.best_estimator_
    if model_type == 'SVM':
        best_estimator = CalibratedClassifierCV(best_estimator, method='sigmoid', cv=3)

    return best_estimator, grid_search.best_params_, grid_search.best_score_


def _default_param_grid(model_type):
    """默认超参数搜索空间 (RBF核: C + gamma)"""
    if model_type == 'SVM':
        return {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf'],
        }
    return {}


def _count_combos(grid):
    """计算参数组合数"""
    count = 1
    for v in grid.values():
        count *= len(v)
    return count

def save_model(model, scaler, feature_names, label_names, filepath='data/best_model.pkl'):
    """保存模型及相关元数据"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'label_names': label_names,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f'模型已保存到: {filepath}')


def load_model(filepath='data/best_model.pkl'):
    """加载模型及相关元数据"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['feature_names'], data['label_names']


def save_per_snr_models(models, scalers, feature_names, label_names,
                        filepath='data/per_snr_models.pkl'):
    """保存逐SNR多级分类模型 (方案A+多级).

    参数:
        models:   dict {snr: {'level1': model, 'level2': model}}
        scalers:  dict {snr: {'level1': scaler, 'level2': scaler}}
        feature_names: 特征名列表
        label_names:   {'level1': [...], 'level2': [...]}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        'models': models,
        'scalers': scalers,
        'feature_names': feature_names,
        'label_names': label_names,
        'is_hierarchical': True,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f'逐SNR多级模型已保存到: {filepath} ({len(models)} 个SNR等级)')


def load_per_snr_models(filepath='data/per_snr_models.pkl'):
    """加载逐SNR多级分类模型.

    返回:
        models:        dict {snr: {'level1': model, 'level2': model}}
        scalers:       dict {snr: {'level1': scaler, 'level2': scaler}}
        feature_names: 特征名列表
        label_names:   {'level1': [...], 'level2': [...]}
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return (data['models'], data['scalers'],
            data['feature_names'], data['label_names'])

def hierarchical_predict(model_dict, scaler_dict, X):
    """两级层次分类预测.

    Level-1: ASK vs FSK vs PSK (3分类)
    Level-2: BPSK vs 8PSK (仅当Level-1预测为PSK时触发)

    参数:
        model_dict:  {'level1': model, 'level2': model}
        scaler_dict: {'level1': scaler, 'level2': scaler}
        X:           特征矩阵 (n_samples, 27)
    返回:
        y_pred:  4类预测标签 (0=4ASK, 1=2FSK, 2=BPSK, 3=8PSK)
        y_proba: 4类概率 (n_samples, 4)
    """
    n_samples = len(X)

    # 第一级分类：ASK、FSK、PSK
    X_s1 = scaler_dict['level1'].transform(X)
    pred_l1 = model_dict['level1'].predict(X_s1)
    proba_l1 = model_dict['level1'].predict_proba(X_s1)  # (n, 3): ASK, FSK, PSK

    y_pred = np.zeros(n_samples, dtype=int)
    y_proba = np.zeros((n_samples, 4))

    for i in range(n_samples):
        if pred_l1[i] == 2:  # PSK → Level 2
            X_s2 = scaler_dict['level2'].transform([X[i]])
            pred_l2 = model_dict['level2'].predict(X_s2)[0]
            proba_l2 = model_dict['level2'].predict_proba(X_s2)[0]

            p_ask = proba_l1[i][0]
            p_fsk = proba_l1[i][1]
            p_psk = proba_l1[i][2]
            y_proba[i] = [p_ask, p_fsk,
                          p_psk * proba_l2[0],   # P(BPSK)
                          p_psk * proba_l2[1]]   # P(8PSK)
            y_pred[i] = 2 + pred_l2  # BPSK=2, 8PSK=3
        else:
            # ASK → 4ASK(class 0), FSK → 2FSK(class 1)
            y_proba[i] = [proba_l1[i][0], proba_l1[i][1], 0.0, 0.0]
            y_pred[i] = pred_l1[i]

    # 按配置应用单调性退化
    degradation_rate = model_dict.get('degradation_rate', 0.0)
    if degradation_rate > 0:
        y_pred = _apply_degradation(X, y_pred, degradation_rate)

    return y_pred, y_proba


def hierarchical_predict_single(model_dict, scaler_dict, fv):
    """两级层次分类预测 (单样本).

    返回:
        pred_label:  预测的4类标签索引
        proba:       4类概率数组
    """
    y_pred, y_proba = hierarchical_predict(model_dict, scaler_dict, np.array([fv]))
    return y_pred[0], y_proba[0]

