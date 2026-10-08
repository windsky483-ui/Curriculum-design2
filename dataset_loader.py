"""
外部数据集加载模块。
支持格式: .pkl (RadioML) / .npy / .mat / .h5
"""
import numpy as np
import os
import pickle
from feature_extraction import extract_features

# RadioML 调制名 → 本系统调制名 映射 (4类: 4ASK, 2FSK, BPSK, 8PSK)
RADIOML_NAME_MAP = {
    'BPSK': 'BPSK', '8PSK': '8PSK',
    'PAM4': '4ASK',
    'GFSK': '2FSK',
}
SUPPORTED_MODS = {'4ASK', '2FSK', 'BPSK', '8PSK'}


def load_external_signals(filepath, min_snr=None):
    """
    加载外部信号文件。自动检测格式:
      .pkl: RadioML 2016.10a dict {(mod,snr): ndarray}
      .h5/.hdf5: RadioML 2018 格式
      .npy: 通用格式
      .mat: MATLAB 格式
    返回: (signals, labels, snr_values)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pkl':
        return _load_radioml_pkl(filepath, min_snr=min_snr)
    elif ext in ['.h5', '.hdf5']:
        return _load_radioml_h5(filepath, min_snr=min_snr)
    elif ext == '.npy':
        data = np.load(filepath, allow_pickle=True)
        return _parse_generic(data)
    elif ext == '.mat':
        from scipy.io import loadmat
        data = loadmat(filepath)
        return _parse_generic(data)
    else:
        raise ValueError(f'不支持的文件格式: {ext}')


def _load_radioml_h5(filepath, min_snr=None):
    """
    加载 RadioML 2018 HDF5 格式。
    原始格式: X(2555904,1024) IQ交错, Y(调制标签), Z(SNR标签)
    简化格式: dataset(N,1024) IQ交错, 无标签
    """
    import h5py
    signals_list, labels_list, snr_list = [], [], []

    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())

        # 格式1: 完整格式 X/Y/Z
        if 'X' in keys and 'Y' in keys and 'Z' in keys:
            X = f['X'][:]
            Y = f['Y'][:]
            Z = f['Z'][:]

            # Y 是 one-hot 编码，解码为调制名称
            if Y.ndim == 2:
                # 标准 RadioML 2018 调制列表
                mod_names_2018 = [
                    'OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
                    '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM',
                    '64QAM','128QAM','256QAM','AM-SSB-WC','AM-SSB-SC',
                    'AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK'
                ]
                y_labels = np.argmax(Y, axis=1)
                labels_list = [mod_names_2018[i] for i in y_labels]
            else:
                labels_list = [str(y) for y in Y]

            snr_list = [float(z) for z in Z]

            # X: (n,1024) IQ交错 → 复信号
            for i in range(len(X)):
                row = X[i]
                signals_list.append(row[0::2] + 1j * row[1::2])

        # 格式2: 简化格式, 只有 dataset
        elif 'dataset' in keys:
            ds = f['dataset'][:]
            for i in range(len(ds)):
                row = ds[i]
                signals_list.append(row[0::2] + 1j * row[1::2])
            # 无标签, 用于推理
            labels_list = None
            snr_list = None

        else:
            # 取第一个数据集
            key = keys[0]
            ds = f[key][:]
            for i in range(len(ds)):
                row = ds[i]
                signals_list.append(row[0::2] + 1j * row[1::2])
            labels_list = None
            snr_list = None

    # 过滤 SNR
    if min_snr is not None and snr_list is not None:
        valid = [i for i, s in enumerate(snr_list) if s >= min_snr]
        signals_list = [signals_list[i] for i in valid]
        if labels_list:
            labels_list = [labels_list[i] for i in valid]
        snr_list = [snr_list[i] for i in valid]

    return signals_list, labels_list, snr_list


def _load_radioml_pkl(filepath, min_snr=None):
    """
    加载 RadioML 2016.10a/2018.01a 格式的 pickle 数据集。
    格式: dict, keys=(mod_name, snr) tuples, values=ndarray (n_samples, 2, signal_len)

    参数:
      min_snr: 最低 SNR 阈值(dB), 默认 0 (过滤负 SNR)
    """
    if min_snr is None:
        min_snr = 0

    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    signals_list, labels_list, snr_list = [], [], []
    skipped_mods = set()
    skipped_snr = 0

    for (mod_name, snr), iq_array in data.items():
        # 过滤不支持的调制类型
        mapped = RADIOML_NAME_MAP.get(mod_name, None)
        if mapped is None or mapped not in SUPPORTED_MODS:
            skipped_mods.add(mod_name)
            continue

        # 过滤负 SNR
        if snr < min_snr:
            skipped_snr += iq_array.shape[0]
            continue

        # RadioML 格式: (n, 2, L) → I = [:, 0, :], Q = [:, 1, :]
        n_samples = iq_array.shape[0]
        for i in range(n_samples):
            iq = iq_array[i, 0, :] + 1j * iq_array[i, 1, :]
            signals_list.append(iq)
            labels_list.append(mapped)
            snr_list.append(snr)

    if skipped_mods:
        print(f'  跳过的调制类型 (本系统不支持): {sorted(skipped_mods)}')
    if skipped_snr > 0:
        print(f'  跳过的负SNR样本: {skipped_snr} 个')

    return signals_list, labels_list, snr_list


def _parse_generic(data):
    """解析通用格式 (npy/mat/h5)"""
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if not k.startswith('__')}

    if isinstance(data, dict):
        if 'signals' in data and 'labels' in data:
            sigs, labs = data['signals'], data['labels']
        elif 'X' in data and 'y' in data:
            sigs, labs = data['X'], data['y']
        elif 'X' in data and 'Y' in data:
            sigs, labs = data['X'], data['Y']
        else:
            keys = list(data.keys())
            sigs, labs = data[keys[0]], None
    elif isinstance(data, np.ndarray):
        sigs, labs = data, None
    else:
        raise ValueError('无法解析的数据格式')

    sigs = np.array(sigs)
    signals_list = _iq_to_complex(sigs)
    labels_list = [_decode_label(l) for l in labs] if labs is not None else None
    return signals_list, labels_list, None


def _iq_to_complex(sigs):
    """IQ 数组 → 复信号列表"""
    if sigs.ndim == 3:
        # (n, 2, L) → I + jQ
        return [s[0] + 1j * s[1] for s in sigs]
    elif sigs.ndim == 2:
        if sigs.shape[1] == 2:
            return [s[0] + 1j * s[1] for s in sigs]
        return [s for s in sigs]
    return [sigs]


def _decode_label(label):
    if isinstance(label, bytes):
        return label.decode('utf-8', errors='replace')
    if isinstance(label, np.ndarray):
        return str(label.item())
    return str(label)


def extract_features_from_dataset(signals, labels=None, show_progress=True):
    """
    从信号列表提取特征矩阵。
    返回: X (特征矩阵), y (标签列表), valid_mask
    """
    X_list, y_list, valid_mask = [], [], []

    for i, sig in enumerate(signals):
        try:
            fv, _ = extract_features(sig)
            X_list.append(fv)
            if labels is not None:
                y_list.append(labels[i])
            valid_mask.append(True)
        except Exception:
            valid_mask.append(False)
            continue

        if show_progress and (i + 1) % 200 == 0:
            print(f'  已处理 {i + 1}/{len(signals)} 样本...')

    X = np.array(X_list)
    if labels is not None:
        y = np.array(y_list)
    else:
        y = None

    return X, y, valid_mask


def test_dataset_accuracy(filepath, model, scaler, label_names, min_snr=0):
    """
    完整流程: 加载外部数据集 → 提取特征 → 预测 → 计算准确率。
    支持扁平模型和层级式多级分类模型。
    返回: dict with accuracy stats
    """
    # 检测层级式模型
    is_hierarchical = isinstance(label_names, dict)

    print(f'加载数据集: {filepath}')
    signals, labels, snr_values = load_external_signals(filepath, min_snr=min_snr)
    print(f'  有效信号: {len(signals)}, 标签: {len(labels) if labels else 0}')

    print('提取特征...')
    X, y_true_names, valid = extract_features_from_dataset(signals, labels)
    n_valid = sum(valid)
    print(f'  成功提取: {n_valid}/{len(signals)}')

    if y_true_names is not None:
        y_true_names = y_true_names[valid]

    print('模型预测...')
    if is_hierarchical:
        from ai_model import hierarchical_predict
        preds, probs = hierarchical_predict(model, scaler, X)
        confs = np.max(probs, axis=1)
        flat_labels = ['4ASK', '2FSK', 'BPSK', '8PSK']
    else:
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)
            confs = np.max(probs, axis=1)
        else:
            confs = np.ones(len(preds))
        flat_labels = label_names

    pred_names = [flat_labels[p] if flat_labels else str(p) for p in preds]

    # 计算准确率（只统计本系统支持的调制类型）
    name_to_idx = {n: i for i, n in enumerate(flat_labels)} if flat_labels else {}
    correct = 0
    total = 0
    per_mod = {}

    for i, (pred_name, true_name) in enumerate(zip(pred_names, y_true_names)):
        true_str = str(true_name).strip()
        if true_str not in name_to_idx:
            continue
        total += 1
        is_correct = (pred_name == true_str)
        if is_correct:
            correct += 1

        if true_str not in per_mod:
            per_mod[true_str] = {'correct': 0, 'total': 0}
        per_mod[true_str]['total'] += 1
        if is_correct:
            per_mod[true_str]['correct'] += 1

    results = {
        'total_signals': len(signals),
        'valid_features': n_valid,
        'tested': total,
        'correct': correct,
        'accuracy': correct / max(total, 1) * 100,
        'per_mod': per_mod,
        'pred_names': pred_names,
        'confs': confs,
    }

    print(f'\n=== 外部数据集测试结果 ===')
    print(f'  总信号数: {len(signals)}')
    print(f'  有效特征: {n_valid}')
    print(f'  可测试样本: {total}')
    print(f'  正确识别: {correct}')
    print(f'  准确率: {results["accuracy"]:.2f}%')
    if per_mod:
        print(f'  各类别准确率:')
        for mod in sorted(per_mod.keys()):
            d = per_mod[mod]
            print(f'    {mod}: {d["correct"]}/{d["total"]} = {d["correct"]/max(d["total"],1)*100:.1f}%')

    return results

