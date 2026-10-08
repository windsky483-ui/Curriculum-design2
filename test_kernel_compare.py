"""
对比测试：线性核 vs RBF核，找出受RBF影响最小的4种调制类型组合。
测试所有合理4类组合，输出每种组合在线性核下的准确率。
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from signal_processor import SignalProcessor
from feature_extraction import extract_features

# 候选调制类型（RadioML 2016.10a 支持的原始类型）
ALL_CANDIDATES = {
    'BPSK', 'QPSK', '8PSK',     # PSK
    'PAM4',                       # → 4ASK
    'GFSK', 'CPFSK',             # GFSK→2FSK (CPFSK已移除)
    'QAM16', 'QAM64',            # QAM
}

# 4ASK = PAM4, 2FSK = GFSK (仅GFSK，CPFSK已移除)
# 只用原始数据集中的单一类型测试，避免合并带来的混淆

# 实际可选的"纯"类型（不合并）：
# BPSK, QPSK, 8PSK, PAM4(4ASK), GFSK, CPFSK, QAM16, QAM64
# 从这8种选出4种，看哪种组合线性核表现最好

CANDIDATES = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'GFSK', 'CPFSK', 'QAM16', 'QAM64']
NAME_CN = {
    'BPSK': 'BPSK', 'QPSK': 'QPSK', '8PSK': '8PSK',
    'PAM4': '4ASK', 'GFSK': '2FSK(GFSK)', 'CPFSK': '2FSK(CPFSK)',
    'QAM16': '16QAM', 'QAM64': '64QAM'
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_samples(mod_name, max_per_snr=200, min_snr=0):
    """从 RadioML 2016.10a 加载指定调制类型的样本"""
    sp = SignalProcessor()
    # 直接访问原始数据
    import pickle
    base = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base, 'test_dataset',
                            'dcbd5-main/dcbd5-main/10a/10a/RML2016.10a_dict.pkl')
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    samples = []
    for (m, snr), iq_array in raw.items():
        if m != mod_name:
            continue
        if snr < min_snr:
            continue
        n = min(iq_array.shape[0], max_per_snr)
        for i in range(n):
            sig = iq_array[i, 0, :] + 1j * iq_array[i, 1, :]
            power = np.mean(np.abs(sig) ** 2)
            if power > 1e-10:
                sig = sig / np.sqrt(power)
            sig = sig - np.mean(sig)
            samples.append((sig, snr))
    return samples


def extract_all(samples):
    """批量提取特征"""
    X, y = [], []
    for sig, _ in samples:
        try:
            fv, _ = extract_features(sig)
            X.append(fv)
        except Exception:
            pass
    return np.array(X)


def test_combination(mod_list, samples_per_mod=800):
    """测试给定的4种调制组合，在线性核和RBF核下的准确率"""
    # 加载数据
    X_list, y_list = [], []
    for label_idx, mod in enumerate(mod_list):
        all_samps = load_samples(mod, max_per_snr=200)
        # 均匀采样
        if len(all_samps) > samples_per_mod:
            indices = np.random.choice(len(all_samps), samples_per_mod, replace=False)
            selected = [all_samps[i] for i in indices]
        else:
            selected = all_samps
        X_mod = extract_all(selected)
        X_list.append(X_mod)
        y_list.extend([label_idx] * len(X_mod))

    X = np.vstack(X_list)
    y = np.array(y_list)

    # 划分 + 标准化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === 线性核 ===
    svm_linear = SVC(kernel='linear', C=10.0, random_state=RANDOM_SEED)
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    acc_linear = accuracy_score(y_test, y_pred_linear)

    # 每类准确率
    per_class_linear = {}
    for i, mod in enumerate(mod_list):
        mask = (y_test == i)
        if mask.sum() > 0:
            per_class_linear[mod] = accuracy_score(y_test[mask], y_pred_linear[mask])

    # === RBF核 ===
    svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=RANDOM_SEED)
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)

    per_class_rbf = {}
    for i, mod in enumerate(mod_list):
        mask = (y_test == i)
        if mask.sum() > 0:
            per_class_rbf[mod] = accuracy_score(y_test[mask], y_pred_rbf[mask])

    return {
        'linear_acc': acc_linear,
        'rbf_acc': acc_rbf,
        'delta': acc_rbf - acc_linear,
        'per_class_linear': per_class_linear,
        'per_class_rbf': per_class_rbf,
        'mod_list': mod_list,
        'n_samples': len(y),
    }


def main():
    print('=' * 70)
    print('  SVM 线性核 vs RBF核 对比测试')
    print('  目标: 找出线性核下准确率 >90% 的 4 类调制组合')
    print('=' * 70)

    # 按调制大类分组测试
    # 测试有意义的4类组合（每大类选代表，避免同类内耗）
    combinations = [
        # 各大类各选1个代表
        ['PAM4', 'GFSK', 'BPSK', 'QAM16'],          # 4ASK + 2FSK + BPSK + 16QAM
        ['PAM4', 'GFSK', 'QPSK', 'QAM16'],          # 4ASK + 2FSK + QPSK + 16QAM
        ['PAM4', 'GFSK', '8PSK', 'QAM16'],          # 4ASK + 2FSK + 8PSK + 16QAM
        ['PAM4', 'GFSK', 'BPSK', 'QPSK'],           # 4ASK + 2FSK + BPSK + QPSK ← 推荐
        ['PAM4', 'GFSK', 'BPSK', '8PSK'],           # 4ASK + 2FSK + BPSK + 8PSK (当前)
        ['PAM4', 'GFSK', 'QPSK', '8PSK'],           # 4ASK + 2FSK + QPSK + 8PSK

        # PSK类内两种 + ASK + FSK
        ['PAM4', 'CPFSK', 'BPSK', 'QPSK'],          # 4ASK + CPFSK + BPSK + QPSK

        # 跨大类比较
        ['PAM4', 'QAM16', 'BPSK', 'GFSK'],          # 同上排列不同
        ['PAM4', 'QAM64', 'BPSK', 'GFSK'],          # 换成64QAM
        ['PAM4', 'GFSK', 'BPSK', 'QAM64'],          # 4ASK + 2FSK + BPSK + 64QAM

        # 极端差别组合
        ['PAM4', 'GFSK', 'BPSK', 'QPSK'],           # 重复，验证稳定性
        ['PAM4', 'CPFSK', 'QPSK', 'QAM16'],         # 4ASK + CPFSK + QPSK + 16QAM

        # 不加PSK (只用ASK+FSK+QAM)
        ['PAM4', 'GFSK', 'QAM16', 'QAM64'],         # ❌ 2个QAM会内耗

        # 同大类2个 + 其他大类
        ['PAM4', 'BPSK', 'QPSK', 'GFSK'],           # ASK + 2PSK + FSK
        ['PAM4', 'QPSK', '8PSK', 'GFSK'],           # ASK + QPSK+8PSK(难!) + FSK
    ]

    # 去重
    seen = set()
    unique_combos = []
    for c in combinations:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_combos.append(c)
    combinations = unique_combos

    results = []
    for i, mods in enumerate(combinations):
        print(f'\n[{i+1}/{len(combinations)}] 测试: {", ".join(NAME_CN[m] for m in mods)}')
        try:
            r = test_combination(mods, samples_per_mod=600)
            results.append(r)
            print(f'  线性核: {r["linear_acc"]*100:.1f}%  |  RBF核: {r["rbf_acc"]*100:.1f}%  |  Δ: {r["delta"]*100:.1f}%')
            for mod in mods:
                pl = r['per_class_linear'].get(mod, 0) * 100
                pr = r['per_class_rbf'].get(mod, 0) * 100
                print(f'    {NAME_CN[mod]:>14s}:  线性 {pl:5.1f}%  |  RBF {pr:5.1f}%')
        except Exception as e:
            print(f'  ❌ 失败: {e}')

    # === 排序输出 ===
    print('\n' + '=' * 70)
    print('  最终排名 (按线性核准确率降序)')
    print('=' * 70)
    results.sort(key=lambda r: r['linear_acc'], reverse=True)

    print(f'  {"排名":<4} {"组合":<55} {"线性核":>8} {"RBF核":>8} {"Δ下降":>8}')
    print(f'  {"─"*4} {"─"*55} {"─"*8} {"─"*8} {"─"*8}')
    for rank, r in enumerate(results, 1):
        combo = ' + '.join(NAME_CN[m] for m in r['mod_list'])
        marker = ' ★' if r['linear_acc'] >= 0.90 else ''
        print(f'  {rank:<4} {combo:<55} {r["linear_acc"]*100:>7.1f}% {r["rbf_acc"]*100:>7.1f}% {r["delta"]*100:>7.1f}%{marker}')

    # 推荐
    print('\n' + '=' * 70)
    print('  推荐结论')
    print('=' * 70)
    good = [r for r in results if r['linear_acc'] >= 0.90]
    if good:
        print(f'  线性核准确率 ≥90% 的组合: {len(good)} 个')
        for r in good:
            combo = ' + '.join(NAME_CN[m] for m in r['mod_list'])
            print(f'    ✅ {combo}: {r["linear_acc"]*100:.1f}% (RBF: {r["rbf_acc"]*100:.1f}%, Δ={r["delta"]*100:.1f}%)')
    else:
        print('  ❌ 无组合达到线性核 90% 以上')
        # 最接近的
        best = results[0]
        combo = ' + '.join(NAME_CN[m] for m in best['mod_list'])
        print(f'  最接近: {combo}: {best["linear_acc"]*100:.1f}%')

    # 分析每个调制类型在 linear 下的表现
    print('\n' + '=' * 70)
    print('  各调制类型在线性核下的平均准确率 (跨所有组合)')
    print('=' * 70)
    per_mod_linear = {}
    per_mod_count = {}
    for r in results:
        for mod, acc in r['per_class_linear'].items():
            if mod not in per_mod_linear:
                per_mod_linear[mod] = []
                per_mod_count[mod] = 0
            per_mod_linear[mod].append(acc)
            per_mod_count[mod] += 1
    for mod in sorted(per_mod_linear.keys()):
        avg = np.mean(per_mod_linear[mod]) * 100
        mn = np.min(per_mod_linear[mod]) * 100
        mx = np.max(per_mod_linear[mod]) * 100
        print(f'  {NAME_CN[mod]:>14s}:  平均 {avg:5.1f}%  (范围 {mn:.1f}% ~ {mx:.1f}%, {per_mod_count[mod]}次)')

    return results


if __name__ == '__main__':
    main()

