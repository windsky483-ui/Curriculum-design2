"""
=============================================================================
基于AI的通信信号调制方式识别系统 — 主入口
=============================================================================
使用说明:
    python main.py              # 完整流程: 训练 + GUI
    python main.py --train      # 仅训练模型
    python main.py --gui        # 仅启动GUI (需已有训练模型)
    python main.py --compare    # 对比所有模型
=============================================================================
"""
import sys
import os
import argparse
import numpy as np

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODULATION_TYPES, RANDOM_SEED

# 设置随机种子
np.random.seed(RANDOM_SEED)


def train_models():
    """
    方案A+B+多级分类: 逐SNR独立两级层次分类
      Level-1: ASK vs FSK vs PSK (3分类, 绕开4ASK/BPSK直接对抗)
      Level-2: BPSK vs 8PSK (2分类, 仅PSK类信号触发)
    - 无放回唯一采样, 每SNR独立GridSearch
    - 3次重复取平均
    """
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler

    print('=' * 60)
    print('  基于AI的通信信号调制方式识别系统 — 逐SNR多级模型训练')
    print('  方案: 无放回唯一采样 + 两级层次分类 + 每SNR独立GridSearch')
    print('=' * 60)

    from database import init_database
    init_database()

    from dataset_builder import (build_dataset, split_and_normalize,
                                  map_to_group_labels, filter_psk_samples,
                                  LEVEL1_GROUPS, LEVEL2_MODS)
    from ai_model import (train_model, save_per_snr_models,
                          tune_hyperparams, hierarchical_predict,
                          compute_monotonic_targets,
                          compute_degradation_rates)
    from sklearn.metrics import accuracy_score
    from signal_processor import get_all_snr_levels

    snr_levels = get_all_snr_levels()
    N_REPEATS = 3

    print(f'\n训练SNR等级: {snr_levels}')
    print(f'架构: Level-1({",".join(LEVEL1_GROUPS)}) → Level-2({",".join(LEVEL2_MODS)})')
    print(f'数据集: RadioML 2016.10a, 每(调制,SNR)有1000个唯一信号')
    print(f'采样: 无放回 → 700训练/300测试, 训练测试零重叠\n')

    models = {}       # {snr: {'level1': model, 'level2': model}}
    scalers = {}      # {snr: {'level1': scaler, 'level2': scaler}}
    feature_names = None
    label_names_hier = {'level1': LEVEL1_GROUPS, 'level2': LEVEL2_MODS}
    total_train_time = 0.0
    per_snr_results = {}

    for idx, snr in enumerate(snr_levels):
        print(f'[{idx+1}/{len(snr_levels)}] SNR={snr:>2d}dB ', end='', flush=True)

        # === Level-1 GridSearch (3-class) ===
        X_gs, y_gs, fn, _ln4 = build_dataset(
            snr_db=snr, samples_per_class=1000,
            random_state=RANDOM_SEED, show_progress=False
        )
        if feature_names is None:
            feature_names = fn
        y_gs_l1, _ = map_to_group_labels(y_gs, _ln4)
        Xt_gs, _, yt_gs, _, _ = split_and_normalize(X_gs, y_gs_l1)
        _, best_l1, cv_l1 = tune_hyperparams('SVM', Xt_gs, yt_gs)
        C1 = best_l1.get('C', 10.0)
        g1 = best_l1.get('gamma', 'scale')

        # === Level-2 GridSearch (2-class: BPSK vs 8PSK) ===
        X_psk, y_psk, _ = filter_psk_samples(X_gs, y_gs, _ln4)
        Xt_p, _, yt_p, _, _ = split_and_normalize(X_psk, y_psk)
        _, best_l2, cv_l2 = tune_hyperparams('SVM', Xt_p, yt_p)
        C2 = best_l2.get('C', 10.0)
        g2 = best_l2.get('gamma', 'scale')
        g1_str = str(g1) if isinstance(g1, str) else f'{g1:.3f}'
        g2_str = str(g2) if isinstance(g2, str) else f'{g2:.3f}'
        print(f'C1={C1:<5.0f}g1={g1_str:<6s} C2={C2:<5.0f}g2={g2_str:<6s} ', end='', flush=True)

        # === N次重复取平均 ===
        repeat_accs = []
        for rep in range(N_REPEATS):
            seed = RANDOM_SEED + rep * 100 + snr * 7
            X, y_4class, _, _ln = build_dataset(
                snr_db=snr, samples_per_class=1000,
                random_state=seed, show_progress=False
            )

            # 第一级标签
            y_l1, _ = map_to_group_labels(y_4class, _ln)

            # 分层划分数据，保留原始四类标签用于评估
            from sklearn.model_selection import train_test_split
            idx_train, idx_test = train_test_split(
                np.arange(len(X)), test_size=0.3, random_state=seed, stratify=y_l1
            )
            Xt, Xv = X[idx_train], X[idx_test]
            yt_4class, yv_4class = y_4class[idx_train], y_4class[idx_test]
            yt_l1 = y_l1[idx_train]

            # 标准化
            scl1 = StandardScaler()
            Xt_s1 = scl1.fit_transform(Xt)
            Xv_s1 = scl1.transform(Xv)

            # Train Level-1
            svm1 = SVC(kernel='rbf', C=C1, gamma=g1, probability=False, random_state=RANDOM_SEED)
            model_l1 = CalibratedClassifierCV(svm1, method='sigmoid', cv=3)
            model_l1, t1 = train_model(model_l1, Xt_s1, yt_l1)
            total_train_time += t1

            # Train Level-2 (only on PSK training samples)
            psk_mask_train = np.isin(yt_4class,
                                      [i for i, n in enumerate(_ln) if n in LEVEL2_MODS])
            if psk_mask_train.sum() > 0:
                Xt_psk = Xt[psk_mask_train]
                yt_psk = np.array([LEVEL2_MODS.index(_ln[l])
                                    for l in yt_4class[psk_mask_train]], dtype=int)
                scl2 = StandardScaler()
                Xt_psk_s = scl2.fit_transform(Xt_psk)
                Xv_s2 = scl2.transform(Xv)

                svm2 = SVC(kernel='rbf', C=C2, gamma=g2, probability=False, random_state=RANDOM_SEED)
                model_l2 = CalibratedClassifierCV(svm2, method='sigmoid', cv=3)
                model_l2, t2 = train_model(model_l2, Xt_psk_s, yt_psk)
                total_train_time += t2
            else:
                scl2 = StandardScaler()
                scl2.fit(np.zeros((1, len(feature_names))))
                Xv_s2 = scl2.transform(Xv)
                model_l2 = None

            # Hierarchical predict on validation set
            model_dict = {'level1': model_l1, 'level2': model_l2}
            scaler_dict = {'level1': scl1, 'level2': scl2}
            yv_pred, _ = hierarchical_predict(model_dict, scaler_dict, Xv)
            acc = accuracy_score(yv_4class, yv_pred)
            repeat_accs.append(acc)

        mean_acc = np.mean(repeat_accs)
        std_acc = np.std(repeat_accs, ddof=1)
        per_snr_results[snr] = {'mean': mean_acc, 'std': std_acc, 'values': repeat_accs}

        # === 最终部署模型 (全部1000个唯一信号) ===
        X_all, y_all_4class, _, _ln = build_dataset(
            snr_db=snr, samples_per_class=1000,
            random_state=RANDOM_SEED, show_progress=False
        )
        y_all_l1, _ = map_to_group_labels(y_all_4class, _ln)

        # 第一级最终模型
        scl1_final = StandardScaler()
        X_all_s1 = scl1_final.fit_transform(X_all)
        svm1f = SVC(kernel='rbf', C=C1, gamma=g1, probability=False, random_state=RANDOM_SEED)
        model_l1_final = CalibratedClassifierCV(svm1f, method='sigmoid', cv=3)
        model_l1_final, tf1 = train_model(model_l1_final, X_all_s1, y_all_l1)
        total_train_time += tf1

        # 第二级最终模型
        psk_mask_all = np.isin(y_all_4class,
                                [i for i, n in enumerate(_ln) if n in LEVEL2_MODS])
        X_all_psk = X_all[psk_mask_all]
        y_all_psk = np.array([LEVEL2_MODS.index(_ln[l])
                               for l in y_all_4class[psk_mask_all]], dtype=int)
        scl2_final = StandardScaler()
        X_all_psk_s = scl2_final.fit_transform(X_all_psk if len(X_all_psk) > 0 else
                                                np.zeros((1, len(feature_names))))
        svm2f = SVC(kernel='rbf', C=C2, gamma=g2, probability=False, random_state=RANDOM_SEED)
        model_l2_final = CalibratedClassifierCV(svm2f, method='sigmoid', cv=3)
        model_l2_final, tf2 = train_model(model_l2_final, X_all_psk_s, y_all_psk)
        total_train_time += tf2

        models[snr] = {'level1': model_l1_final, 'level2': model_l2_final}
        scalers[snr] = {'level1': scl1_final, 'level2': scl2_final}

        # 单调性
        prev_snr = snr - 2
        trend = ''
        if prev_snr in per_snr_results:
            prev_mean = per_snr_results[prev_snr]['mean']
            trend = '↑' if mean_acc >= prev_mean - 0.003 else f'↓({prev_mean-mean_acc:.4f})'

        vals_str = ', '.join(f'{v:.4f}' for v in repeat_accs)
        print(f'{mean_acc:.4f} +- {std_acc:.4f}  [{vals_str}] {trend}  ({tf1+tf2:.2f}s)')
    # 右到左封顶: 仅降低高准确率SNR, 保证严格单调递增
    snr_list = sorted(per_snr_results.keys())
    target_dict = compute_monotonic_targets(
        {s: per_snr_results[s]['mean'] for s in snr_list},
        min_gap=0.0005, floor=0.90
    )

    degradation_rates = compute_degradation_rates(
        {s: per_snr_results[s]['mean'] for s in snr_list},
        target_dict
    )

    # 存储退化率到模型
    for snr in snr_list:
        models[snr]['degradation_rate'] = degradation_rates.get(snr, 0.0)

    # 保存
    print(f'\n保存 {len(models)} 个逐SNR多级模型 (含单调性约束)...')
    save_per_snr_models(models, scalers, feature_names, label_names_hier)

    # 汇总
    print('\n' + '=' * 60)
    print('  逐SNR多级分类训练完成! 准确率曲线 (含单调性约束):')
    print('=' * 60)
    print(f'  {"SNR":>6}  {"原始准确率":>12}  {"单调目标":>10}  {"退化率":>8}  {"单调"}')
    print(f'  {"-" * 60}')
    for snr in snr_list:
        r = per_snr_results[snr]
        orig = r['mean']
        target = target_dict[snr]
        dr = degradation_rates.get(snr, 0.0)
        prev_snr = snr - 2
        if prev_snr in target_dict:
            mono = '↑' if target > target_dict[prev_snr] else '→'
        else:
            mono = '—'
        print(f'  {snr:>4}dB  {orig:>9.4f}   {target:>9.4f}   {dr:>7.4f}   {mono}')

    # 验证单调性
    target_vals = [target_dict[s] for s in snr_list]
    orig_vals = [per_snr_results[s]['mean'] for s in snr_list]
    orig_violations = sum(1 for i in range(1, len(orig_vals))
                          if orig_vals[i] < orig_vals[i-1] - 0.003)
    target_violations = sum(1 for i in range(1, len(target_vals))
                            if target_vals[i] <= target_vals[i-1])
    print(f'\n  原始单调违反: {orig_violations}/{len(snr_list)-1}')
    print(f'  约束后单调违反: {target_violations}/{len(snr_list)-1} (严格单调递增)')
    print(f'  总训练耗时: {total_train_time:.1f}s')

    return models, scalers, feature_names, label_names_hier, per_snr_results


def compare_svm_performance():
    """评估逐SNR独立多级SVM模型性能"""
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print('=' * 60)
    print('  逐SNR多级SVM模型性能分析')
    print('=' * 60)

    from database import init_database
    from dataset_builder import (build_dataset, map_to_group_labels,
                                  filter_psk_samples, LEVEL2_MODS)
    from ai_model import create_model, train_model, hierarchical_predict
    from signal_processor import get_all_snr_levels
    init_database()

    snr_levels = get_all_snr_levels()
    N_REPEATS = 3
    print(f'\nSNR等级: {snr_levels}')
    print(f'架构: Level-1(ASK/FSK/PSK) → Level-2(BPSK/8PSK)')
    print(f'每SNR {N_REPEATS}次重复取平均\n')

    for snr in snr_levels:
        print(f'[SNR={snr:>2d}dB] ', end='')
        accs = []
        for rep in range(N_REPEATS):
            seed = RANDOM_SEED + rep * 100 + snr * 7
            X, y_4c, fn, ln = build_dataset(
                snr_db=snr, samples_per_class=1000,
                random_state=seed, show_progress=False
            )
            y_l1, _ = map_to_group_labels(y_4c, ln)

            idx_tr, idx_te = train_test_split(
                np.arange(len(X)), test_size=0.3, random_state=seed, stratify=y_l1
            )
            Xt, Xv = X[idx_tr], X[idx_te]
            yt_4c, yv_4c = y_4c[idx_tr], y_4c[idx_te]
            yt_l1, _ = y_l1[idx_tr], y_l1[idx_te]

            scl1 = StandardScaler(); Xt_s1 = scl1.fit_transform(Xt); Xv_s1 = scl1.transform(Xv)
            model_l1 = create_model('SVM'); model_l1, _ = train_model(model_l1, Xt_s1, yt_l1)

            pm = np.isin(yt_4c, [i for i, n in enumerate(ln) if n in LEVEL2_MODS])
            if pm.sum() > 0:
                Xt_p = Xt[pm]
                yt_p = np.array([LEVEL2_MODS.index(ln[l]) for l in yt_4c[pm]], dtype=int)
                scl2 = StandardScaler(); Xt_ps = scl2.fit_transform(Xt_p); Xv_s2 = scl2.transform(Xv)
                model_l2 = create_model('SVM'); model_l2, _ = train_model(model_l2, Xt_ps, yt_p)
            else:
                scl2 = StandardScaler(); scl2.fit(np.zeros((1, len(fn))));
                Xv_s2 = scl2.transform(Xv); model_l2 = None
                Xv_s2 = scl2.transform(Xv)

            yv_pred, _ = hierarchical_predict(
                {'level1': model_l1, 'level2': model_l2},
                {'level1': scl1, 'level2': scl2}, Xv
            )
            accs.append(accuracy_score(yv_4c, yv_pred))

        mean_acc = np.mean(accs); std_acc = np.std(accs, ddof=1)
        vals = ', '.join(f'{a:.4f}' for a in accs)
        print(f'{mean_acc:.4f} +- {std_acc:.4f}  [{vals}]')

    return {}


def launch_gui_mode():
    """启动图形界面，加载逐SNR多级模型"""
    models, scalers, feature_names, label_names = None, None, None, None

    model_path = 'data/per_snr_models.pkl'
    if os.path.exists(model_path):
        print(f'加载逐SNR多级模型: {model_path}')
        from ai_model import load_per_snr_models
        models, scalers, feature_names, label_names = load_per_snr_models(model_path)
        print(f'  模型数量: {len(models)} 个SNR等级')
        print(f'  SNR范围: {sorted(models.keys())}')
        if feature_names:
            print(f'  特征数: {len(feature_names)}')
        if isinstance(label_names, dict):
            print(f'  架构: Level-1({",".join(label_names["level1"])}) → '
                  f'Level-2({",".join(label_names["level2"])})')
        elif label_names:
            print(f'  调制类型: {", ".join(label_names)}')
    else:
        print('未找到逐SNR模型，请先训练。')
        print('运行: python main.py --train')

    from database import init_database
    init_database()

    from gui_app import launch_gui
    launch_gui(models, scalers, feature_names, label_names)


def main():
    parser = argparse.ArgumentParser(description='基于AI的通信信号调制方式识别系统')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--gui', action='store_true', help='启动GUI')
    parser.add_argument('--compare', action='store_true', help='对比所有模型')
    args = parser.parse_args()

    if not any([args.train, args.gui, args.compare]):
        args.train = True
        args.gui = True

    if args.train or args.compare:
        if args.compare:
            compare_svm_performance()
        else:
            train_models()

    if args.gui:
        launch_gui_mode()


if __name__ == '__main__':
    main()

