"""
=============================================================================
GUI可视化模块 — 基于 Tkinter + Matplotlib 的调制识别交互界面
=============================================================================
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import time
import threading

from config import (
    MODULATION_TYPES, FS, FC, RS, N_SYMBOLS, SPB, GUI_TITLE, GUI_SIZE, PLOT_DPI
)
from signal_processor import get_signal, get_all_snr_levels, get_available_mod_types
from feature_extraction import extract_features, extract_features_batch
from ai_model import (create_model, train_model, evaluate_model, load_model,
                       hierarchical_predict, hierarchical_predict_single)
from database import (
    init_database, insert_signal_record, insert_recognition_result,
    get_recognition_history, get_recognition_statistics,
    insert_model_performance
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModulationRecognitionGUI:
    """调制方式识别系统主界面"""

    def __init__(self, root, models=None, scalers=None, feature_names=None, label_names=None):
        self.root = root
        self.root.title(GUI_TITLE)
        self.root.geometry(GUI_SIZE)
        self.root.minsize(1200, 750)

        # 方案A+多级分类: 逐SNR独立两级模型 {snr: {'level1': model, 'level2': model}}
        self.models = models or {}
        self.scalers = scalers or {}
        self.feature_names = feature_names

        # 检测模型类型: 层级式 (label_names为dict) 还是 扁平式 (label_names为list)
        if isinstance(label_names, dict):
            # 层级式: {'level1': ['ASK','FSK','PSK'], 'level2': ['BPSK','8PSK']}
            self.label_names = label_names
            self._is_hierarchical = True
            # 展平4类标签用于显示
            self._flat_labels = ['4ASK', '2FSK', 'BPSK', '8PSK']
        else:
            # 扁平式 (旧模型兼容)
            if label_names is not None:
                self.label_names = ['2FSK' if n == '4FSK' else n for n in label_names]
            else:
                self.label_names = None
            self._is_hierarchical = False
            self._flat_labels = self.label_names
        self.current_signal = None
        self.current_symbols = None
        self.current_mod_type = None
        self.current_snr = None

        # 确保snr_levels排序
        self._sorted_snrs = sorted(self.models.keys()) if self.models else []

        # 样式配置
        self._setup_style()
        # 布局
        self._build_layout()
        # 状态栏
        self._build_statusbar()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', font=('Microsoft YaHei', 9))
        style.configure('TButton', font=('Microsoft YaHei', 9))
        style.configure('TCombobox', font=('Microsoft YaHei', 9))
        style.configure('TLabelframe.Label', font=('Microsoft YaHei', 10, 'bold'))
        style.configure('Header.TLabel', font=('Microsoft YaHei', 16, 'bold'))
        style.configure('Result.TLabel', font=('Microsoft YaHei', 11, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')

    def _build_layout(self):
        # 顶部标题
        header = ttk.Label(self.root, text=GUI_TITLE, style='Header.TLabel')
        header.pack(pady=(10, 5))

        # 主区域: 左侧控制面板 + 右侧可视化
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧控制面板
        self._build_control_panel(main_frame)
        # 右侧可视化区域
        self._build_visualization_panel(main_frame)

        # 底部状态/历史表格
        self._build_history_panel()

    def _build_control_panel(self, parent):
        """左侧控制面板"""
        panel = ttk.Frame(parent, width=320)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        panel.pack_propagate(False)
        sig_frame = ttk.Labelframe(panel, text='信号参数设置', padding=10)
        sig_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sig_frame, text='调制类型:').grid(row=0, column=0, sticky='w', pady=2)
        self.mod_type_var = tk.StringVar(value='BPSK')
        mod_combo = ttk.Combobox(sig_frame, textvariable=self.mod_type_var,
                                  values=MODULATION_TYPES, state='readonly', width=18)
        mod_combo.grid(row=0, column=1, sticky='ew', pady=2)

        ttk.Label(sig_frame, text='SNR (dB):').grid(row=1, column=0, sticky='w', pady=2)
        self.snr_var = tk.StringVar(value='10')
        snr_combo = ttk.Combobox(sig_frame, textvariable=self.snr_var,
                                  values=[str(s) for s in get_all_snr_levels()],
                                  state='readonly', width=8)
        snr_combo.grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(sig_frame, text='信号来源:').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Label(sig_frame, text='RML2016.10a-main (txt, 128点)').grid(row=2, column=1, sticky='w', pady=2)

        # 按钮
        btn_frame = ttk.Frame(sig_frame)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(btn_frame, text='加载信号', command=self._on_load_signal).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='识别信号', command=self._on_recognize).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='批量测试', command=self._on_batch_test).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text='导入外部数据', command=self._on_import_dataset).pack(
            side=tk.LEFT, padx=3)
        result_frame = ttk.Labelframe(panel, text='识别结果', padding=10)
        result_frame.pack(fill=tk.X, pady=(0, 8))

        self.result_text = tk.Text(result_frame, height=6, width=35, font=('Consolas', 9))
        self.result_text.pack(fill=tk.X)
        model_frame = ttk.Labelframe(panel, text='模型配置', padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(model_frame, text='使用模型:').pack(anchor='w')
        self.model_type_var = tk.StringVar(value='SVM')
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_type_var,
                                    values=['SVM'],
                                    state='readonly', width=20)
        model_combo.pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text='重新训练模型', command=self._on_retrain).pack(
            fill=tk.X, pady=(5, 0))

    def _build_visualization_panel(self, parent):
        """右侧可视化区域 (4个子图)"""
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9, 8), dpi=PLOT_DPI)
        self.fig.subplots_adjust(hspace=0.35, wspace=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, viz_frame)
        toolbar.update()

        # 创建4个子图
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        self.ax_wave   = self.fig.add_subplot(gs[0, 0])  # 时域波形
        self.ax_spec   = self.fig.add_subplot(gs[0, 1])  # 频谱
        self.ax_const   = self.fig.add_subplot(gs[1, 0])  # 星座图
        self.ax_acc     = self.fig.add_subplot(gs[1, 1])  # 准确率曲线

        self.ax_wave.set_title('时域信号波形', fontsize=10)
        self.ax_wave.set_xlabel('时间 (s)'); self.ax_wave.set_ylabel('幅度')
        self.ax_spec.set_title('幅度谱', fontsize=10)
        self.ax_spec.set_xlabel('频率 (Hz)'); self.ax_spec.set_ylabel('幅度')
        self.ax_const.set_title('星座图 (I/Q)', fontsize=10)
        self.ax_const.set_xlabel('I'); self.ax_const.set_ylabel('Q')
        self.ax_acc.set_title('SVM准确率 vs SNR 曲线', fontsize=10)
        self.ax_acc.set_xlabel('SNR (dB)'); self.ax_acc.set_ylabel('准确率 (%)')

        for ax in [self.ax_wave, self.ax_spec, self.ax_const, self.ax_acc]:
            ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def _build_history_panel(self):
        """底部历史记录"""
        hist_frame = ttk.Labelframe(self.root, text='识别历史记录', padding=5)
        hist_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        columns = ('#', '实际类型', '识别结果', '置信度', '正确', '模型', '时间')
        self.hist_tree = ttk.Treeview(hist_frame, columns=columns, show='headings',
                                       height=4)
        for col in columns:
            self.hist_tree.heading(col, text=col)
        widths = [30, 80, 80, 60, 50, 80, 140]
        for col, w in zip(columns, widths):
            self.hist_tree.column(col, width=w, anchor='center')

        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL,
                                   command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=scrollbar.set)
        self.hist_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(hist_frame, text='刷新历史', command=self._refresh_history).pack(
            side=tk.RIGHT, padx=5)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value='就绪 — 请加载信号或训练模型 (数据集: RadioML 2016.10a)')
        statusbar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor='w')
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _on_load_signal(self):
        """加载信号按钮回调 — 从 RadioML 2016.10a 数据集中取信号"""
        mod_type = self.mod_type_var.get()
        snr = int(self.snr_var.get())

        self.set_status(f'正在从数据集加载 {mod_type} 信号 (SNR={snr}dB)...')

        try:
            sig, actual_mod, actual_snr = get_signal(mod_type, snr_db=snr)
            self.current_signal = sig
            self.current_symbols = None
            self.current_mod_type = actual_mod
            self.current_snr = actual_snr

            # 更新波形图
            self._plot_signal(sig, actual_mod, actual_snr)
            # 记录到数据库
            try:
                insert_signal_record(actual_mod, actual_snr, FS, RS, N_SYMBOLS)
            except Exception:
                pass

            self.set_status(f'已加载 {actual_mod} 信号 (SNR={actual_snr}dB, 128采样点)')
        except Exception as e:
            messagebox.showerror('错误', f'信号加载失败:\n{e}')
            self.set_status('信号加载失败')

    def _on_recognize(self):
        """识别信号 — 使用与信号SNR最匹配的逐SNR模型 (支持多级分类)"""
        if self.current_signal is None:
            messagebox.showwarning('提示', '请先生成信号')
            return
        if not self.models:
            messagebox.showwarning('提示', '模型未加载，请先训练或加载模型')
            return

        self.set_status('正在提取特征并识别...')

        try:
            # 选择与当前信号SNR最接近的模型
            snr = self.current_snr
            if snr is None:
                snr = 10  # fallback
            model_snr = min(self._sorted_snrs, key=lambda s: abs(s - snr))
            model_entry = self.models[model_snr]
            scaler_entry = self.scalers[model_snr]

            # 提取特征
            fv, _ = extract_features(self.current_signal)

            # 预测 (分支: 层级式 vs 扁平式)
            t0 = time.time()
            if self._is_hierarchical:
                pred, proba = hierarchical_predict_single(
                    model_entry, scaler_entry, fv)
            else:
                fv_scaled = scaler_entry.transform([fv])
                pred = model_entry.predict(fv_scaled)[0]
                proba = model_entry.predict_proba(fv_scaled)[0] if hasattr(
                    model_entry, 'predict_proba') else None
            elapsed = time.time() - t0

            pred_label = self._flat_labels[pred] if self._flat_labels else str(pred)
            is_correct = (pred_label == self.current_mod_type)
            max_prob = np.max(proba) if proba is not None else 1.0

            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f'实际调制方式: {self.current_mod_type}\n')
            self.result_text.insert(tk.END, f'信号 SNR:     {self.current_snr} dB\n')
            self.result_text.insert(tk.END, f'使用模型:     SNR={model_snr}dB')
            if self._is_hierarchical:
                self.result_text.insert(tk.END, ' (多级分类)\n')
            else:
                self.result_text.insert(tk.END, '\n')
            self.result_text.insert(tk.END, f'预测调制方式: {pred_label}\n')
            self.result_text.insert(tk.END, f'置信度:       {max_prob:.4f}\n')
            self.result_text.insert(tk.END, f'识别正确:     {"✓ 正确" if is_correct else "✗ 错误"}\n')
            self.result_text.insert(tk.END, f'推理耗时:     {elapsed * 1000:.2f} ms\n')

            if proba is not None:
                self.result_text.insert(tk.END, '\n各类别概率:\n')
                for name, p in zip(self._flat_labels, proba):
                    marker = '→ ' if name == pred_label else '   '
                    self.result_text.insert(tk.END, f'{marker}{name}: {p:.4f}\n')

            # 保存结果
            try:
                insert_recognition_result(
                    None, self.current_mod_type, pred_label,
                    max_prob, int(is_correct), self.model_type_var.get()
                )
            except Exception:
                pass

            self._refresh_history()
            self.set_status(
                f'识别完成: {self.current_mod_type} → {pred_label} '
                f'({"正确" if is_correct else "错误"}, {elapsed*1000:.1f}ms)'
            )
        except Exception as e:
            messagebox.showerror('错误', f'识别失败:\n{e}')
            self.set_status('识别失败')

    def _on_batch_test(self):
        """批量测试 — 逐SNR独立模型，准确率天然单调递增"""
        if not self.models:
            messagebox.showwarning('提示', '模型未加载，请先训练或加载模型')
            return

        snr_list = get_all_snr_levels()
        mod_types = MODULATION_TYPES
        n_per = 200  # 每类每SNR测试样本数

        # 预生成所有测试用例: [(snr, mod), ...]
        self._bt_cases = []
        for snr in snr_list:
            for mod in mod_types:
                for _ in range(n_per):
                    self._bt_cases.append((snr, mod))

        self._bt_total_cases = len(self._bt_cases)
        self._bt_case_idx = 0
        self._bt_snr_correct = {}
        self._bt_snr_total = {}
        self._bt_snr_accuracies = {}

        for snr in snr_list:
            self._bt_snr_correct[snr] = 0
            self._bt_snr_total[snr] = 0

        self.set_status(f'逐SNR批量测试开始 ({n_per}样本/类/SNR, 共{self._bt_total_cases}测试)...')
        self.root.after(50, self._batch_test_step)

    def _batch_test_step(self):
        """每片处理 10 个测试用例 — 逐SNR使用对应模型 (支持多级分类)"""
        step_size = 10
        try:
            for _ in range(step_size):
                if self._bt_case_idx >= self._bt_total_cases:
                    # 全部完成 — 计算各SNR准确率
                    for snr in self._bt_snr_total:
                        total = self._bt_snr_total[snr]
                        correct = self._bt_snr_correct.get(snr, 0)
                        self._bt_snr_accuracies[snr] = correct / max(total, 1) * 100
                    self._show_batch_result(self._bt_snr_accuracies)
                    return

                snr, mod = self._bt_cases[self._bt_case_idx]
                sig, _, actual_snr = get_signal(mod, snr_db=snr)

                # 选择与信号SNR最匹配的逐SNR模型
                model_snr = min(self._sorted_snrs, key=lambda s: abs(s - actual_snr))
                model_entry = self.models[model_snr]
                scaler_entry = self.scalers[model_snr]

                fv, _ = extract_features(sig)

                # 预测 (分支: 层级式 vs 扁平式)
                if self._is_hierarchical:
                    pred, _ = hierarchical_predict_single(
                        model_entry, scaler_entry, fv)
                else:
                    fv_s = scaler_entry.transform([fv])
                    pred = model_entry.predict(fv_s)[0]
                pred_label = self._flat_labels[pred] if self._flat_labels else str(pred)

                # 按 actual_snr 分组统计
                stat_snr = actual_snr
                if stat_snr not in self._bt_snr_total:
                    self._bt_snr_total[stat_snr] = 0
                    self._bt_snr_correct[stat_snr] = 0

                self._bt_snr_total[stat_snr] += 1
                if pred_label == mod:
                    self._bt_snr_correct[stat_snr] += 1

                self._bt_case_idx += 1

        except Exception as e:
            messagebox.showerror('错误', f'批量测试失败:\n{e}')
            self.set_status('批量测试失败')
            return

        # 更新进度
        if self._bt_case_idx < self._bt_total_cases:
            pct = self._bt_case_idx / self._bt_total_cases * 100
            current_snr = self._bt_cases[self._bt_case_idx][0]
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f'=== 逐SNR批量测试进行中 ===\n')
            model_type = '多级SVM' if self._is_hierarchical else 'SVM'
            self.result_text.insert(tk.END, f'模型: {model_type}\n')
            self.result_text.insert(tk.END, f'当前SNR: {current_snr}dB\n')
            self.result_text.insert(tk.END, f'进度: {self._bt_case_idx}/{self._bt_total_cases} ({pct:.0f}%)\n\n')
            self.result_text.insert(tk.END, '已完成:\n')
            for s in sorted(self._bt_snr_total.keys()):
                t = self._bt_snr_total[s]
                if t > 0:
                    c = self._bt_snr_correct.get(s, 0)
                    a = c / t * 100
                    bar = '█' * int(a / 5) + '░' * (20 - int(a / 5))
                    self.result_text.insert(tk.END, f'  SNR={s:>3d}dB: {a:5.1f}% {bar} ({t}样本)\n')

        self.root.after(10, self._batch_test_step)

    def _on_import_dataset(self):
        """导入外部数据集按钮 — 选择文件并测试 (支持多级分类)"""
        if not self.models:
            messagebox.showwarning('提示', '模型未加载，请先训练或加载模型')
            return

        filepath = filedialog.askopenfilename(
            title='选择数据集文件',
            filetypes=[('数据文件', '*.pkl *.npy *.mat *.h5 *.hdf5'),
                       ('Pickle', '*.pkl'), ('NumPy', '*.npy'), ('MATLAB', '*.mat'),
                       ('HDF5', '*.h5 *.hdf5'), ('所有文件', '*.*')],
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_dataset')
        )
        if not filepath:
            return

        self.set_status(f'正在加载外部数据集: {os.path.basename(filepath)}...')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f'正在加载: {os.path.basename(filepath)}\n请稍候...\n')

        def run_import():
            try:
                from ai_model import load_per_snr_models
                import os as _os
                rml_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                         'data', 'per_snr_models.pkl')
                if _os.path.exists(rml_path):
                    r_models, r_scalers, rfn, rln = load_per_snr_models(rml_path)
                else:
                    self.root.after(0, lambda: messagebox.showerror('错误',
                        '未找到逐SNR模型，请先运行 python main.py --train'))
                    return

                from dataset_loader import test_dataset_accuracy
                # 选择一个代表性SNR模型 (优先10dB, 否则选中间的)
                sorted_snrs = sorted(r_models.keys())
                rep_snr = 10 if 10 in sorted_snrs else sorted_snrs[len(sorted_snrs)//2]
                results = test_dataset_accuracy(filepath, r_models[rep_snr],
                                                r_scalers[rep_snr], rln, min_snr=0)

                self.root.after(0, lambda: self._show_import_result(
                    filepath, results['total_signals'],
                    results['tested'], results['correct'],
                    results['accuracy']))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror('错误', f'导入失败:\n{e}'))
                self.root.after(0, lambda: self.set_status('导入失败'))

        threading.Thread(target=run_import, daemon=True).start()

    def _show_import_result(self, filepath, n_signals, n_valid, n_correct, acc):
        """显示外部数据集测试结果"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f'=== 外部数据集测试 ===\n')
        self.result_text.insert(tk.END, f'文件: {os.path.basename(filepath)}\n')
        self.result_text.insert(tk.END, f'总样本数: {n_signals}\n')
        if acc is not None:
            self.result_text.insert(tk.END, f'有效样本: {n_valid}\n')
            self.result_text.insert(tk.END, f'正确识别: {n_correct}\n')
            self.result_text.insert(tk.END, f'准确率: {acc:.2f}%\n')
        else:
            self.result_text.insert(tk.END, f'预测完成（无标签，无法计算准确率）\n')
        self.set_status(f'外部数据集测试完成: {os.path.basename(filepath)}')

    def _show_batch_result(self, snr_accuracies):
        """显示逐SNR独立模型批量测试结果 (准确率天然单调递增)"""
        snr_list = sorted(snr_accuracies.keys())
        if not snr_list:
            return
        overall_acc = sum(snr_accuracies.values()) / len(snr_accuracies)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, '=== 逐SNR独立模型批量测试结果 ===\n')
        self.result_text.insert(tk.END, f'测试SNR范围: {snr_list[0]} ~ {snr_list[-1]} dB\n')
        n_mods = len(MODULATION_TYPES)
        self.result_text.insert(tk.END, f'各SNR下测试样本数: ~{n_mods * 200} ({n_mods}类 × 200样本)\n')
        self.result_text.insert(tk.END, f'平均准确率: {overall_acc:.2f}%\n')
        self.result_text.insert(tk.END, f'策略: 逐SNR独立训练, 高SNR天然准确率更高\n\n')

        self.result_text.insert(tk.END, '各SNR准确率:\n')
        for snr in snr_list:
            acc = snr_accuracies[snr]
            bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
            self.result_text.insert(tk.END, f'  SNR={snr:>3d}dB: {acc:5.1f}% {bar}\n')

        # 更新准确率曲线图
        self._plot_accuracy_curve(snr_accuracies)
        self.set_status(f'逐SNR批量测试完成: 平均准确率 {overall_acc:.2f}%')

    def _on_retrain(self):
        """重新训练所有逐SNR独立多级模型 (方案A + 多级分类)"""
        n_snrs = len(get_all_snr_levels())
        answer = messagebox.askyesno('确认',
            f'将逐SNR训练 {n_snrs} 个两级层次SVM模型,\n'
            f'含GridSearch超参搜索, 预计需要约{n_snrs * 25}秒，是否继续？')
        if not answer:
            return
        self.set_status(f'正在逐SNR训练多级模型，请稍候...')

        def train():
            try:
                # 调用 main.py 中的完整训练流程
                from main import train_models
                models, scalers, fn, ln, per_snr_results = train_models()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror('错误', f'训练失败:\n{e}'))
                self.root.after(0, lambda: self.set_status('训练失败'))
                return

            self.root.after(0, lambda: self._on_per_snr_train_done(
                models, scalers, fn, ln))

        threading.Thread(target=train, daemon=True).start()

    def _on_per_snr_train_done(self, models, scalers, fn, ln):
        """逐SNR训练完成回调 (支持层级式模型)"""
        self.models = models
        self.scalers = scalers
        self.feature_names = fn

        # 检测模型类型
        if isinstance(ln, dict):
            self.label_names = ln
            self._is_hierarchical = True
            self._flat_labels = ['4ASK', '2FSK', 'BPSK', '8PSK']
        else:
            self.label_names = ln
            self._is_hierarchical = False
            self._flat_labels = ln
        self._sorted_snrs = sorted(models.keys())

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f'=== 逐SNR多级训练完成 ===\n')
        self.result_text.insert(tk.END, f'模型数量: {len(models)} 个SNR等级\n')
        if self._is_hierarchical:
            self.result_text.insert(tk.END, f'架构: Level-1({",".join(ln["level1"])}) '
                                      f'→ Level-2({",".join(ln["level2"])})\n')
        self.result_text.insert(tk.END, f'特征数: {len(fn)}\n')
        self.result_text.insert(tk.END, f'调制类型: {", ".join(self._flat_labels)}\n\n')
        self.result_text.insert(tk.END, '请点击"批量测试"查看准确率曲线\n')

        self.set_status(f'多级模型训练完成: {len(models)} 个SNR等级')
        messagebox.showinfo('完成', f'{len(models)} 个逐SNR多级模型已训练并保存。')

    def _refresh_history(self):
        """刷新历史记录表格"""
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)
        try:
            rows = get_recognition_history(limit=20)
            for row in reversed(rows):
                id_, actual, pred, conf, correct, model, ts = row
                tag = 'correct' if correct else 'wrong'
                self.hist_tree.insert('', 'end', values=(
                    id_, actual, pred, f'{conf:.3f}' if conf else '-',
                    '✓' if correct else '✗', model, ts
                ), tags=(tag,))
            self.hist_tree.tag_configure('correct', background='#d4edda')
            self.hist_tree.tag_configure('wrong', background='#f8d7da')
        except Exception:
            pass

    def _plot_signal(self, sig, mod_type, snr):
        """绘制信号波形、频谱、星座图 (RadioML 128点信号)"""
        # 清空
        for ax in [self.ax_wave, self.ax_spec, self.ax_const]:
            ax.clear()
            ax.grid(True, alpha=0.3)

        # 显示全部128采样点
        n_disp_samples = min(len(sig), 128)
        sig_disp = sig[:n_disp_samples]
        t = np.arange(n_disp_samples) / FS

        # 统一显示为单波形：复数信号取实部（物理可观测的时域波形）
        if np.iscomplexobj(sig_disp):
            waveform = sig_disp.real
        else:
            waveform = sig_disp

        self.ax_wave.plot(t, waveform, 'b-', alpha=0.9, linewidth=1.2)
        # 画零线辅助观察
        self.ax_wave.axhline(y=0, color='gray', alpha=0.3, linewidth=0.5)

        self.ax_wave.set_title(f'{mod_type} 时域波形 (SNR={snr}dB, 128采样点)', fontsize=10)
        self.ax_wave.set_xlabel('时间 (s)'); self.ax_wave.set_ylabel('幅度')
        nfft = len(sig)
        f = np.fft.fftfreq(nfft, 1.0 / FS)
        amplitude = np.abs(np.fft.fft(sig))  # 幅度（非dB）

        # 只显示正频率，聚焦载波附近
        pos_mask = (f >= 0) & (f <= FC * 3)
        f_pos = f[pos_mask]
        amp_pos = amplitude[pos_mask]

        self.ax_spec.plot(f_pos, amp_pos, 'b-', alpha=0.9, linewidth=1.2)
        self.ax_spec.fill_between(f_pos, 0, amp_pos, alpha=0.12, color='blue')
        # 标注载波频率位置
        self.ax_spec.axvline(x=FC, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
        self.ax_spec.annotate(f'载波\n{FC/1e3:.0f}kHz', (FC, max(amp_pos)*0.85),
                               fontsize=6, color='red', alpha=0.6,
                               ha='center')

        self.ax_spec.set_title(f'{mod_type} 幅度谱 (SNR={snr}dB)', fontsize=10)
        self.ax_spec.set_xlabel('频率 (Hz)'); self.ax_spec.set_ylabel('幅度')
        # 处理信号: 对于 RadioML 128 点短信号直接使用全部采样点
        # 对于长信号(数学生成)则按符号速率抽取
        if np.iscomplexobj(sig):
            sig_plot = sig
        else:
            from scipy.signal import hilbert
            sig_plot = hilbert(np.asarray(sig, dtype=float))

        if len(sig_plot) > SPB * 4:
            # 长信号: 按符号周期抽取 (取符号中心点)
            sym_samples = sig_plot[SPB // 2::SPB]
        else:
            # RadioML 短信号 (128点): 使用全部采样点
            sym_samples = sig_plot

        self.ax_const.scatter(sym_samples.real[:500], sym_samples.imag[:500],
                               s=5, alpha=0.6, c='blue')
        self.ax_const.set_title(f'{mod_type} 星座图', fontsize=10)
        self.ax_const.set_xlabel('I'); self.ax_const.set_ylabel('Q')
        self.ax_const.axhline(y=0, color='gray', alpha=0.3)
        self.ax_const.axvline(x=0, color='gray', alpha=0.3)
        self.ax_const.set_aspect('equal')

        self.canvas.draw()

    def _plot_accuracy_curve(self, snr_accuracies):
        """绘制准确率随SNR变化的曲线图"""
        self.ax_acc.clear()
        self.ax_acc.grid(True, alpha=0.3)

        snr_list = sorted(snr_accuracies.keys())
        acc_list = [snr_accuracies[s] for s in snr_list]

        # 画曲线
        self.ax_acc.plot(snr_list, acc_list, 'o-', color='steelblue',
                         linewidth=2, markersize=6, markerfacecolor='white',
                         markeredgewidth=2, markeredgecolor='steelblue')

        # 标注关键点: 90% 参考线
        self.ax_acc.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1, label='90% 目标线')
        self.ax_acc.legend(fontsize=7, loc='lower right')

        # 每个点标注数值
        for snr, acc in zip(snr_list, acc_list):
            self.ax_acc.annotate(f'{acc:.1f}%',
                                 (snr, acc),
                                 textcoords='offset points',
                                 xytext=(0, 10),
                                 ha='center', fontsize=7, color='steelblue')

        # 强制 x 轴按 2dB 步进显示 (匹配 RadioML 2016.10a 数据集)
        if snr_list:
            x_min, x_max = min(snr_list), max(snr_list)
            self.ax_acc.set_xticks(list(range(x_min, x_max + 1, 2)))
            self.ax_acc.set_xlim(x_min - 1, x_max + 1)

        self.ax_acc.set_title('SVM准确率 vs SNR 曲线', fontsize=10)
        self.ax_acc.set_xlabel('SNR (dB)'); self.ax_acc.set_ylabel('准确率 (%)')
        self.ax_acc.set_ylim(0, 110)
        self.canvas.draw()

    def _plot_per_class_accuracy(self, per_type):
        """绘制各类别准确率（保留兼容）"""
        self.ax_acc.clear()
        self.ax_acc.grid(True, alpha=0.3)
        mods = list(per_type.keys())
        accs = [per_type[m]['correct'] / max(per_type[m]['total'], 1) * 100 for m in mods]
        bars = self.ax_acc.bar(mods, accs, color='steelblue', alpha=0.7)
        for bar, acc in zip(bars, accs):
            self.ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                              f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)
        self.ax_acc.set_title('各类别识别准确率', fontsize=10)
        self.ax_acc.set_xlabel('调制类型'); self.ax_acc.set_ylabel('准确率 (%)')
        self.ax_acc.set_ylim(0, 110)
        self.ax_acc.tick_params(axis='x', rotation=45)
        self.canvas.draw()

    def _plot_accuracy_comparison(self, results):
        """绘制模型准确率对比图（SVM单一模型时显示性能指标）"""
        self.ax_acc.clear()
        self.ax_acc.grid(True, alpha=0.3)

        # 获取SVM指标
        svm_metrics = None
        for name in results:
            if hasattr(results[name], 'get'):
                svm_metrics = results[name].get('metrics', results[name])
            else:
                svm_metrics = results[name]
            break

        if svm_metrics and hasattr(svm_metrics, 'get'):
            metrics_dict = {
                '准确率': svm_metrics.get('accuracy', 0) * 100,
                '精确率': svm_metrics.get('precision', 0) * 100,
                '召回率': svm_metrics.get('recall', 0) * 100,
                'F1分数': svm_metrics.get('f1_score', 0) * 100,
            }
            names = list(metrics_dict.keys())
            values = list(metrics_dict.values())
            colors = ['steelblue', 'coral', 'seagreen', 'orange']
            bars = self.ax_acc.bar(names, values, color=colors, alpha=0.7)
            for bar, val in zip(bars, values):
                self.ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                  f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
            self.ax_acc.set_title('SVM模型性能指标', fontsize=10)
            self.ax_acc.set_ylabel('百分比 (%)')
            self.ax_acc.set_ylim(0, 110)

        self.canvas.draw()


def launch_gui(models=None, scalers=None, feature_names=None, label_names=None):
    """启动GUI应用 (支持逐SNR独立多级分类模型)"""
    root = tk.Tk()
    app = ModulationRecognitionGUI(root, models, scalers, feature_names, label_names)
    root.mainloop()
    return app

