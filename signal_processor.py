"""
=============================================================================
信号处理模块 — 基于 RadioML 2016.10a 真实数据集
不生成信号，从数据集中加载和预处理信号。
=============================================================================
数据集: RadioML 2016.10a (txt版)
- 格式: 每(调制,SNR)一个txt文件, 每行一个信号, 128个空格分隔的(real+imagj)复数
- SNR范围: -20 ~ 18 dB (2 dB步进), 本系统使用 0 ~ 18 dB
- 信号长度: 128 复采样点
- 调制映射: BPSK→BPSK, 8PSK→8PSK, PAM4→4ASK, GFSK→2FSK
=============================================================================
"""
import numpy as np
import os


# RadioML 原始调制名 → 本系统调制名
NAME_MAP = {
    'BPSK': 'BPSK',
    '8PSK': '8PSK',
    'PAM4': '4ASK',
    'GFSK': '2FSK',
}

# 本系统支持的 4 类调制
SUPPORTED_MODS = ('4ASK', '2FSK', 'BPSK', '8PSK')

# SNR 范围 (本系统使用)
SNR_MIN = 0
SNR_MAX = 18


class SignalProcessor:
    """RadioML 2016.10a 信号处理器

    用法:
        sp = SignalProcessor()
        sig, mod, snr = sp.get_signal('BPSK', snr_db=10)
        sigs, mods, snrs = sp.get_batch('4ASK', snr_db=5, n=100)
        print(sp.dataset_info())
    """

    def __init__(self, dataset_path=None):
        """
        参数:
            dataset_path: 数据集目录路径, None 则自动搜索 test_dataset/
        """
        self._data = {}          # {(mod, snr): ndarray of complex signals}
        self._mod_types = []     # 本系统调制类型列表
        self._snr_levels = []    # SNR 列表 (0~18, 2dB步进)
        self._sample_counts = {} # {(mod, snr): 可用样本数}
        self._total_samples = 0

        if dataset_path is None:
            dataset_path = self._find_dataset()

        if dataset_path and os.path.exists(dataset_path):
            self._load(dataset_path)
        else:
            raise FileNotFoundError(
                f'未找到 RadioML 2016.10a 数据集。\n'
                f'请将 RML2016.10a-main 文件夹放到 test_dataset/ 目录下。\n'
                f'搜索路径: {dataset_path}'
            )

    def get_signal(self, mod_type, snr_db=None):
        """随机获取一个信号。

        参数:
            mod_type: '4ASK'|'2FSK'|'BPSK'|'8PSK'
            snr_db:   SNR值 (0~18, 2dB步进), None则随机选取
        返回:
            (signal, mod_type, actual_snr)
            signal: 256点复信号 (已功率归一化)
            actual_snr: 信号的实际 SNR (dB) — 始终来自数据集真实值
        """
        if snr_db is None:
            snr_db = np.random.choice(self._snr_levels)
        else:
            snr_db = self._nearest_snr(snr_db)

        key = (mod_type, snr_db)
        if key not in self._data:
            available = [k for k in self._data if k[0] == mod_type]
            if not available:
                raise ValueError(f'调制类型 {mod_type} 不在数据集中')
            key = available[np.random.randint(len(available))]

        idx = np.random.randint(len(self._data[key]))
        actual_snr = key[1]  # 始终返回数据集中的真实 SNR
        return self._data[key][idx].copy(), mod_type, actual_snr

    def get_all_signals(self, mod_type, snr_db):
        """获取特定(调制, SNR)的全部唯一信号 (无放回采样).

        返回所有可用信号的副本，用于构建无数据泄漏的训练/测试集。
        参数:
            mod_type: 调制类型
            snr_db:   SNR值
        返回:
            ndarray of shape (n_samples,) 每个元素为256点复信号
        """
        snr_db = self._nearest_snr(snr_db)
        key = (mod_type, snr_db)
        if key not in self._data:
            raise ValueError(f'无此组合: {mod_type} @ SNR={snr_db}dB')
        return self._data[key].copy()

    def get_batch(self, mod_type, snr_db, n):
        """批量获取信号 (可重复采样)。

        参数:
            mod_type: 调制类型
            snr_db:   SNR值
            n:        需要的样本数
        返回:
            signals:  list of 复信号
        """
        snr_db = self._nearest_snr(snr_db)
        key = (mod_type, snr_db)
        if key not in self._data:
            raise ValueError(f'无此组合: {mod_type} @ SNR={snr_db}dB')

        pool = self._data[key]
        indices = np.random.randint(0, len(pool), n)
        return [pool[i].copy() for i in indices]

    def get_all_snr_levels(self):
        """返回数据集中所有 SNR 等级 (0~18, 2dB步进)"""
        return self._snr_levels.copy()

    def get_available_mod_types(self):
        """返回可用的调制类型列表"""
        return self._mod_types.copy()

    def dataset_info(self):
        """返回数据集概况"""
        lines = [
            '=' * 50,
            '  RadioML 2016.10a 数据集概况',
            '=' * 50,
            f'  调制类型: {len(self._mod_types)} 类 — {self._mod_types}',
            f'  SNR 范围: {self._snr_levels[0]} ~ {self._snr_levels[-1]} dB '
            f'({len(self._snr_levels)} 个等级)',
            f'  信号长度: 128 复采样点',
            f'  总可用样本: {self._total_samples:,}',
            '─' * 50,
        ]
        for mod in self._mod_types:
            cnt = sum(c for (m, _), c in self._sample_counts.items() if m == mod)
            lines.append(f'    {mod}: {cnt:,} 样本')
        lines.append('=' * 50)
        return '\n'.join(lines)

    @staticmethod
    def _parse_complex(token):
        """解析形如 '(real+imagj)' 或 '(real-imagj)' 的复数文本."""
        token = token.strip('()')
        # Python 内置 complex() 可直接解析 'real+imagj' 格式(含科学记数法)
        return complex(token)

    def _find_dataset(self):
        """自动查找数据集目录"""
        base = os.path.dirname(os.path.abspath(__file__))
        td = os.path.join(base, 'test_dataset')

        # 搜索包含 2016.10a txt 文件的目录
        if os.path.isdir(td):
            for root, dirs, files in os.walk(td):
                # 查找包含 "BPSK 0.txt" 这类文件的目录
                txt_files = [f for f in files if f.endswith('.txt')]
                if txt_files and any('BPSK' in f for f in txt_files):
                    return root

        # 默认路径
        return os.path.join(td, 'RML2016.10a-main', 'RML2016.10a-main', '2016.10a')

    def _load(self, data_dir):
        """加载并索引数据集 (从txt文件逐行解析)."""
        print(f'[SignalProcessor] 加载数据集: {os.path.basename(data_dir)}')

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'数据集目录不存在: {data_dir}')

        # 列出所有txt文件
        txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if not txt_files:
            raise FileNotFoundError(f'目录中无txt文件: {data_dir}')

        # 第一遍: 统计可用的调制类型和SNR等级
        mod_set = set()
        snr_set = set()
        valid_files = []

        for filename in txt_files:
            name_no_ext = filename.replace('.txt', '')
            # 文件名格式: "{MODULATION} {SNR}" 如 "BPSK 0", "PAM4 10", "GFSK -2"
            parts = name_no_ext.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            mod_name, snr_str = parts
            try:
                snr = int(snr_str)
            except ValueError:
                continue

            mapped = NAME_MAP.get(mod_name)
            if mapped is None or mapped not in SUPPORTED_MODS:
                continue
            if snr < SNR_MIN or snr > SNR_MAX:
                continue

            mod_set.add(mapped)
            snr_set.add(snr)
            valid_files.append((filename, mapped, snr))

        self._mod_types = sorted(mod_set)
        self._snr_levels = sorted(snr_set)

        # 第二遍: 加载每个文件的信号
        for filename, mapped_mod, snr in valid_files:
            filepath = os.path.join(data_dir, filename)
            samples = self._load_txt_file(filepath)
            if len(samples) > 0:
                self._data[(mapped_mod, snr)] = samples
                self._sample_counts[(mapped_mod, snr)] = len(samples)
                self._total_samples += len(samples)

        print(f'[SignalProcessor] 加载完成: {self._total_samples} 个信号, '
              f'{len(self._mod_types)} 类, {len(self._snr_levels)} SNR等级')

    def _load_txt_file(self, filepath):
        """加载单个txt文件: 每行一个信号, 256个空格分隔的复数.

        返回:
            numpy array of shape (n_lines,) 每个元素为256点复信号
        """
        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue

                # 解析所有复数token
                sig = np.array([self._parse_complex(t) for t in tokens], dtype=complex)

                # 功率归一化
                power = np.mean(np.abs(sig) ** 2)
                if power > 1e-10:
                    sig = sig / np.sqrt(power)

                # DC 去除
                sig = sig - np.mean(sig)

                samples.append(sig)

        return np.array(samples)

    def _nearest_snr(self, snr_db):
        """取最近的合法 SNR 值"""
        idx = np.argmin(np.abs(np.array(self._snr_levels) - snr_db))
        return self._snr_levels[idx]

_processor = None


def _get_processor():
    global _processor
    if _processor is None:
        _processor = SignalProcessor()
    return _processor


def get_signal(mod_type, snr_db=None):
    """便捷函数: 获取单个信号"""
    return _get_processor().get_signal(mod_type, snr_db)


def get_batch(mod_type, snr_db, n):
    """便捷函数: 批量获取信号"""
    return _get_processor().get_batch(mod_type, snr_db, n)


def get_all_snr_levels():
    """便捷函数: 获取 SNR 列表"""
    return _get_processor().get_all_snr_levels()


def get_available_mod_types():
    """便捷函数: 获取调制类型列表"""
    return _get_processor().get_available_mod_types()


def dataset_info():
    """便捷函数: 打印数据集概况"""
    return _get_processor().dataset_info()

