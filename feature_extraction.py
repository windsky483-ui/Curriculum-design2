"""
=============================================================================
特征提取模块 — 提取信号的时域、频域特征及高阶累积量
实现基于瞬时特征和高阶统计量的调制识别特征集
=============================================================================
参考文献:
[1] A. K. Nandi and E. E. Azzouz, "Algorithms for automatic modulation recognition
    of communication signals," IEEE Trans. Commun., 1998.
[2] A. Swami and B. M. Sadler, "Hierarchical digital modulation classification
    using cumulants," IEEE Trans. Commun., 2000.
=============================================================================
"""
import numpy as np
from scipy.signal import hilbert
from config import CUMULANT_ORDERS, FS

def _analytic(signal):
    """计算信号的解析表示 (Hilbert变换)"""
    # 如果信号是纯实数，做 hilbert
    if np.isrealobj(signal):
        return hilbert(np.asarray(signal, dtype=float))
    return np.asarray(signal)


def _unwrap_phase(phase):
    """相位解卷绕"""
    return np.unwrap(phase)

def instantaneous_features(signal):
    """
    提取瞬时幅度、相位、频率的统计特征
    """
    z = _analytic(signal)

    # 瞬时幅度
    amp = np.abs(z)
    amp_mean = np.mean(amp)
    amp_var = np.var(amp)
    amp_std = np.std(amp)

    # 归一化中心化瞬时幅度
    amp_centered = amp - amp_mean
    amp_norm = amp_centered / max(amp_mean, 1e-10)

    # 瞬时相位
    phase = np.angle(z)
    phase_unwrapped = _unwrap_phase(phase)

    # 瞬时频率 (相位差分, Hz)
    freq_inst = np.diff(phase_unwrapped) * FS / (2 * np.pi)
    freq_inst = np.clip(freq_inst, -FS/2, FS/2)

    # === γ_max: 幅度谱峰 (区分ASK) ===
    amp_centered_norm = amp_centered / max(amp_mean, 1e-10)
    N = len(amp_centered_norm)
    gamma_max = np.max(np.abs(np.fft.fft(amp_centered_norm)) ** 2) / max(N, 1)

    # === sigma_ap & sigma_dp: 相位特征 (区分PSK) ===
    # 使用全采样点而非阈值筛选(恒定阈值在高SNR下对PSK/FSK恒包络信号会全过滤)
    # Circular statistics: 复指数方法天然避免 2π 缠绕问题
    C = np.mean(np.cos(phase))
    S = np.mean(np.sin(phase))
    R = np.sqrt(C**2 + S**2)
    sigma_ap = np.sqrt(-2 * np.log(max(R, 1e-15)))  # circular std (abs)

    # sigma_dp: 先算 circular mean, 再求各点绕回偏差, 确保高SNR下也正确
    circ_mean = np.arctan2(S, C)
    devs = phase - circ_mean
    devs = np.arctan2(np.sin(devs), np.cos(devs))  # wrap to [-π,π]
    sigma_dp = np.std(devs)

    # === P: 低幅度比 (区分ASK) ===
    amp_abs_norm = np.abs(amp_norm)
    low_amp_ratio = np.sum(amp_abs_norm < 0.05) / max(len(amp_abs_norm), 1)

    # === sigma_aa: 归一化幅度绝对标准差 ===
    sigma_aa = np.std(np.abs(amp_norm))

    # === sigma_af: 归一化频率绝对标准差 (区分FSK) ===
    freq_centered = freq_inst - np.mean(freq_inst)
    freq_abs_std = np.std(np.abs(freq_centered))
    freq_abs_mean = np.mean(np.abs(freq_centered))
    if freq_abs_mean > 1e-6:
        sigma_af = freq_abs_std / freq_abs_mean
    else:
        sigma_af = freq_abs_std

    # 幅度高阶统计量
    amp_kurtosis = _kurtosis(amp)
    amp_skewness = _skewness(amp)

    # 频率统计量
    freq_mean = np.mean(freq_inst)
    freq_var = np.var(freq_inst)
    freq_std = np.std(freq_inst)

    features = {
        'amp_mean':     amp_mean,
        'amp_std':      amp_std,
        'amp_var':      amp_var,
        'amp_kurtosis': amp_kurtosis,
        'amp_skewness': amp_skewness,
        'freq_mean':    freq_mean,
        'freq_std':     freq_std,
        'freq_var':     freq_var,
        'gamma_max':    gamma_max,
        'sigma_ap':     sigma_ap,
        'sigma_dp':     sigma_dp,
        'sigma_aa':     sigma_aa,
        'sigma_af':     sigma_af,
        'P':            low_amp_ratio,
    }
    return features

def spectral_features(signal):
    """
    提取频域特征
    参数:
        signal: 输入信号
    返回:
        dict: 频谱特征字典
    """
    # 计算功率谱密度
    f, pxx = _psd(signal)

    # 频谱统计
    total_power = np.sum(pxx)
    pxx_norm = pxx / (total_power + 1e-10)

    # 频谱质心
    spectral_centroid = np.sum(f * pxx_norm)

    # 频谱带宽 (均方根带宽)
    spectral_bandwidth = np.sqrt(
        np.sum(((f - spectral_centroid) ** 2) * pxx_norm)
    )

    # 频谱峰度
    spectral_kurtosis = _kurtosis(pxx)

    # 频谱偏度
    spectral_skewness = _skewness(pxx)

    # 峰值频率
    peak_freq = f[np.argmax(pxx)]

    # 对称性度量
    half = len(f) // 2
    lower_energy = np.sum(pxx[:half])
    upper_energy = np.sum(pxx[half:])
    symmetry = min(lower_energy, upper_energy) / (max(lower_energy, upper_energy) + 1e-10)

    # 频谱平坦度
    spectral_flatness = np.exp(np.mean(np.log(pxx_norm + 1e-10))) / (np.mean(pxx_norm) + 1e-10)

    # 峰均比
    peak_to_avg_ratio = np.max(pxx) / (np.mean(pxx) + 1e-10)

    features = {
        'spectral_centroid':   spectral_centroid,
        'spectral_bandwidth':  spectral_bandwidth,
        'spectral_kurtosis':   spectral_kurtosis,
        'spectral_skewness':   spectral_skewness,
        'peak_freq':           peak_freq,
        'symmetry':            symmetry,
        'spectral_flatness':   spectral_flatness,
        'peak_to_avg_ratio':   peak_to_avg_ratio,
    }
    return features

def cumulant_features(signal):
    """
    提取信号的高阶累积量 (2阶、4阶、6阶、8阶)
    累积量对高斯噪声不敏感，是调制识别的关键特征

    对于零均值复平稳信号 X:
    - C20 = cum(X, X)
    - C21 = cum(X, conj(X))
    - C40 = cum(X, X, X, X)
    - C41 = cum(X, X, X, conj(X))
    - C42 = cum(X, X, conj(X), conj(X))
    - C60, C61, C62, C63 (6阶)
    - C80 (8阶)
    参数:
        signal: 输入信号 (实数或复数)
    返回:
        dict: 累积量特征字典
    """
    # 先变换到基带 (去除载波)
    x = np.asarray(signal)
    if np.isrealobj(x):
        z = hilbert(x)
    else:
        z = x.copy()

    # 中心化
    z = z - np.mean(z)

    N = len(z)

    # === 2阶累积量 ===
    C20 = np.mean(z ** 2)
    C21 = np.mean(np.abs(z) ** 2)  # = 平均功率

    # === 4阶累积量 ===
    M20 = C20
    M21 = C21
    M40 = np.mean(z ** 4)
    M41 = np.mean((z ** 3) * np.conj(z))
    M42 = np.mean((z ** 2) * np.abs(z) ** 2)

    # 零均值: C40 = M40 - 3*M20^2
    C40 = M40 - 3 * M20 ** 2
    # C41 = M41 - 3*M20*M21
    C41 = M41 - 3 * M20 * M21
    # C42 = M42 - |M20|^2 - 2*M21^2
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2

    # === 6阶累积量 ===
    M60 = np.mean(z ** 6)
    M61 = np.mean((z ** 5) * np.conj(z))
    M62 = np.mean((z ** 4) * np.abs(z) ** 2)
    M63 = np.mean((z ** 3) * np.abs(z) ** 4)

    # C60 = M60 - 15*M40*M20 + 30*M20^3  (简化，准确公式需更多项)
    C60 = M60 - 15 * M40 * M20 + 30 * M20 ** 3
    # C61 = M61 - 5*M41*M20 - 10*M40*M21 + 30*M20^2*M21
    C61 = M61 - 5 * M41 * M20 - 10 * M40 * M21 + 30 * M20 ** 2 * M21
    # C62 = M62 - 6*M42*M20 - 8*M41*M21 - M40*M42 + 6*M20^2*C42 + 24*M20*M21^2
    C62 = M62 - 6 * M42 * M20 - 8 * M41 * M21 - M40 * M42 + 6 * M20 ** 2 * C42 + 24 * M20 * M21 ** 2
    # C63 = M63 - 9*M42*M21 - ... (简化)
    C63 = M63 - 9 * M42 * M21 + 12 * M21 ** 3

    # === 8阶累积量 ===
    M80 = np.mean(z ** 8)
    C80 = M80 - 28 * M60 * M20 - 35 * M40 ** 2 + 420 * M40 * M20 ** 2 - 630 * M20 ** 4

    features = {}
    for order in CUMULANT_ORDERS:
        key = f'C{order}'
        if key in locals():
            val = locals()[key]
            features[key] = np.real(val) if np.isrealobj(val) else np.abs(val)

    # 归一化 (相对于 C21 即功率)
    power = C21
    if power > 1e-10:
        for k in list(features.keys()):
            # 根据累积量阶数归一化: C_pq / C21^(p/2)
            p = int(k[1])  # e.g. C42 -> p=4
            features[k] = features[k] / (power ** (p / 2))

    return features

def extract_features(signal):
    """
    提取信号的全部特征
    参数:
        signal: 输入信号
    返回:
        features: 特征向量 (1D numpy array)
        feature_names: 特征名称列表
    """
    f_inst = instantaneous_features(signal)
    f_spec = spectral_features(signal)
    f_cum = cumulant_features(signal)

    all_features = {}
    all_features.update(f_inst)
    all_features.update(f_spec)
    all_features.update(f_cum)

    feature_names = list(all_features.keys())
    feature_vector = np.array([all_features[k] for k in feature_names], dtype=float)

    # 检查并替换 NaN/Inf
    # NaN/Inf → 0: 数值异常的特征在 StandardScaler 后自动归零
    # 不能用 1e6 等极端值，会在 StandardScaler 中主导整个特征列
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector, feature_names


def extract_features_batch(signals):
    """
    批量提取信号特征
    参数:
        signals: 信号列表
    返回:
        X: 特征矩阵 (n_samples x n_features)
        feature_names: 特征名称列表
    """
    feature_list = []
    feature_names = None

    for sig in signals:
        fv, fn = extract_features(sig)
        feature_list.append(fv)
        if feature_names is None:
            feature_names = fn

    X = np.array(feature_list)
    return X, feature_names

def _psd(signal, nperseg=None):
    """计算功率谱密度（自适应信号长度）"""
    if nperseg is None:
        nperseg = min(256, 2 ** int(np.log2(len(signal))))
    f, pxx = sp_periodogram(signal, fs=FS, nfft=nperseg, return_onesided=True)
    return np.asarray(f), np.asarray(pxx)


def _kurtosis(x):
    """计算峰度 (超值峰度)"""
    x = np.asarray(x)
    x = x - np.mean(x)
    m4 = np.mean(x ** 4)
    m2 = np.mean(x ** 2)
    if m2 < 1e-10:
        return 0.0
    return m4 / (m2 ** 2) - 3


def _skewness(x):
    """计算偏度"""
    x = np.asarray(x)
    x = x - np.mean(x)
    m3 = np.mean(x ** 3)
    m2 = np.mean(x ** 2)
    if m2 < 1e-10:
        return 0.0
    return m3 / (m2 ** 1.5)


def sp_periodogram(x, fs=1.0, nfft=256, return_onesided=True):
    """简易周期图功率谱估计"""
    x = np.asarray(x)
    if len(x) < nfft:
        nfft = 2 ** int(np.log2(len(x)))
    segments = len(x) // nfft
    if segments < 2:
        segments = 1
        nfft = len(x)
    pxx_sum = np.zeros(nfft)
    for i in range(segments):
        seg = x[i*nfft:(i+1)*nfft]
        if len(seg) < nfft:
            break
        seg = seg * np.hanning(nfft)
        pxx_sum += np.abs(np.fft.fft(seg, nfft)) ** 2
    pxx = pxx_sum / segments
    if return_onesided:
        pxx = pxx[:nfft//2]
        pxx[1:-1] *= 2
    return np.fft.fftfreq(nfft, 1.0/fs)[:nfft//2], pxx

