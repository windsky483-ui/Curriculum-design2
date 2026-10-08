"""
=============================================================================
信号生成模块 — 生成多种数字调制信号 (ASK/FSK/PSK/QAM) + AWGN 噪声
=============================================================================
"""
import numpy as np
from scipy import signal as sp_signal
from config import FS, FC, RS, N_SYMBOLS, SPB


def _gen_sym_seq(n, bits_per_sym):
    """生成随机符号序列 [0, M-1]"""
    return np.random.randint(0, 2**bits_per_sym, n)


def _pulse_shape(symbols, spb):
    """矩形脉冲成型：每个符号重复 spb 次"""
    return np.repeat(symbols, spb)


def _add_awgn(sig, snr_db):
    """
    添加加性高斯白噪声 (AWGN)
    参数:
        sig: 原始信号
        snr_db: 信噪比 (dB)
    返回:
        加噪后的信号
    """
    if snr_db is None or snr_db >= 100:
        return sig.copy()
    sig_power = np.mean(np.abs(sig) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
    )
    return sig + noise


def _time_base(n_sym):
    """生成时间轴"""
    t = np.arange(n_sym * SPB) / FS
    return t

def generate_ask(M=2, snr_db=15, n_symbols=N_SYMBOLS):
    """
    生成 M-ASK 调制信号
    M=2: 2ASK (OOK), M=4: 4ASK
    """
    bits_per_sym = int(np.log2(M))
    symbols = _gen_sym_seq(n_symbols, bits_per_sym)  # [0, M-1]
    levels = 2 * symbols / (M - 1) - 1               # 归一化到 [-1, 1]
    baseband = _pulse_shape(levels, SPB)
    t = _time_base(n_symbols)
    carrier = np.cos(2 * np.pi * FC * t)
    modulated = baseband * carrier
    rx_signal = _add_awgn(modulated, snr_db)
    return rx_signal.real, symbols, n_symbols

def generate_fsk(M=2, snr_db=15, n_symbols=N_SYMBOLS):
    """
    生成 M-FSK 调制信号
    M=2: 2FSK (BFSK), M=4: 4FSK
    """
    bits_per_sym = int(np.log2(M))
    symbols = _gen_sym_seq(n_symbols, bits_per_sym)
    t = _time_base(n_symbols)
    delta_f = RS * 2  # 频偏
    phase = 0.0
    modulated = np.zeros(len(t), dtype=complex)
    for i, sym in enumerate(symbols):
        freq_offset = (2 * sym - (M - 1)) * delta_f
        idx_start = i * SPB
        idx_end = idx_start + SPB
        t_ = t[idx_start:idx_end]
        # 保持相位连续性
        arg = 2 * np.pi * (FC + freq_offset) * t_ + phase
        phase = arg[-1]
        modulated[idx_start:idx_end] = np.exp(1j * arg)
    rx_signal = _add_awgn(modulated, snr_db)
    return rx_signal, symbols, n_symbols

def generate_psk(M=2, snr_db=15, n_symbols=N_SYMBOLS):
    """
    生成 M-PSK 调制信号
    M=2: BPSK, M=4: QPSK, M=8: 8PSK
    """
    bits_per_sym = int(np.log2(M))
    symbols = _gen_sym_seq(n_symbols, bits_per_sym)
    phases = 2 * np.pi * symbols / M
    baseband = np.exp(1j * phases)
    shaped = _pulse_shape(baseband, SPB)
    t = _time_base(n_symbols)
    carrier = np.exp(1j * 2 * np.pi * FC * t)
    modulated = shaped * carrier
    rx_signal = _add_awgn(modulated, snr_db)
    return rx_signal, symbols, n_symbols

def generate_qam(M=16, snr_db=15, n_symbols=N_SYMBOLS):
    """
    生成 M-QAM 调制信号 (方形星座图)
    M=16: 16QAM, M=64: 64QAM
    """
    order = int(np.sqrt(M))  # 4 for 16QAM, 8 for 64QAM
    symbols_i = np.random.randint(0, order, n_symbols)
    symbols_q = np.random.randint(0, order, n_symbols)
    # 映射到归一化星座图
    I = 2 * symbols_i / (order - 1) - 1
    Q = 2 * symbols_q / (order - 1) - 1
    symbols = symbols_i * order + symbols_q  # 复合符号ID
    baseband = I + 1j * Q
    shaped = _pulse_shape(baseband, SPB)
    t = _time_base(n_symbols)
    carrier = np.exp(1j * 2 * np.pi * FC * t)
    modulated = shaped * carrier
    # 能量归一化
    modulated /= np.sqrt(np.mean(np.abs(modulated) ** 2))
    rx_signal = _add_awgn(modulated, snr_db)
    symbols = symbols_i * order + symbols_q
    return rx_signal, symbols, n_symbols

GENERATORS = {
    '2ASK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_ask(M=2, snr_db=snr_db, n_symbols=n_sym),
    '4ASK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_ask(M=4, snr_db=snr_db, n_symbols=n_sym),
    '2FSK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_fsk(M=2, snr_db=snr_db, n_symbols=n_sym),
    '4FSK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_fsk(M=4, snr_db=snr_db, n_symbols=n_sym),
    'BPSK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_psk(M=2, snr_db=snr_db, n_symbols=n_sym),
    'QPSK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_psk(M=4, snr_db=snr_db, n_symbols=n_sym),
    '8PSK':  lambda snr_db=15, n_sym=N_SYMBOLS: generate_psk(M=8, snr_db=snr_db, n_symbols=n_sym),
    '16QAM': lambda snr_db=15, n_sym=N_SYMBOLS: generate_qam(M=16, snr_db=snr_db, n_symbols=n_sym),
    '64QAM': lambda snr_db=15, n_sym=N_SYMBOLS: generate_qam(M=64, snr_db=snr_db, n_symbols=n_sym),
}


def generate_signal(mod_type, snr_db=15, n_symbols=N_SYMBOLS):
    """
    根据调制类型生成信号
    参数:
        mod_type: 调制类型字符串，如 'BPSK', '16QAM'
        snr_db:   信噪比 (dB)
        n_symbols: 符号数
    返回:
        rx_signal: 接收到的复基带/带通信号
        symbols:   原始符号序列
        n_symbols: 符号数
    """
    gen = GENERATORS.get(mod_type)
    if gen is None:
        raise ValueError(f'不支持的调制类型: {mod_type}')
    return gen(snr_db=snr_db, n_sym=n_symbols)

