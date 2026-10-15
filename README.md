# 基于 AI 的通信信号调制方式识别系统

本项目使用 Python 实现通信信号生成、特征提取、SVM 分类、RadioML 数据加载和 Tkinter 可视化。

## 最终报告的调制类别

训练、数据加载、混淆矩阵与 GUI 统一采用以下标签顺序：

| 数字标签 | 调制类别 |
| --- | --- |
| 0 | BPSK |
| 1 | QPSK |
| 2 | 8PSK |
| 3 | QAM16 |

QAM16 与 16QAM 表示同一种调制。内部使用最终报告和 RadioML 的名称
`QAM16`，数据加载兼容 `16QAM`；不再将 PAM4、GFSK 映射为训练类别。
保留通用信号生成器和候选类别对比脚本中的其他调制方式供扩展实验使用。

现有主入口仍使用逐 SNR 两级 SVM：先区分 PSK/QAM，再区分
BPSK/QPSK/8PSK。此次修改统一调制类别，并未将现有训练架构改成报告中的
多 SNR 混合单层 SVM。数据范围仍为 0~18 dB，步长为 2 dB。

旧的 4ASK/2FSK/BPSK/8PSK 模型不能复用。更新代码后必须重新运行
`python main.py --train`；加载器会拒绝旧标签模型，避免误解数字预测结果。
仓库不包含 RadioML 数据及重新训练后的权重；修改类别本身不代表复现了
报告中的 93.50% 准确率，指标需要在完整数据上重新训练、测试后确认。
现有主入口还包含为曲线单调性调整预测的逻辑，调整后结果不能当作原始
SVM 测试准确率；真实性能应使用调整前的预测计算。

## 运行环境

- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- h5py（读取 HDF5 数据时需要）

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

```bash
# 训练模型并启动 GUI
python main.py

# 仅训练模型
python main.py --train

# 仅启动 GUI
python main.py --gui

# 运行模型对比
python main.py --compare
```

RadioML 数据集需要由用户单独准备。默认数据路径为：

```text
test_dataset/RML2016.10a-main/RML2016.10a-main/2016.10a
```

运行过程中生成的模型、数据库、缓存和测试数据不会提交到仓库。

## 类别回归检查

```bash
python -m unittest test_report_modulations -v
```

测试使用临时数据，覆盖四类 TXT/PKL 加载、16QAM 别名、两级分类标签与概率、
小规模训练流程、旧类别模型拒绝加载，不需要下载完整 RadioML 数据集。
