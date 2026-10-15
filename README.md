# 基于 AI 的通信信号调制方式识别系统

本项目使用 Python 实现通信信号生成、特征提取、SVM 分类、RadioML 数据加载和 Tkinter 可视化。

## 报告的调制类别

训练、数据加载、混淆矩阵与 GUI 统一采用以下标签顺序：

| 数字标签 | 调制类别 |
| --- | --- |
| 0 | BPSK |
| 1 | QPSK |
| 2 | 8PSK |
| 3 | QAM16 |





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
