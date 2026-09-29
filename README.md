# DSCA 复现

论文：**DSCA: A Digital Subtraction Angiography Sequence Dataset and Spatio-Temporal
Model for Cerebral Artery Segmentation**（IEEE TMI 2025），网络名 **DSANet**。

---



---

## 1. 目录结构

```
F:\Projects\DSCA\
├── models\                     模型（L1 / L2 / L3）
│   ├── blocks.py               ConvBlock / MaxAvgPoolT / Up —— 论文 Fig.3 的图例单元
│   ├── temporalformer.py       L1  TemporalFormer
│   ├── stf.py                  L2  STF 时空融合
│   ├── encoder.py              L3  TEB / SEB 双编码分支
│   ├── decoder.py              L3  解码器 + 深监督
│   ├── dsanet.py               L3  总装
│   └── losses.py               CE + Dice + 深监督
├── data\
│   ├── prepare.py              P0  原始数据 -> .npz
│   └── dataset.py              Dataset + 数据增强
├── tests\                      验收测试
│   ├── test_temporalformer.py  L1  20 项判据
│   ├── test_stf.py             L2  11 项判据
│   └── test_dsanet.py          L3  13 项判据
├── metrics.py                  Dice 指标
├── inference.py                推理：镜像 TTA
├── train.py                    训练
├── evaluate.py                 评估
├── diagnose.py                 数据/优化器/可学性诊断（data / scale / overfit ...）
├── viz_failures.py             失败样本可视化（8 帧 + MinIP/MaxIP + 直方图）
├── plot_learning_curve.py      学习曲线出图（滑动平均 + 噪声带 + 脱困点）
├── fetch_histories.sh          
└── README.md                   本文件
```

数据（**不在 C 盘，符号链接也没有**）：

```
E:\Datasets\DSCA\
├── DSA_public\                 原始数据
└── processed\                  预处理产物
    ├── train\*.npz             180 例
    ├── test\*.npz              44 例
    ├── folds.json              5 折划分
    └── report.json             数据自检报告
```

---

## 2. 环境



```
E:\Anaconda\envs\dsca\python.exe
```

关键版本：torch 2.14.0+cu126、einops、scipy、scikit-image、pydicom、nibabel、SimpleITK。


---



所有命令的工作目录都是 `F:\Projects\DSCA`。

### 第 1 步：数据准备（约 1.5 分钟）

```bash
E:\Anaconda\envs\dsca\python.exe data\prepare.py
```

产物：`E:\Datasets\DSCA\processed\` 下的 224 个 `.npz` + `folds.json` + `report.json`。

跑完请**看一眼终端输出**，确认这四行：

- `方向处置分布 : {'axis_T': 180}` —— 🔴 **180 例全部是转置类处置，这是对的**：
  DSCA 的 label nii 一律按 `(cols, rows)` 存，跟 DICOM 的 `(rows, cols)` 差一次转置。
  早期版本这里会是 `{'label_ok': 158, 'label_transposed': 22}` —— 那说明**又退回
  形状判据了**，158 例正方形样本的标签会错位 90°（详见 §5.5，代价是 test Dice
  被锁死在 0.1675）
- `🔴 与旧形状判据不同 : 158 例` —— 这就是旧判据静默放行的样本数
- `对齐 AUC : 均值 ~0.96 / 最小 ~0.83 / <0.85 的 0~1 例`
- `像素类别占比 : 背景 ~95.5% / BV ~3.7% / MAT ~0.9%`

**跑完必须做一次方向验收**（几秒，只看 npz 的 label 与 minip）：

```bash
E:\Anaconda\envs\dsca\python.exe tmp\audit_label_align.py --proc "E:\Datasets\DSCA\processed"
```

判据：**「方向确定 224/224」**（即每例所选方向对转置的领先 ≥ 0.15）。出现
`⚠️ 方向存疑` 就**不要开始训练**，先查那一例的 DICOM 与 nii。

> 换过折划分策略、或想重新划分时，不用重跑预处理：
> `python data\prepare.py --folds_only`

### 第 2 步：训练

```bash

E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 3 --iters-per-epoch 20 --out runs\smoke

E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 40 --iters-per-epoch 100 --val-every 10 --lr-total-epochs 500 --out runs\probe_sgd_lralign
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 40 --iters-per-epoch 100 --val-every 10 --lr-total-epochs 500 --optimizer adam --lr 1e-3 --out runs\probe_adam_lralign

:: 第一阶段：1 折 100 epoch（约 2.5 小时）
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 100 --out runs\fold0_100ep

:: 完整：1 折 500 epoch（约 12 小时）
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 500 --out runs\fold0_full

:: 断点续训（2026-09-26 加；之前没有 resume，中断只能从 epoch 0 重来）
::   'auto' = 读 <out>/last.pth。会恢复 权重+优化器状态+epoch+best+history，
::   lr 曲线按**绝对 epoch** 继续，不会从头重来
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 500 --out runs\fold0_full --resume auto

:: 小样本过拟合探针（查容量/优化/梯度路径类问题）
::   🔴 它**查不出轴序/标签错位类 bug** —— 见 §5.5 第 3 条
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --limit-train 5 --epochs 150 ^
    --iters-per-epoch 20 --lr-total-epochs 500 --optimizer adam --lr 1e-3 ^
    --val-every 50 --out runs\probe_overfit5

:: 5 折全部（约 2.5 天，建议挂着跑）
for f in 0 1 2 3 4; do
  E:\Anaconda\envs\dsca\python.exe train.py --fold $f --epochs 500 --out runs\fold${f}_full
done
```

> 🔴 **2026-09-26 之前的所有 5 折 / lc / 对比 run 都在错位数据上训练，结论作废。**
> 修好轴序后（§5.5）需要在新 `processed` 上重跑。折划分与旧版逐字节一致，
> 所以**新旧同名 run 可以直接对比**（如 `runs/f0_axis_fixed` vs 旧 `f0`）。

产物在 `--out` 目录里：

| 文件 | 内容 |
|---|---|
| `best.pth` / `last.pth` | 最佳 / 最后一轮的权重 |
| `train.log` | 完整训练日志（含每次验证的 Dice） |
| `history.json` | 逐 epoch 的 loss / lr / Dice |

### 第 3 步：评估

```bash
:: 训练时的验证折（默认带镜像 TTA）
E:\Anaconda\envs\dsca\python.exe evaluate.py --ckpt runs\fold0_100ep\best.pth --split val --fold 0

:: 官方独立测试集（44 例）
E:\Anaconda\envs\dsca\python.exe evaluate.py --ckpt runs\fold0_100ep\best.pth --split test

:: 关掉 TTA 对比提升幅度
E:\Anaconda\envs\dsca\python.exe evaluate.py --ckpt runs\fold0_100ep\best.pth --split test --no-tta
```

产物：`eval_val.json` / `eval_test.json`，含总体 Dice 与**逐例** Dice。

### 第 4 步：出问题了先跑诊断（不要直接怀疑网络）

```bash
:: 数据侧：强度分布 / 跨样本尺度差 / 图像-标签对齐 / 类别占比
E:\Anaconda\envs\dsca\python.exe diagnose.py --stage data

:: 判别实验：前景学不出来是"损失的锅"还是"输入的锅"
::   （含 oracle 阈值上界 + 小 U-Net 在 归一化×损失 网格上的过拟合对照）
E:\Anaconda\envs\dsca\python.exe diagnose.py --stage scale

:: 过拟合测试：单张 / 多张一起。多张一起过拟合是检验"跨样本一致性"的直接办法
E:\Anaconda\envs\dsca\python.exe diagnose.py --stage overfit --cores CXL_AP_38,FXR_AP_25 --grad-checkpoint
```

---

## 4. 验收测试

```bash
E:\Anaconda\envs\dsca\python.exe tests\test_temporalformer.py
E:\Anaconda\envs\dsca\python.exe tests\test_stf.py
E:\Anaconda\envs\dsca\python.exe tests\test_dsanet.py
```

三层合计 44 项判据，目前 **全部通过**。其中最重要的一条是
`test_temporalformer.py` 的 **T1 手工索引对拍** ——
TF 的维度链上有 4 次 reshape 往返，轴序写错**不会报错**，shape 全对、
loss 照降，只是帧与像素的对应关系悄悄错位，最后表现为"能训但精度偏低"。
唯一可靠的发现手段就是手工索引对拍。

---

## 5.相关配置

| 项 | 论文（IV-A 节） | 本机实现 | 原因 |
|---|---|---|---|
| 骨干 | nnUNet 风格，瓶颈 c=512 | 同（chs 32→512，4 次下采样→32×32） | — |
| 优化器 | SGD, momentum 0.99, wd 3e-5 | 同 | — |
| 学习率 | 0.01 + polynomial decay | 同（power 0.9） | 论文没给 power |
| 迭代 | 500 epoch × 100 iter | 同 | — |
| batch | 2 @ 512×512 | **1 × 梯度累积 2 步** | 见下 |
| AMP | — | **开** | 必须 |
| 梯度检查点 | — | **开** | 必须 |
| 5 折 | 180 例 4 训 1 验 | 同（按病人分组，每折 144/36） | — |
| 后处理 | 不做 | 不做 | — |
| **归一化** | 论文**完全没写** | **per-image z-score** | 见 §5.1，这条是决定性的 |
| Dice 是否算背景 | 论文没写 | **不算** | 背景占 95.5%，算进去前景梯度会被淹 |

