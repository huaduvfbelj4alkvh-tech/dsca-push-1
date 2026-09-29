# DSCA 复现（按论文从零实现）

论文：**DSCA: A Digital Subtraction Angiography Sequence Dataset and Spatio-Temporal
Model for Cerebral Artery Segmentation**（IEEE TMI 2025），网络名 **DSANet**。

---

## 0. 先读这一条：这是什么性质的复现

官方仓库 `jiongzhang-john/DSCA` **只有 `README.md` 和两张图，没有任何 `.py`**
（`language: None`、0 个 release、1 个 branch，已用 GitHub API 核实）。

所以本仓库是 **"基于论文从零实现"**，不是"跑通别人的代码"。

> ⚠️ 汇报、简历、答辩里请如实写「基于论文实现」。写「复现开源代码」是事实错误，
> 被问一句 `forward` 里某个模块怎么写的就会露。

> 🔴 **先看 §5.5**：2026-09-26 定位到 P0 数据管线的轴序 bug，**87.5% 的样本标签
> 被静默转置、与图像整体错位 90°**。本文件此前所有基于 5 折 test Dice 的结论
> （"瓶颈是泛化能力"、学习曲线、分辨率假设）**全部作废，必须在新数据上重跑**。

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
│   ├── prepare.py              P0  原始数据 -> .npz（含自检报告、5 折划分）
│   └── dataset.py              Dataset + 数据增强
├── tests\                      验收测试（跑通了才算这一层做完）
│   ├── test_temporalformer.py  L1  20 项判据
│   ├── test_stf.py             L2  11 项判据
│   └── test_dsanet.py          L3  13 项判据（含端到端显存实测）
├── metrics.py                  Dice 指标（含缺失类别的处理规则）
├── inference.py                推理：镜像 TTA
├── train.py                    训练
├── evaluate.py                 评估
├── diagnose.py                 数据/优化器/可学性诊断（data / scale / overfit ...）
├── viz_failures.py             失败样本可视化（8 帧 + MinIP/MaxIP + 直方图）
├── plot_learning_curve.py      学习曲线出图（滑动平均 + 噪声带 + 脱困点）
├── fetch_histories.sh          只把集群的 history.json 拉回本机（供本机出图）
└── README.md                   本文件
```

数据（**不在 C 盘，符号链接也没有**）：

```
E:\Datasets\DSCA\
├── DSA_public\                 原始数据（Zenodo 11255024，已获批下载）
└── processed\                  预处理产物（约 350 MiB）
    ├── train\*.npz             180 例
    ├── test\*.npz              44 例
    ├── folds.json              5 折划分
    └── report.json             数据自检报告
```

---

## 2. 环境

用**已经建好的** conda 环境 `dsca`（不要新建，也不要复用其它环境）：

```
E:\Anaconda\envs\dsca\python.exe
```

关键版本：torch 2.14.0+cu126、einops、scipy、scikit-image、pydicom、nibabel、SimpleITK。

在 PyCharm 里打开 `F:\Projects\DSCA` 时，解释器选 **Python 3.11 (DSCA)**
（不要用 PyCharm 自动建的 `.venv`）。

---

## 3. 三步跑起来

> 📘 **逐条命令的完整手册见 [`跑通手册.md`](跑通手册.md)**（本机生成/验收 → 集群传数据/选卡/训练/巡检/续训 → 评估 → 取回 → 出图 → 记录更新，含每步的 Pass/Fail 判据与常见报错对症表）。
> 下面是精简版。

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
:: 冒烟（约 1 分钟，只为确认能跑通）
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 3 --iters-per-epoch 20 --out runs\smoke

:: 先跑短周期看 Dice 有没有起来 —— 强烈建议先做这一步（约 1 小时）
::   🔴 必须加 --lr-total-epochs 500！否则 lr 曲线被压缩到 40 轮、
::      末期只剩 3.6%，测出来的 Dice 是"油门被松掉"的结果（见 §5.3）
::   默认 SGD（忠于论文）；若前景 Dice 一直是 0，见 §5.2，改用 Adam 那条
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

判读规则：
- **前景 Dice 恒为 0** → 先跑 `--stage data` 看有没有尺度异常样本，再跑 `--stage scale`
- `--stage scale` 里某格 BV/MAT 明显 > 0 → 数据可学，去查配置
- 所有格都 ≈ 0 → 输入表示不够，要改输入（分辨率 / 投影 / 多帧），不是改损失
- ⚠️ **单样本能学会 ≠ 跨样本能学会**。前者过了后者没过，一律先怀疑尺度/分布不一致

---

## 4. 验收测试（改完代码先跑这个）

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

## 5. 论文超参 vs 本机适配

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

### 5.1 🔴 实测数据怪癖：三成样本的像素尺度只有其余的 1/20

论文和 Zenodo 都没有提这件事。实测（`diagnose.py --stage data`，全 180 例）：

| 项 | 值 |
|---|---|
| MinIP 原始 mean | 最小 **106** / 中位 2288 / 最大 **2699** |
| 跨样本 mean 最大/最小 | **25.5 倍** |
| `mean < 中位数 20%` 的样本 | **55 / 180 例（30%）** |
| 这些样本除以 4095 后的 max | **0.033 ~ 0.056 → 整图近乎全黑** |

而且是有规律的：**全部是侧位（`*_L_*`）系列 / 472×512 那批**，像是另一套标定
（8-bit 或未做 rescale）。其余样本正常落在 0~4095。

**后果**：用固定 `/4095` 时，这三成样本等于"喂了一张黑图"进训练集。
模型在这些图上永远学不到东西，还会把前景的决策边界搅乱 ——
表现为 **loss 能降但前景 Dice 恒为 0**。

**证据链**（`diagnose.py --stage scale`）：
小 U-Net 单张图过拟合 400 iter，z-score 下 BV Dice **0.70**（暗样本 0.99）、
MAT Dice **0.93**；而固定 `/4095` 跨样本训练 1000 iter 前景 Dice 恒为 **0**。
单张学得会、混着训学不会 ⇒ 问题在**样本间分布不一致**，不在网络容量。

> ⚠️ 复盘时踩过的坑：最初用"3 层直筒 CNN + SGD + 200 iter"做这个判别实验，
> 得到前景 Dice=0，于是写下"类别不平衡导致坍缩"的结论 —— **这个结论是错的**。
> 3 层 CNN 感受野只有 7×7、200 步 SGD 根本不够，是个测不出信号的弱实验。
> 换成带 skip 的小 U-Net + Adam + 400 iter 后同一份数据前景 Dice 能到 0.70~0.99。
> **教训：用来"证伪数据可学性"的实验，本身必须先有足够容量和步数。**

`--norm` 四个取值：`zscore`（默认）/ `percentile` / `minmax` / `fixed`（旧行为，仅作对照）。

> 🔴 归一化是**数据侧**的事：`evaluate.py` 会自动沿用 ckpt 里记录的 `--norm`，
> 不要手工改成别的，否则输入分布对不上，Dice 会莫名其妙地低。

### 5.2 🔴 第二个坑：论文的 SGD 配方收敛极慢

**证据一：单样本过拟合隔离实验**（`diagnose.py --stage overfit --cores CXL_AP_38 --iters 150`）

| 配置 | 150 iter 后 | 判定 |
|---|---|---|
| SGD lr0.01 momentum0.99 + AMP(bf16) | 主力 loss 停在 **1.18**（= 全背景解） | ❌ 卡住 |
| SGD lr0.01 + nesterov + **关掉 AMP（fp32）** | 仍 **1.178**，卡在全背景 | ❌ 卡住 |
| **Adam lr1e-3** + AMP | **BV 0.439 / MAT 0.833**，预测出 14263 个前景像素 | ✅ 逃出 |

关掉 AMP 没有改善 ⇒ **不是 bf16 精度问题**。

**证据二：真实训练**（fold 0，144 训 / 36 验，12 epoch × 100 iter = 1200 iter）

| 优化器 | epoch 4 | epoch 8 | epoch 12 |
|---|---|---|---|
| SGD（论文配方 + nesterov） | 0.0000 | 0.0000 | 0.0341 |
| Adam lr1e-3 | 0.1020 | 0.0767 | **0.1224** |

对照：**改成 per-image z-score 之前**，1000 iter 的每一次验证前景 Dice 都是**精确的 0.0000**。
⇒ 归一化是决定性的；SGD 只是**在相同迭代数下比 Adam 慢约 4 倍**，不是完全学不动。

**怎么选：**

- **A. 忠于论文**：SGD + 500 epoch（`--optimizer sgd`，默认）。单折约 12 小时。
- **B. 想更快看到信号**：`--optimizer adam --lr 1e-3`。
  ⚠️ **这是偏离论文的地方，汇报时必须主动说明**，不能混过去。

`train.py` 的 `--optimizer` 默认仍是 `sgd`。另外给 SGD 补上了 **`nesterov=True`**
（nnUNet 的实际配方有这一项，原先漏了；可用 `--no-nesterov` 关掉对比）。

> ⚠️ **别用 12 epoch 的数字当结果**：论文跑 50000 iter（500ep×100iter），
> 上面 1200 iter 只占 2.4%，loss 和 Dice 都远未收敛，**只能用来判断"趋势对不对"**。
> 要和论文比 Dice，必须跑满 500 epoch。
>
> ⚠️ 而且上面这两组 12ep 数字还叠了**第二重低估**：当时 `--epochs 12` 让 lr 曲线也被
> 压缩到 12 个 epoch 的长度，末期 lr 只剩 10.7%（论文同期还有 98%）。见 §5.3。

> 排查顺序建议：`--stage data`（尺度/对齐）→ `--stage scale`（数据可学性）→
> `--stage overfit`（优化器/损失能否收敛）。三步都过了再跑长训练。

**为什么 batch 不一样** —— 实测端到端显存（batch=1，可用 5136 MiB）：

| 配置 | 峰值 | 结论 |
|---|---|---|
| fp32 | 6155 MiB | ❌ 超 |
| AMP(bf16) | 5194 MiB | ❌ 差 56 MiB |
| **AMP + 梯度检查点** | **2574 MiB** | ✅ 推荐 |

论文的 batch=2 在本机（RTX 3060 Laptop, 6 GB）光 TEB 一个分支就要 7856 MiB。
所以用 `batch=1 + 梯度累积 2 步`，**在数学上等效于 batch=2**，只是慢。

---

### 5.3 🔴 第三个坑：短 probe 的 lr 曲线被压缩，把"还没学会"误判成"学不会"

论文的 lr 是 **500 epoch** 的 polynomial decay。`poly_lr(epoch, total, lr0, power)` 里第一个
`total` 是**曲线总长度**，不一定是本次实际跑的 epoch 数 —— 这个区别能坑死人。

如果做 probe 时直接写 `--epochs 12`，曲线就被压缩了 40 倍。下表是 **lr 相对 lr0 的保留比例**
（`tmp/check_lr_semantics.py` 实测）：

| epoch 下标 | `--epochs 12`（旧做法） | `--epochs 40` | `--lr-total-epochs 500`（论文语义） |
|---|---|---|---|
| 4 | 0.694 | 0.910 | **0.993** |
| 11 | **0.107** | 0.749 | **0.980** |
| 20 | 0 | 0.536 | 0.964 |
| 39 | 0 | **0.036** | 0.930 |

⇒ 旧 probe 在**末期都等于把学习率掐到接近归零**。`0.12` 这种低 Dice 有很大一部分是
"油门被松掉了"，**不能当成模型能力上限**。

**正确姿势**（跑 40 ep 看趋势，但 lr 语义与论文对齐）：

```bash
python train.py --fold 0 --epochs 40 --iters-per-epoch 100 \
                --optimizer adam --lr 1e-3 --lr-total-epochs 500 \
                --out runs/probe_adam_40ep_lralign
```

日志会打印 `lr 曲线总长 500 epoch  ⚠️ 本实验只跑 40 epoch（=论文前 8.0% 的训练阶段）`，
一眼能看出这次 probe 相当于论文的哪个位置。

> ⚠️ **旧结果需要重新解读**：`cmp_sgd` / `cmp_adam`（都 12ep）和 `probe_adam_40ep`
> 用的都是"压缩版"曲线。它们**能**回答"归一化修好没有"（修之前是精确的 0.0000），
> 但**不能**回答"模型最终能到多少 Dice"。

---

### 5.4 🔴 第四个坑：val 只有 36 例 —— 同条件跨折就能差 0.19

`train.py` 每次验证只用该折的 36 例（`folds.json` 的 `val`）。样本量这么小，
**单次验证本身就是一次抽样**，它的波动不是"模型不稳"，而是"尺子有噪声"。

**两个不同尺度的噪声，别混用**（f0…f4 各 50 个验证点实测）：

| 尺度 | 实测 | 用途 |
|---|---|---|
| **曲线内**相邻点跳变 | 中位 **0.011**、p95 0.018~0.051 | 只用于判断"单点是不是异常" |
| **同条件跨折离散**（同为 144 例、只换折划分） | 末段 val **std 0.078、极差 0.186** | ✅ **跨实验比较的门槛** |

> ⚠️ **本文件早期版本写错过，现更正**：曾把 f0 的两次尾部跌落
> （ep110 0.2079→0.1647 = −0.043、ep220 0.2339→0.1515 = −0.082）
> 说成"典型抖动 0.04~0.08"。那两个是 p99 级单点事件，**不是典型值** ——
> 真实的中位跳变只有 0.011。修正后结论**更强**：门槛不是 0.04，而是 **0.078**。

**判读规则（做任何跨实验比较前先执行）**

1. 只比**平滑曲线**（≥10 个验证点的滑动平均）或**末段均值**，绝不看单点；
2. **进度不同必须先对齐 epoch**：`lc_n*` 只跑到 130、`f0` 跑到 500 时，
   比"末段均值"等于拿 135 轮比 500 轮，结论必错；
3. 两条曲线差距 **< 0.078 时不解释为"有差别"**；只在曲线内部比时才用 0.011 这条线；
4. 门槛由 `plot_learning_curve.py` 给出：`--band` 直接指定；
   `--band-from f0,f1,f2,f4` 用**同条件重复实验**现算（推荐）；
   都不给则退化为曲线内抖动 —— **那只是下限，会让人过度解读**。

**同一个坑的第二个面：`iters_per_epoch` 固定 ⇒ 这是「等算力」曲线**

`--iters-per-epoch` 默认 100，训练循环是 `for step in range(iters_per_epoch)` +
loader 取空就**重建迭代器**继续取。所以：

- 每个 epoch 恒定 100 步 / 200 次抽样（batch1×accum2），**与数据集大小无关**；
- 进度条 `step 40/100` 里的 `/100` 是**常量**，**不能**用来判断读进了多少样本；
  唯一可信的判据是日志头部 `训练 / 验证 : X / Y 例`；
- 横向比较时 **500 epoch = 50000 步，三份等步数可比**；
  但"每例被重复看多少次"差别巨大（n=36 ≈ 2778 次 vs n=144 ≈ 694 次）——
  汇报必须写明是**等步数口径**，否则会被质疑"小数据集多跑了好多遍当然过拟合"。

⚠️ 另一个必须同时声明的口径问题：**val 系统性高于 test**。
f0：val **0.2590** / test **0.1587**，**差 0.10**。所以
"val 上 A 比 B 好"不能直接外推成"test 上 A 比 B 好"，除非差距远大于 0.10。

配套脚本：

- `plot_learning_curve.py` —— 滑动平均 + 噪声带 + 脱困点，产 PNG + CSV；
  终端同时打印每条曲线的 best / 末段均值 / 脱困 epoch，以及**两两 Δ 的判定**
  （Δ < sigma 直接打「不可分辨」）。关键参数：`--smooth`（窗口）、`--tail`（末段长度）、
  `--ref`（哪条用来估噪声带）、`--spec name=path ...`（自定义曲线）。
- `fetch_histories.sh` —— **只把集群上的 `history.json` 拉回本机**（每个几十 KB，
  不拉 `.pth`），之后在本机出图、调窗口、反复改样式，不用来回 scp 图片。
  ⚠️ 需要先装好免密公钥（脚本里有未通时的三分支诊断）。
- `histories/` —— 手动搬运的落点。`histories/_需要哪些文件.txt` 列了**要哪几个文件、
  改什么名字**，按清单拖进来即可。脚本支持两种摆放，按顺序找、找到就用：
  `histories/<run>.json`（平铺，最省事）→ `runs/<run>/history.json`（完整目录）。

```bash
bash fetch_histories.sh                # → runs/<名字>/history.json
E:\Anaconda\envs\dsca\python.exe plot_learning_curve.py    # 零参数，本机直接出图
```

> 出图脚本图内文字**一律英文**：集群节点常缺中文字体，中文会渲染成方块。
> 本机渲染样式参考 `viz/plot_demo_local.png`（用本机已有 run 跑的自检，
> 标题已注明不是学习曲线实验）。

---

### 5.5 🔴🔴 第五个坑（最严重）：正方形样本的标签被静默转置，87.5% 的数据是错位训练的

**这一条推翻了本文件此前「瓶颈 = 泛化能力」的结论。** 2026-09-26 定位并修复。

#### 病灶

`data/prepare.py` 原来靠**形状**判断 label 要不要转置：

```python
if label.shape == (rows, cols):  return label,   "ok"          # ← 正方形样本走这里
if label.shape == (cols, rows):  return label.T, "transposed"
```

而 DSCA 的 label nii **一律**按 `(cols, rows)` 存 —— 非正方形样本形状就能看出来
（`fu_ZHL_AP_39.nii.gz` 是 `(742,960)` 而 DICOM 是 `960×742`），所以判据把 28 例
非正方形样本救回来了。但**正方形样本 `label.shape` 恰好等于 `(rows, cols)`**，
判据静默放行 → 标签带着 90° 错位直接进了训练集。

> 224 例里 **196 例中招**（训练 158/180 = 87.8%，测试 38/44 = 86.4%）。
> 新数据跑完 `prepare.py` 的日志是硬证据：`方向处置分布 : {'axis_T': 180}` ——
> 180 个训练样本**全部**该转置，其中 158 例被旧判据放行。

#### 证据链（5 条独立检验，全部可复现）

| # | 检验 | 结果 |
|---|---|---|
| 1 | 5 折 test 44 例按「旧判据是否生效」分组 | 生效的 6 例全部 >0.3（均值 **0.801**）；放行的 38 例全部 <0.23（均值 **0.052**） |
| 2 | 4 折 val 重复同样分组 | 生效 21/21 >0.3；放行 123/123 <0.23 → 合计 **27/27 vs 0/161，零例外** |
| 3 | processed npz 的「标签 vs MinIP 暗像素对齐 AUC」 | 非正方形 原方向 0.964 / 转置 0.501（本来就对）；正方形 原方向 **0.596** / 转置 **0.974**（喂进去的是错的那版） |
| 4 | 原始 `fu_*.nii.gz` vs 原始 DICOM | 正方形 原方向 0.41~0.74 / **转置 0.91~0.99** → nii 本身即转置存储 |
| 5 | npz **图像**方向 | `corr(npz.seq0, DICOM_min)` identity 0.49~0.99、转置 ≈0 → **图像是对的，只有标签错** |

顺带排除：224 例 nii 的 affine 全是 `diag(-1,-1,1)`、`qform_code=1`，**不含轴交换项**，
所以不能靠 affine 判方向（`as_closest_canonical` 也不会改变形状）。

#### 修复

`fix_label_axis()` 改成**证据驱动**：枚举 8 种二面体变换、只保留变换后形状合法者，
各算一次「标签 vs 该样本自己的 MinIP」对齐 AUC，取最大者，并把 AUC / 对次优的领先
幅度写进 `report.json`。加 `--axis-mode shape` 可复现旧行为做 A/B。

| | AUC 均值 | AUC < 0.9 | 可用（AUC>0.9） | 方向确定* |
|---|---|---|---|---|
| 旧（shape 判据） | 0.6223 | 199/224 | **25**/224 | 28/224 |
| 新（align 判据） | **0.9634** | 14/224 | **210**/224 | **224/224** |

\* 方向确定 = 所选方向对转置的领先幅度 ≥ 0.15。**审计要用这一条**，不能用绝对
AUC 阈值 —— 14 例低对比样本 AUC 只有 0.83~0.90，但它们对转置的领先是 +0.28~+0.47，
方向是确定的（属图像对比度问题）。

```bash
# 重新生成数据（95 秒）
E:\Anaconda\envs\dsca\python.exe data\prepare.py --dst "E:\Datasets\DSCA\processed_fix"

# 验收：方向确定 224/224 才算过；--compare 给出被纠正的样本清单
E:\Anaconda\envs\dsca\python.exe tmp\audit_label_align.py ^
    --proc "E:\Datasets\DSCA\processed_fix" --compare "E:\Datasets\DSCA\processed"

# 出 A/B 对照图（红=旧标签错位，绿=新标签贴合血管）
E:\Anaconda\envs\dsca\python.exe tmp\viz_axis_fix.py     # → viz/axis_fix_compare.png
```

旧数据留档：`E:\Datasets\DSCA\processed_axisbug_20260926`；新的已扶正为
`processed`，`folds.json` 与旧版**逐字节一致**（折划分没变，新老结果可直接对比）。

#### 修复后实测（250 步就超过旧版 50000 步）

同代码、同折（fold 0）、同优化器（adam lr 1e-3），**只换数据**：

```bash
E:\Anaconda\envs\dsca\python.exe train.py --fold 0 --epochs 10 --iters-per-epoch 25 ^
    --val-every 5 --lr-total-epochs 500 --optimizer adam --lr 1e-3 ^
    --num-workers 0 --out runs\verify_axis_fix
```

| | 步数 | mean_loss | val 前景均值 |
|---|---|---|---|
| 旧数据（f0 全量） | **50 000**（500 ep） | 1.61 → **1.24**（几乎没动） | **0.2590**（曲线末值） |
| 新数据（本次自检） | **250**（10 ep） | 1.85 → **0.74** | **0.6285**（@ep10；@ep5 已 0.5411） |

- 旧数据 5 万步后 loss 才降 23%；新数据 250 步就降 60%，且 val Dice 是旧版**最好的那个数**的 **2.4 倍**。
- ⇒ **根因确认无误**。计划里「Phase 2 修 A/B/C/D/F 之一后预期 0.40~0.65」的区间，
  现在光靠改数据、250 步就已经落在里面了 —— 所以**先别动网络**，把干净的 500 ep 单折跑完再说。

#### 为什么这个 bug 特别难查

1. **形状判据"看起来"是对的**：它确实修好了 28 例非正方形样本，让人以为方向问题已解决。
2. **正方形样本从形状上不可判**，必须回像素内容比对才能发现。
3. 🔴 **单样本过拟合探针查不出来**：转置是**确定性几何映射**，网络对单张图可以硬背 ——
   实测标签错位的 `CXL_AP_38`（952×952）单样本 300 iter 照样到 **BV 0.62 / MAT 0.97**。
   ⇒ 本文件此前把"单样本过拟合成功"当成"实现无 bug"的论据，**这个论据无效**。
4. **表现像"泛化不了"**：loss 只从 1.61 降到 1.24，test Dice 双峰，跨折乱飘 ——
   全是"图标签不匹配"该有的样子，很难与"模型容量/优化/超参"区分。
5. **分辨率/站位等元数据全都对不上**：`952×952` 得 0.05 而 `960×742` 得 0.80，
   `512×512` 得 0.06 而 `472×512` 得 0.81 —— 分辨率假设（疑点 A）就是被这个假象带偏的。

#### 连带作废的结论（引用旧数字前先看这里）

| 旧结论 | 现状 |
|---|---|
| 「瓶颈 = 泛化能力，不是数据/配置」 | ⛔ **错**，是数据管线 bug |
| 疑点 A：输入被缩放到 512 导致细血管丢失 | 🔄 **旧判据作废，但问题本身没被否掉**。当时用**错位数据**算出 r=−0.19、ρ=−0.02，这两个数**不能采信**。2026-09-26 新数据出现新线索（差额几乎全在 BV，而论文协议是"原生分辨率裁 512 + 滑窗还原"，见 §5.7）→ **已重写为待复检**：`tmp/audit_res_vs_dice.py`（改成按"压缩倍数"做剂量-反应 + 置换检验） |
| 5 折 test **0.1675 ± 0.0070**、val 离散是 test 的 10.4 倍、val/test 反相关 r=−0.67 | ⛔ **错位数据下的产物**，需在新数据上重算（反相关本身就是"指标无意义"的信号） |
| 学习曲线 n=36→0.1328 / 72→0.1372 / 108→0.1645 / 144→0.1587 | ⛔ **作废**。各子集里非正方形样本占比不同，成分效应不能当数据量效应 |
| 单样本过拟合 BV 0.809 / 0.882、MAT 0.874 ⇒ 数据可学 | ⚠️ 结论偶然成立，但**推理无效**，见上 |
| `fu_JYQ_BAP_45` 转置后仍不对齐（−3.4） | ✅ 在新数据上 AUC **0.9763**（领先 +0.51），是旧管线的产物 |
| 8-bit / 16-bit 像素尺度差 ~20 倍 | ✅ 仍成立，`dataset.py` 的 per-image z-score 已消化 |

---

## 5.6 🔴🔴 第六个坑：缺失类口径 —— 论文的 MAT 数字在"我们的口径"下数学上不可能

- **事实**：官方标注里 **8 / 44 张测试图根本没有 MAT（class 2）**。已对源头
  `DSA_public/labelsTs/fu_*.nii.gz` 逐例核实：未降采样样本 raw↔npz 体素数**逐位相同**，
  降采样样本严格按面积比（952²→0.289、1024²→0.25）⇒ **是官方标注的性质，不是 `prepare.py` 弄丢的**。
  名单：`GJW_AP_05 / GJW_AP_35 / GJW_L_14 / GJW_L_41 / LXJ_AP_07 / LXJ_AP_32 / LXJ_L_51 / LYL_AL_20`
- **两种口径**：`union`（本仓库 `metrics.py` 默认）= 标签有 **OR** 预测有 → 计入；
  `label`（论文口径）= 只看"标签里确实有"的样本。两者关系：
  `union 分子 = label 分子 + Σ(幻觉样本 dice)`，而幻觉样本 dice ≈ 1e-5/S ≈ 0 ⇒ 只差分母。
- **判定论文口径的硬证据**：union 下 MAT 的理论上限 = 36/44 = **81.8%**，
  但论文 Table III 里 **CENet 的 MAT Dice = 84.32%**、DSANet = **92.26%** —— **数学上不可能**。
  ⇒ 论文必用 label 口径。（这个论证只用论文自己的数字，可当场复算。）
- **对我们数字的影响**（5 折 test）：

| | union 口径 | **label 口径（可比论文）** |
|---|---|---|
| BV | 0.8173 ± 0.0026 | 0.8173 ± 0.0026（44 例全有 BV，不变）|
| MAT | 0.8421 ± 0.0313 | **0.8926 ± 0.0053** |
| **两类均** | 0.8297 ± 0.0148 | **0.8549 ± 0.0037** |

- **换算工具**：`eval_calibrate.py`（扫已有 eval json，两口径一起打，不用重跑模型）。

## 5.7 🔴 第七个坑：论文的测试协议是「原生分辨率 + 滑窗」，我们是「整图缩到 512」

论文原文（IV-A）：
> "Each batch comprised 2 samples at 512 × 512 pixels which were **randomly cropped from
> images of varying dimensions**." / "For testing ... **cropped patches of 512 × 512 pixels
> from the test image via sliding windows** ... **reconstructed to match the original image
> size**. Finally, we compared the reconstructed predictions with the GT **across the entire image**."

即论文**不缩放**：训练从原生分辨率随机裁 512²，测试滑窗推理后**还原到原生尺寸**再与 GT 比。
而我们 `prepare.py::fit_to_square()` 把长边压到 512 —— 实测 **train 125/180 = 69.4%**、
**test 36/44 = 81.8%** 的样本被压 **1.65 ~ 2.80 倍**。

**为什么怀疑它**：差距的**类别分布**是歪的（label 口径，百分数）：

| | 我们（5 折） | 论文 MinIP-only | 论文全模型 |
|---|---|---|---|
| BV（小血管） | **81.73** | 86.45 → **−4.72** | 87.32 → **−5.59** |
| MAT（主干） | **89.26** | 86.22 → **+3.04** | 92.26 → −3.00 |

差额几乎**全在 BV**，MAT 反而超过论文基线 ⇒ 更像"小结构在预处理里被磨掉"，不像容量不足。
（注：论文自己的 MinIP-only 里 BV≈MAT，而我们 BV 比 MAT 低 7.5 分，这个不对称本身就是线索。）

**复检命令**（⚠️ 筛查，n=44 不足以判决）：
```bash
python tmp/audit_res_vs_dice.py --data ~/dsca_data/processed \
    --evals 'runs/f*_axfix/eval_test.json' --out tmp/audit_res_vs_dice_axfix.txt
```
**支持假说的判据**（三条须同时满足）：BV 的 `r ≤ −0.30` 且 `p < 0.05`；BV 比 MAT 再负 `≥ 0.15`；
分组均值里 BV 随压缩倍数单调不增。
**不满足就别动预处理** —— 那会把一条已通过 44 项验收的数据管线推倒重来；
转去查 BV 相关配方（loss 权重 / 前景采样 / 增强强度）。

---

## 6. 时间预算（实测）

| 阶段 | 耗时 |
|---|---|
| 数据预处理（224 例） | 86 秒（修好轴序判据后 **95 秒**） |
| 训练 1 iter（含 2 次 micro-batch） | **约 0.85 秒** |
| 1 折 500 epoch | **约 12 小时** |
| 5 折全部 | **约 2.5 天**（连续） |

所以**不要一上来跑 500 epoch**。建议路径：

```
冒烟(1min) → 1折100ep(2.5h) → 看 Dice 趋势对不对 → 再决定要不要跑全量
```

---

## 7. 已知假设与遗留问题（诚实清单）

从论文里读不出来、由本实现**自行假设**的地方，全部做成了开关：

| # | 论文没交代的 | 本实现的默认 | 怎么改 |
|---|---|---|---|
| A1 | TF 注意力头数 | 8 | `--tf-heads` |
| A2 | q/k/v 投影层数 | 单层 Linear | 改 `temporalformer.MHSA` |
| A3 | MLP 隐藏维 | 4×dim | 改 `mlp_ratio` |
| A4 | LayerNorm 位置 | post-LN（依 Fig.3 的 Add&Norm） | `norm_first=True` 切回 pre-LN |
| A5 | 位置编码加几层 | 只第 1 层 | `pos_first_layer_only` |
| A6 | T 维下采样算子 | **Max-Avg pool**（从 Fig.3 读出） | `temporal_down` 可选 maxavg/conv3d/mean/last |
| A7 | α 的维度（**原文自相矛盾**） | `attention` 读法 | `--alpha-mode spatial_gate` 试另一种 |
| A8 | STF 的 Q/K/V 是否两路共享 | 不共享 | `share_qkv=True` |
| A9 | 编码器下采样次数 | 4 次 → 32×32（由解码器 4 次上采样反推） | `pool_first_layer=True` 切 5 次 |
| A10 | 损失权重 / 深监督权重 | CE 1.0 + Dice 1.0，辅助项 0.5 | `--ce-weight` / `--dice-weight` / `--aux-weight` |
| A11 | **输入归一化方式** | per-image z-score | `--norm fixed/percentile/minmax/zscore`，见 §5.1 |
| A12 | Dice 是否含背景 | 不含 | `--dice-include-bg` |
| A13 | 是否需要类别加权 / focal | 不需要（默认关闭） | `--class-weights invfreq`、`--focal-gamma 2.0` |
| A14 | **lr 曲线的总长度**（≠ 实际跑的 epoch 数） | 500（论文值） | `--lr-total-epochs N`，短 probe 必须显式设 500，见 §5.3 |

**A11 是唯一一个"论文没写但影响最大"的选项** —— 选错会让三成数据失效，见 §5.1。
A12/A13 是保险：先把归一化调对，再考虑要不要动损失；
实测下归一化调对之后，纯 CE+Dice 就能学起来，不需要靠加权硬顶。

**A14 是做实验时的隐形陷阱**：`--lr-total-epochs` 默认 0 = "曲线总长就是本次 `--epochs`"。
长训练时两者本来就相等，无需关心；但**跑短 probe 时一定要手动设成 500**，
否则 lr 会被压缩着衰减（12ep 时只剩 10.7%），测出来的 Dice 严重低估，见 §5.3。

**A7 是最需要人工判断的一处**：论文正文写 `α ∈ R^(B × hw/4)`（每个位置一个标量），
但按公式 `Softmax(Q Kᵀ)` 字面算应当是 `(B × hw/4 × hw/4)`。两种读法参数量完全相同，
只能靠论文的消融数字判对错。两种都实现好了，切开关即可。

**其它遗留：**

1. 有 1 例（`fu_JYQ_BAP_45`）的标签与图像疑似存在约 3 px 的内容错位（形状是对的，
   所以预处理不会报警）。要处理需要做内容级配准校验，暂未做。
2. **少数样本前景/背景对比极弱甚至反号**：`--stage data` 的对齐检查里，
   `CXL_AP_50` / `CXL_L_39` / `FZL_AP_39` 等少数例的"前景均值/背景均值"比值略大于 1
   （1.026~1.030），而中位数是 0.963。可能是 8 帧里没抓到造影剂峰值，
   或标签与图像有轻微错位。数量少，暂不作为主因处理，但评估时若见到个别样本
   Dice 异常低，可以先怀疑这几例。
3. 弹性形变增强已实现但**默认关闭**（每例多花约 250 ms，会拖慢 batch=1 的训练）。
   要开：`--elastic`。
4. 论文的"随机裁切"在本实现里不适用：预处理已把所有图统一到 512×512，
   没有比 512 更大的区域可裁。
   （实测：原图统一到 512 这一步**没有破坏标签**，`BV`/`MAT` 面积保留率 98%。）

---

## 8. 常见报错

| 现象 | 原因与处置 |
|---|---|
| `缺少 N 个文件` | 没跑 `data\prepare.py`，或 `--data` 指错了目录 |
| `FileNotFoundError: folds.json` | 同上；或只想重算折划分，用 `--folds_only` |
| CUDA out of memory | 确认 **没有**加 `--no-amp` 或 `--no-grad-checkpoint` |
| 训练极慢（每 iter 好几秒） | 多半是显存分页到内存了。用 `nvidia-smi` 看显存占用，或调小输入 |
| `nnUNet_raw is not defined` | 与本项目无关（那是 nnUNet 的环境变量检查）；本项目不依赖 nnUNet |
| PyCharm 里跑报 import 错误 | 解释器选错了（要选 `Python 3.11 (DSCA)`，不是项目里的 `.venv`） |

---

## 9. 参考：论文与数据

- 数据：Zenodo record `11255024`（restricted，已申请获批）。
  ⚠️ 数据集只放在**个人目录**，不要放学校公共盘 —— 可能构成再分发，违反申请时的承诺。
- 论文全文与中文精读笔记在 `F:\ResearchVault\`。
