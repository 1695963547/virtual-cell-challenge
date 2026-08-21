# Virtual Cell Challenge — 单细胞扰动响应预测复现

> 本项目是 [Virtual Cell Challenge 2025](https://virtualcellchallenge.org/)（Arc Institute）的参赛与复现仓库。
> 任务为 **跨细胞系预测 H1 人胚胎干细胞上未见 CRISPRi 扰动的全转录组响应**。

## 官方任务

**跨细胞系泛化到 H1**：用其他细胞系（K562、RPE1、Jurkat、HepG2）的公开扰动数据 + H1 的 151 个已知扰动，预测 H1 细胞系上未见 CRISPRi 扰动的全转录组响应。

- **目标细胞系**：H1 人胚胎干细胞（hESCs）
- **核心挑战**：跨细胞系的分布外泛化（H1 与 K562、A375 等常见训练细胞系存在显著分布差异）
- **关键约束**：训练时已提供 H1 的 151 个已知扰动，任务不是 few-shot adaptation，而是把多细胞系知识迁移到 H1 的未见扰动上
- **评测工具**：[cell-eval](https://github.com/ArcInstitute/cell-eval)（Arc Institute 官方）
- **评测指标**：7 项指标，其中 DES / PDS / MAE 决定 Overall Score

## 两条技术路线

本仓库完整复现并优化了两个前沿模型：

| 路线 | 模型 | 方法 | 目录 |
|---|---|---|---|
| **scDFM** | ICLR 2026 开源 | Distributional Flow Matching（条件生成模型） | `scDFM/` |
| **STATE** | Arc Institute 开源 | Transformer + SE-600M 嵌入 | `state-vcc-local/` |

- **STATE 路线**：在 18080 维基因空间直训，本地 Overall 估算最佳（6.68），DES 0.198 / PDS 0.560 本地领先。
- **scDFM 路线**：条件生成建模，MAE 0.042 / SPEARMAN_LFC 0.388 本地最强。
- **互补性**：scDFM 擅长度量级与 LFC 排序，STATE 擅长 DE 基因检出与扰动可分性。

## 目录结构

```
.
├── scDFM/                    # 路线一：scDFM（Flow Matching）复现与 VCC 适配
│   ├── vcc/                  #   VCC 全部改动：数据构建、训练、推理、评测脚本
│   └── VCC_ADAPTATION.md     #   完整改动记录（数据/模型/训练/推理 4 层）
├── state-vcc-local/          # 路线二：STATE（Transformer）复现与优化
│   ├── state/                #   STATE 仓库完整代码
│   ├── pseudobulk/           #   FCN pseudo-bulk 轻量路线（XLearning 式）
│   └── run_*.sh              #   各实验的训练/评测流水线
├── competition_support_set/  # 官方训练数据（6 个 h5 文件）
├── vcc_official/             # 官方测试集 adata_Test.h5ad（18080 基因，~170K 细胞）
├── vcc_peek/                 # 测试集数据探索
├── SE-600M/                  # SE-600M 嵌入模型权重
├── download_vcc_test.sh      # 测试集下载脚本（断点续传 + crc32c 校验）
├── 指标解释_DES_PDS与MAE权衡.md  # 三个核心指标的原理与权衡分析
├── 结果对比.md                   # 全部实验的 7 项指标详细对比
└── VCC_REVIEW.md             # 技术复盘：从背景到结论的完整梳理
```

## 快速开始

### 环境

```bash
conda create -n vcc python=3.10
conda activate vcc
pip install -r state-vcc-local/requirements.txt
```

### 数据准备

```bash
bash download_vcc_test.sh
```

训练数据已置于 `competition_support_set/` 与 `state-vcc-local/state/vci_pretrain/`。

### STATE 路线训练与评测

```bash
cd state-vcc-local
bash run_train_state_emb_all_lg.sh    # 训练 state_lg 301M 全基因直训模型
bash run_eval_state_emb_all_lg.sh     # 在官方测试集上评测
```

### scDFM 路线训练与评测

```bash
cd scDFM/vcc
bash train_scdfm_vcc.sh       # 训练 scDFM 适配模型
bash eval_scdfm_vcc.sh        # 在官方测试集上评测
```

> 完整训练/评测脚本与超参见各目录下的 `*.sh` 与 `*.toml` 文件。

## 主要结果

> 完整 7 项指标（DES / PDS / MAE / SPEARMAN / SPEARMAN_LFC / AUPRC / PEARSON）见 [结果对比.md](./结果对比.md)。
> 所有本地得分均通过官方 `cell-eval` 在 `adata_Test.h5ad` 上评测得到。

| 模型 | DES↑ | PDS↑ | MAE↓ | 备注 |
|---|---|---|---|---|
| **state_lg（本地最佳，301M）** | **0.198** | **0.560** | 0.507 | Transformer 路线，本地 Overall 估算 6.68 |
| **scDFM（iter15000）** | 0.084 | 0.503 | **0.042** | MAE 与 SPEARMAN_LFC 本地各模型最强 |
| **delta_mse（改进 loss）** | **0.236** | 0.526 | 0.142 | DES 为所有实验最高（+19% vs state_lg） |
| **cleopatra（Generalist #1）** | 0.228 | 0.747 | 0.086 | 榜单参考 |
| **BM_xTVC（Overall #1）** | 0.349 | 0.872 | 1.026 | 榜单参考 |

## 关键提升概览

> 相对官方 baseline（扰动均值预测，DES 0.107 / PDS 0.509 / MAE 0.026）与 Generalist 榜首 cleopatra 的提升。完整对比见 [结果对比.md](./结果对比.md)。

| 阶段 | 指标 | 对比基准 | 结果 | 提升幅度 |
|---|---|---|---|---|
| STATE 基线模型落地 | DES | 官方 baseline 0.107 | **0.198** | **+85%** |
| | PDS | 官方 baseline 0.509 | **0.560** | +10% |
| FusedPDSDESLoss 改进 | DES | STATE 基线 0.198 | **0.236**（本地最高） | **+19%** |
| | MAE | STATE 基线 0.507 | **0.142** | **-72%** |
| scDFM 引入 | MAE | 榜首 cleopatra 0.086 | **0.042** | **-51%** |
| | SPEARMAN_LFC | 榜首 cleopatra 0.396 | **0.388** | 达榜首 98% |

## 核心发现

1. **两路线互补**：scDFM 在表达量幅度（MAE 0.042）与 LFC 排序（SPEARMAN_LFC 0.388）上显著领先；STATE 在 DE 基因检出（DES 0.198）与扰动可分性（PDS 0.560）上更强。
2. **损失函数是关键**：自研 FusedPDSDESLoss 将 DES 从 0.198 提升至 0.236（+19%），为所有实验最高。
3. **PDS 瓶颈不在模型容量**：108M → 201M → 301M 的容量扩展无法突破 PDS 0.53~0.56 的平台，瓶颈在于训练数据中扰动多样性不足。
4. **MAE 基本"退出竞争"**：均值 baseline 的 MAE ≈ 0.026 几乎无法被超越，顶尖队伍 MAE_scaled 普遍为 0。

## 工程优化

- **bf16 混合精度训练**：H20 上训练加速 2.6–2.8×
- **18080 全基因注意力显存优化**：数学等价改写，避免 ~500GB OOM
- **多卡并行评测**：扰动按 rank 分片，消除 3× 重复计算
- **训练稳定性**：checkpoint 断点恢复、训练循环主动 break、动态学习率调度等 13 处改进

## 技术文档索引

| 文档 | 内容 |
|---|---|
| [VCC_REVIEW.md](./VCC_REVIEW.md) | 任务背景、路线、工程优化、实验设计与结论的完整复盘 |
| [scDFM/VCC_ADAPTATION.md](./scDFM/VCC_ADAPTATION.md) | scDFM 迁移到 VCC 的完整改动记录 |
| [结果对比.md](./结果对比.md) | 全部实验的 7 项指标详细对比 |
| [指标解释_DES_PDS与MAE权衡.md](./指标解释_DES_PDS与MAE权衡.md) | DES/PDS/MAE 原理与权衡分析 |
| [state-vcc-local/VCC_CHALLENGE_STATUS.md](./state-vcc-local/VCC_CHALLENGE_STATUS.md) | 项目状态、评测口径与 2026 赛制信息 |

## Citation

若使用本项目代码或数据，请同时引用相关原始工作：

- Arc Institute. *Virtual Cell Challenge*. https://virtualcellchallenge.org/
- Arc Institute. *state*. https://github.com/ArcInstitute/state
- *scDFM* (ICLR 2026). 原始论文与代码请见相应官方发布
