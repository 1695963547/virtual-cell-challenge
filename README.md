# Virtual Cell Challenge (VCC) — 单细胞扰动响应预测复现项目

> 虚拟细胞挑战赛（[Virtual Cell Challenge](https://virtualcellchallenge.org/)，Arc Institute 举办）参赛与复现项目。
> 任务：给定 control 细胞 + 一个基因扰动（CRISPRi 敲低），预测扰动后细胞的基因表达分布。

本项目包含两条技术路线的完整复现与优化：

| 路线 | 模型 | 方法 | 说明 |
|---|---|---|---|
| **scDFM** | ICLR 2026 开源 | Distributional Flow Matching（条件生成模型） | `scDFM/`，flow-fusion + crisper 扰动编码 |
| **STATE** | Arc Institute 开源 | Transformer + SE 嵌入 | `state-vcc-local/`，基因空间直训 / SE+decoder 双路线 |

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
└── VCC_REVIEW.md             # ★ 技术复盘：从背景到结论的完整梳理
```

## 核心工作

1. **两个前沿开源模型的完整复现**：scDFM（ICLR 2026，Flow Matching）与 STATE（Transformer），并适配 VCC 任务（跨细胞系泛化、单基因扰动）。

2. **大规模工程优化（H20 × 4 卡）**：
   - **bf16 混合精度训练**：H20 的 FP32 算力弱而 BF16 tensor core 强，实测训练加速 **2.6-2.8×**
   - **18080 全基因显存优化**：利用 matmul 线性性对注意力层做**数学等价改写**，避免 ~500GB 显存的 OOM
   - **多卡并行评测**：扰动按 rank 分片，消除原版 3 卡重复跑全量的 3× 浪费
   - 每 rank 独立评测、checkpoint 断点恢复、训练循环主动 break 等 13 处训练层改进

3. **三条实验路线与深度消融**：
   - 基因空间直训（18080 维）vs SE latent + decoder（18080→2058→18080）
   - 自研 **FusedPDSDESLoss**（加权 MSE + 对比学习 PDS 代理 + 方向一致性）
   - FCN pseudo-bulk 轻量路线（参照 2025 亚军 XLearning Lab 方案）
   - cs × loss 消融矩阵、模型容量消融（108M / 201M / 301M）

4. **对评测指标的深入理解**：厘清了官方 DES/PDS/MAE 与 cell-eval 的对应关系（`overlap_at_N` / `discrimination_score_l1` / `mae`），并系统分析了**提升 DES/PDS 必然牺牲 MAE 的 fundamental trade-off**——这也解释了为什么榜首队伍普遍选择"放大扰动响应"。

## 主要结果

> 完整 7 项指标（DES/PDS/MAE/SPEARMAN/SPEARMAN_LFC/AUPRC/PEARSON）见 [结果对比.md](./结果对比.md)，完整技术复盘见 [VCC_REVIEW.md](./VCC_REVIEW.md)。

| 模型 | DES↑ | PDS↑ | MAE↓ | 备注 |
|---|---|---|---|---|
| **state_lg（本地最佳，301M）** | **0.198** | **0.560** | 0.507 | Transformer 路线，本地 Overall 估算 6.68 |
| **scDFM（iter15000）** | 0.084 | 0.503 | **0.042** | MAE 与 SPEARMAN_LFC 为本地各模型中最强 |
| **delta_mse（改进 loss）** | **0.236** | 0.526 | 0.142 | DES 为所有实验最高（+19% vs state_lg） |
| **cleopatra（官方 Generalist #1）** | 0.228 | 0.747 | 0.086 | 榜单参考 |
| **BM_xTVC（官方 Overall #1）** | 0.349 | 0.872 | 1.026 | 榜单参考 |

**关键发现**：
- 两路线互补——scDFM 在表达量幅度（MAE 0.042）与幅度排序（SPEARMAN_LFC 0.388）上显著领先；STATE 在 DE 基因检出（DES 0.198）与扰动可区分性（PDS 0.560）上更强
- 换模型容量（108M→301M）无法突破 PDS 瓶颈（0.53~0.56 卡住），瓶颈在**扰动多样性不足**而非模型容量
- MAE 基本"退出竞争"：均值 baseline 的 MAE（≈0.026）几乎无法被超越，顶尖队伍 MAE_scaled 全为 0

## 技术复盘

详见 [VCC_REVIEW.md](./VCC_REVIEW.md)，涵盖：任务背景 → 两条技术路线 → 工程优化 → 实验设计与消融 → 指标分析 → 经验教训。
