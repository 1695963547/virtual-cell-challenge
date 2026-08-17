# VCC 2025 优化方案（供后续 AI 会话同步）

> **最后更新**：2026-08-07
> **当前最佳模型**：state_lg（301M 参数，SE+decoder，cell_set_len=512）
> **当前最佳 avg_score**：0.0472
> **目标**：超越 BM_xTVC（官方 score 榜 #1，Overall=34.0，DES=0.349，PDS=0.872）
> **2026-08-07 重要更新**：① VCC 2025 已闭幕，2025 榜为最终状态，冲分目标改为**本地复现超越 BM_xTVC**（同时为 VCC 2026 练兵）；② Replogle GWPS 全量 + 官方 validation split（50 扰动真值）下载已启动；③ 指标机制与 winners 配方已按官方一手信息修正（详见 STATUS 第 8 节），其中 **"PDS 后处理放大无效"的结论已翻案**（见 1.2）。

> **评测纪律（全局，最高优先）**：
> - **对比榜上的任何得分，一律以 `/home/zjh/vcc_official/adata_Test.h5ad`（官方测试集，L2）测评为准**；未在测试集上跑过的分数不得与榜上分数直接对比，也不得写成"对比榜上得分"。
> - **训练过程中用官方 validation split（50 扰动）得出的分数**（如 val 锚点、valsplit/扩训对照的 val 出分）**仅用于模型选择与超参调优（L1），不属于榜单对比分数**；文档中凡 val 分数必须带 "@ val" 标注，凡 test 分数必须带 "@ test" 标注，两者严禁混用。
> - 所有上 test 的候选方案统一用 `run_vcc_test_eval.sh` 管线出分，写入 `vcc_test_eval/` 目录留存，以便后续对比与复盘。

---

## 1. 核心诊断结论

### 1.1 PDS 塌缩根因

| 诊断项 | 结论 |
|--------|------|
| **根因** | 预测方向不对（扰动间 delta 相关性中位数 0.842），不是幅度小 |
| **量化** | 训练 198 扰动 vs 测试 100 个未见扰动 → 扰动特异性信号不足 |
| **验证** | state（108M）→ state_lg（301M）→ tahoe_best（201M）PDS 都卡在 0.53~0.56，模型容量不是瓶颈 |
| **Top 队伍做法** | 全部自建网络（非 STATE），用 pseudo-bulk + hybrid（AI+统计）策略 |

### 1.2 已排除的路线

| 路线 | 结论 | 证据 |
|------|------|------|
| 增大模型 | ❌ 无效 | state_lg(301M) vs tahoe_best(201M) 基本持平（0.0472 vs 0.0462） |
| 更多训练步数 | ❌ 收益极小 | 8000 步后 val_loss 与测试指标脱钩 |
| PDS 后处理放大（方向错误时） | ⚠️ 结论修正 | 方向错时放大无效（6x 仅 +0.6%）；但 Outlier 证明**方向对之后** CV 全局缩放是 PDS 0.844 的关键组件（arXiv 2511.16954：L1-based PDS 幅度敏感，渐近符号余弦上限）。正确顺序：先修方向（跨细胞系迁移），再做 scaling 收割 |
| 更多训练数据（同类型） | ❌ 非瓶颈 | Tahoe-100M 是药物扰动，与 CRISPRi 不匹配 |

### 1.3 MAE 权衡

- **所有 Top 队伍 MAE 都很差**（BM_xTVC=1.026，Outlier=4.219），MAE_scaled 全部为 0
- 可以放心牺牲 MAE 换取 PDS/DES 提升
- 当前 state_lg MAE=0.507 已经比 Top 队伍好，但 MAE score=0（低于 baseline）

---

## 2. 优化路线（按优先级排序）

### 路线 A：Pseudo-bulk 训练（最高优先级）⭐⭐⭐

**原理**：把每个扰动的所有细胞聚合成一条"平均表达"向量，消除单细胞噪声，让模型直接学习扰动→平均响应的映射。

**为什么有效**：
- Top 3 队伍全部使用 pseudo-bulk
- 单细胞噪声是 PDS 塌缩的重要原因（22 万细胞中每个扰动只有 ~1700 个细胞，噪声淹没信号）
- Pseudo-bulk 后数据量从 22 万行→~580 行，训练极快

**实现方式**：
```
训练数据：每个扰动 → 均值表达向量（1 行）
  - 输入：ctrl_mean（非靶向对照均值）+ perturbation_id
  - 输出：pert_mean（该扰动的平均表达）
  - 或残差学习：输出 pert_mean - ctrl_mean（delta）

推理时：
  - 对测试集的每个扰动，预测其平均表达
  - 复制到该扰动的所有细胞（或直接用均值作为预测）
```

**预期收益**：PDS 从 0.56 → 0.7+（参考 Top 队伍），DES 同步提升

**工作量**：~2-3 天（数据预处理 + 简单 FCN 训练 + 评测）

---

### 路线 B：Hybrid（AI + 统计）策略 ⭐⭐⭐

**原理**：结合深度学习预测和统计先验，取两者之长。

**具体做法**：
1. **统计分支**：对训练集中每个扰动计算 mean delta（log2FC），作为该扰动的"统计指纹"
2. **AI 分支**：SE/STATE 模型预测扰动响应
3. **融合**：
   - 简单加权：`final = α × AI_pred + (1-α) × statistical_prior`
   - 或训练一个轻量融合层

**为什么有效**：
- BM_xTVC 明确使用 hybrid 策略
- 统计先验天然在 SPEARMAN_LFC/PEARSON 上强（baseline_train 的 SPEARMAN_LFC=0.432）
- 可以弥补 AI 模型在幅度预测上的不足

**预期收益**：SPEARMAN_LFC/PEARSON 显著提升，PDS/DES 中等提升

**工作量**：~1-2 天

---

### 路线 C：InfoNCE 对比损失 ⭐⭐

**原理**：在训练中加入对比学习目标，直接优化"不同扰动的预测应该不同"。

**实现**：
```python
# 在 STATE 的 loss 中加入：
# 对同一 batch 中不同扰动的预测 delta，用 InfoNCE 拉开距离
loss_contrast = InfoNCE(pred_delta_i, pred_delta_j)  # i≠j 应远离
total_loss = reconstruction_loss + λ × loss_contrast
```

**为什么有效**：
- 直接对抗 PDS 塌缩（扰动间相关性 0.842）
- 不需要改架构，只加 loss 项

**预期收益**：PDS 中等提升（0.56→0.65），DES 可能小幅提升

**工作量**：~1 天（修改 cell-load 的 loss 计算）

---

### 路线 D：Delta/残差预测 ⭐⭐

**原理**：模型不预测绝对表达，而是预测扰动引起的变化量（delta = pert - ctrl）。

**为什么有效**：
- 绝对表达预测中，大部分信号是"细胞类型基准表达"，扰动信号被淹没
- Delta 预测直接建模扰动效果，信噪比更高
- Top 队伍 XLearning Lab 使用残差学习

**实现**：
- 修改 STATE 的 output 为 delta
- 推理时：pred = ctrl_mean + predicted_delta

**预期收益**：DES/PDS/SPEARMAN 全面提升

**工作量**：~2 天（修改训练逻辑 + 推理逻辑）

---

### 路线 E：额外数据增强 ⭐⭐（Replogle 全量已升级）

**可用数据源**：
| 数据 | 来源 | 价值 |
|------|------|------|
| iPSC CRISPRi atlas（Nourreddine 2024） | 最匹配的额外 CRISPRi 数据 | ⭐⭐⭐ |
| Replogle K562 GWPS **全量** | 本地仅 184 扰动子集，需下载全量（~10,000 基因） | ⭐⭐⭐（升级） |
| Tahoe-100M | 药物扰动，与 CRISPRi 不匹配 | ⭐ |
| scBaseCount | 观测数据（非扰动），已通过 SE 继承 | ❌ |

**2026-08-07 新证据（Replogle 升级理由）**：
- 测试集 100 个扰动基因（EP300、CTNNB1、FGFR1、BECN1、SETDB1…）与本地 k562_gwps.h5 的 184 个扰动**重叠为 0**
- 全量 Replogle K562 GWPS 是全基因组筛选（~10,000 扰动），预计覆盖全部 100 个测试基因
- 拿到全量后，测试任务从"完全未见基因"变为"K562 扰动过、迁移到 H1 的基因"，直接缓解 PDS 塌缩根因
- **下载状态（2026-08-07）**：`replogle_2022_k562_gwps.h5ad`（8.2 GiB，scPerturb 处理版）后台下载中 → `/home/zjh/vcc_official/replogle/`；源 `https://exampledata.scverse.org/pertpy/replogle_2022_k562_gwps.h5ad`（Zenodo record 7041849 为同份数据，但本机访问 Zenodo 超时不可用；figshare 本机 403）
- **新资源（2026-08-07）**：官方 validation split（50 扰动真值，98,927 细胞 × 18080 基因，raw counts，赛后 2025-12-16 公开）下载中 → `/home/zjh/vcc_official/validation/`；定位为**本地零泄漏验证集**，冲分最终版再并入训练
- 注意：新数据无 X_state，SE+decoder 路线需用 SE-600M 跑 `state emb transform`；**pseudo-bulk FCN 路线直接从 raw counts 聚合，不受此影响**

**结论**：数据多样性是 PDS 塌缩根因之一（容量实验已证伪模型瓶颈）；Replogle 全量值得与路线 A/B 并行推进。

**SE+decoder 扩训实验（2026-08-08 启动，执行中）**：
- 消融矩阵：state_lg 老（hepg2-holdout val，val 锚点 0.0578）→ valsplit（官方 validation 作训练时 val，val 出分 0.0478，见 STATUS 3.2.2）→ +replogle_full（valsplit 同配方 + Replogle 全量训练数据，唯一变量 = 数据）
- 管线：Replogle 全量 → SE transform（✅ 08-09 10:05 完成，82.3GB）→ `prep_replogle_for_train.py`（8248→18080 映射 7581 命中 + log1p + 补 target_gene/batch_var/cell_type=k562 + 过滤 158 缺 ESM2 扰动 + 无压缩 csr，**已完成并校验**：1,961,299 细胞 × 18080，对照 75,328 细胞保留）→ 与 6×h5 合训（val 保持官方 validation 不动，toml/脚本已备：`starter_valsplit_replogle.toml` + `run_train_state_emb_all_replogle_full.sh`）
- 价值假设：test 扰动覆盖 0→97/100 是质变；风险：K562 占合训后细胞 83%/扰动 98%，需观察稀释效应
- 对照结论先行：valsplit 证明“只换 val 重训”不提升（0.0478 < 0.0578，且 best.ckpt 由该 val 选出带乐观偏差）→ 扩训若出现提升只能归因于数据；val_loss 与比赛指标不一致，后续 checkpoint 选择应直接看 val 的 cell-eval 分

---

### 路线 F：Self-knockdown 注入 ⭐（已验证，效果微弱）

**做法**：预测中强制目标基因下调（模拟 CRISPRi 的 knockdown 效果）

**验证结果**：DES +1.5%（0.198→0.201），PDS 不受影响（PDS 排除目标基因）

**结论**：收益太小，不值得单独使用，但可以作为其他路线的辅助手段。

---

## 3. 已完成的实验记录

### 3.1 PDS 后处理放大（2026-08-07）

| 实验 | 结果 |
|------|------|
| 快速扫描 12 个倍数 | 最优 6x，PDS 0.507→0.547（+7.9%，快速算法） |
| 完整 cell-eval 6x | PDS 0.560→0.563（+0.6%），DES 0.198→0.201（+1.5%） |
| MAE 影响 | 0.507→0.581（恶化 15%） |
| **结论** | 后处理放大无法解决方向问题，收益可忽略 |

**关键发现**：快速扫描（未归一化）与完整 cell-eval（有 normalize_total）差异大，绝对值不可比。

### 3.2 Baseline 修复（2026-08-07）

- 已将 `run_vcc_test_eval.sh` step 3 改用 `competition_train.h5`
- 正确 baseline：DES=0.1166，PDS=0.5094，MAE=0.0264

### 3.3 ESM-2 蛋白嵌入确认（2026-08-07）

- 确认已正确加载（19792 perturbations，5120 维）
- 无需额外改进

### 3.4 tahoe_best 训练+评测（2026-08-07）

| 项目 | 结果 |
|------|------|
| 配置 | tahoe_best（201M 参数，官方选优配置），8000 步，cs512 |
| val_loss | 0.695 → 0.262 单调下降（比 state_lg 低 25%） |
| avg_score | **0.0462**，与 state_lg（0.0472）基本持平 |
| 优势指标 | PDS 0.564 / MAE 0.504 / PEARSON 0.068 略优 |
| 劣势指标 | SPEARMAN 0.176 vs state_lg 0.284（-38%） |
| **结论** | val_loss 与榜单指标再次脱钩；换模型配置不是出路，PDS 天花板 0.53~0.56 与容量无关 |

### 3.5 未见扰动模拟器验证（2026-08-07，本地方案论证）

**模拟器**：`/home/zjh/state-vcc-local/pseudobulk/`（`build_delta_lib.py` 建 delta 库 + `v2v3_simulator.py`）。H1 150 扰动中与参考细胞系重叠的 136 个轮流留出，严格仿官方 PDS 定义（L1 排序）评测。

| 结论 | 证据 | 对方案的影响 |
|------|------|-------------|
| ✅ **Scaling 机制成立，64-128x 饱和** | PDS：1x=0.496 → 64x=0.666 → 256x=0.667；强效应层 0.17→0.56 | 方向迁移 + 大缩放成为主路线；6x 实验无效是因为远未到渐近区 |
| ✅ **难度分层：弱效应预测≈0 白拿分** | 弱层各模型 PDS≈0.81-0.83（含 zero）；corr(H1_l1, PDS)=-0.83 | 需要"弱效应 gate"：参考效应弱/靶基因 H1 不表达 → 预测≈0 |
| ✅ **方向迁移有效但质量是瓶颈** | cosine 仅 0.145（184 扰动稀疏子集） | Replogle 全量到位后重测，预期显著提升 |
| ❌ **稀疏化迁移伤害 PDS** | top100×64=0.488 vs 稠密×64=0.666 | PDS 预测保持稠密；稀疏化仅可能用于 DES（待验证） |
| ❌ **程序空间迁移（PCA 最小版）** | 0.494 < gene_direct 0.496；共享程序=共享应激响应，丢失扰动特异信号 | 思路③需重设计（残差上分解？），暂不投入 |
| ⚠️ **中位数保守性标量（缩小）有害** | gene_scaled(×s<1) 0.483 < gene_direct 0.496 | 缩放方向是放大不是缩小；与 Outlier 论文一致 |

**V1 保守性测量**：H1∩k562_gwps=136/150（训练配对充足）；跨细胞系 delta 相关中位数 0.15-0.28（pearson_top500），>0.3 仅 24% → 跨细胞系效应保守性有限，校准必要。

### 3.6 第二轮模拟器（V4）+ val 干跑管线（2026-08-07 下午）

**V4 新证伪/简化**（`pseudobulk/v4_gate_wavg.py`）：

| 假设 | 结果 | 结论 |
|------|------|------|
| 弱效应 gate（预测≈0 白拿分） | gate_x64 与 gene_direct_x64 完全相同（0.6660） | **不需要 gate**：弱效应层的 0.83 分是 L1 几何自动送的；特征也确实弱（ref_l1→h1_l1 corr=0.348，R²=0.12） |
| 相似度加权多参考聚合 | wavg_x64=0.6646 ≈ 均匀平均 0.6660 | **不需要加权**：k562_gwps 已主导；均匀平均即可 |

**val 覆盖检查（重要）**：val 50 扰动 ∩ 本地 k562_gwps 184 子集 = **47/50**（val∩H1 训练 = 0，与 test 同构的未见场景）；test 100 ∩ 本地 = 0（仍需 Replogle 全量）。→ validation 下完即可端到端干跑，不必等 Replogle。

**已就绪管线**：
- `pseudobulk/build_transfer_pred.py`：迁移 delta → cell-eval 格式预测 h5ad（BM_xTVC 式 replicate + 缩放参数）
- `run_val_transfer_eval.sh`：一键完成 建预测→cell-eval run→baseline→score（SCALES 环境变量扫倍数）
- `pseudobulk/compute_de_calls.py`：H1 真 DE 基因集（Wilcoxon，后台运行中）→ 补上模拟器的 DES 半边

**V5/V6 模拟器 DES 半边（2026-08-07 晚，`de_calls_H1.npz` = H1 150 扰动 Wilcoxon FDR≤0.05，中位 1082 基因/扰动）**：

| 假设 | 结果 | 结论 |
|------|------|------|
| 稀疏化对 DES 有利 | gene_top50-500 DES=0.16-0.18 < 稠密 0.252 | ❌ 稀疏化对 DES 也有害，彻底放弃 |
| h1_mean 混迁移提升 DES | λ=0.25: PDS -0.058/DES +0.007；λ=1: PDS -0.12/DES +0.03 | ❌ 混合得不偿失（PDS 权重~2×DES；h1_mean 优势含 H1 训练集批次水分，真实测试会缩水） |

**DES 分层发现**：强效应扰动迁移 DES 可达 0.48（弱效应仅 0.08，近噪声）→ DES 提升靠方向质量（Replogle 全量），不靠配方。

**最终单一配方（双指标共用一个预测）**：
```
pred_delta(测试扰动 g) = mean(各参考细胞系的 g delta) × 64
```
模拟器预期：PDS 0.667 / DES 0.252（注意：模拟器数值与真实测试不可直接比，看相对排序）

### 3.7 V7：Replogle 全量 vs 184 子集对照（2026-08-07 晚，已出结果）

Replogle 全量 delta 库已构建（`deltas_replogle_full.npz`，9,867 扰动，7,581/8,248 基因映射到 18080 面板）。`v7_replogle_full.py` 对照结果（H1 留出 136 扰动，×64）：

| 参考 | PDS_x64 | cosine | pearson_top500 | 分层 PDS（weak/mid/strong） |
|------|---------|--------|----------------|------|
| A. k562_gwps 184 子集（单参考） | 0.553 | 0.120 | 0.179 | 0.82 / 0.54 / 0.30 |
| B. replogle_full 全量（单参考） | 0.545 | 0.116 | 0.177 | 0.82 / 0.53 / 0.28 |
| C. 全量+4小参考 均匀平均 | 0.663 | 0.143 | 0.208 | 0.83 / 0.61 / 0.55 |

**结论（3.5 节“全量到位后方向质量预期显著提升”被证伪）**：
- 对这 136 个 H1 留出基因，全量与子集**是同一份 Replogle K562 实验数据** → cosine/PDS 持平（B≈A，C≈V2 的 0.666）。方向质量 ~0.14 cosine 是**跨细胞系保守性的生物学瓶颈**，不是参考数据稀疏导致的。
- 全量 Replogle 的真正价值 = **覆盖率**：test 从 0/100 → 97/100（gate 已通过）。模拟器无法评测此收益（只能玩 H1 有真值的 136 个，这些早被子集覆盖）。
- 多参考均匀平均仍有效（单参考 0.55 → 多参考 0.66），最终配方维持不变。
- 附带修复：`NpzFile.__getitem__` 不缓存（每次访问重新解压），v7/v2v3 的 load 已改为先取数组再切片（原写法对 686MB 数组需重复解压 9867 次 ≈ 6.8TB，跑不完）。

### 3.8 当前路线图（V7 后最终版）

**val 迁移首测（2026-08-09，counts 整数空间同口径，cell-eval vcc profile，真值=官方 validation）**：

| 方案 @ val | DES | PDS | MAE |
|---|---|---|---|
| state_lg 老路线（counts 转换） | **0.2423** | **0.5764** | 0.547 |
| 迁移 x1（多参考 delta 均匀平均） | 0.2146 | 0.5340 | **0.087** |
| 迁移 x64 | 0.1750 | 0.5756 | 0.763 |

**判据未达成**：PDS 0.5756 < 0.581（state_lg 锚点），DES 也低于 state_lg → 纯迁移未确认优势（模拟器 0.667 是理想化上限，真实 val 的 PDS 天花板 ≈ 0.58，两个完全不同的机制同时摸到）。但 x1 的 MAE 0.087 碾压（表达水平预测极准），放大确实把迁移 PDS 拉到 state_lg 水平（0.534→0.576）。

**当前决策**：先试 state_lg 预测 × 迁移预测的线性混合（α 扫描，val 上选优），不行则回到 state_lg 作为最终提交。

**混合扫描结果（2026-08-09 晚，log1p 空间）**：

| 方案 @ val | DES | PDS | MAE |
|---|---|---|---|
| state_lg 老路线 | **0.2120** | **0.5812** | 0.509 |
| 混合 α=0.25 | 0.1970 | 0.5552 | 0.137 |
| 混合 α=0.5 | 0.1979 | 0.5708 | 0.260 |
| 混合 α=0.75 | 0.1985 | 0.5676 | 0.384 |

**混合证伪**：所有 α 的 PDS/DES 都低于 state_lg 纯预测（混合只改善死指标 MAE）→ 两条独立机制（模型/统计迁移）摸到的 PDS 天花板都是 ≈0.58，混合不能叠加。**最终提交 = state_lg 老路线**（val 三线对比最强：老路线 > valsplit > 扩训，迁移与混合均未超越）。

**val 锚点已建立（2026-08-07 晚）**：老路线 state_lg @ validation = DES_raw 0.212 / PDS_raw 0.581 / avg_score 0.0578 / VCC Overall 7.94（`run_val_state_eval.sh`，详见 STATUS 3.2.1，log1p 空间口径）。**注意：以上均为 @ val 分数（L1，只用于选型），任何与榜上得分的对比必须在 adata_Test.h5ad 上测评（L2）后才有意义。**

```
最终配方：pred_delta(g) = mean(各参考细胞系 delta) × 64  （模拟器 PDS 0.663 / DES 0.252）
下一步①：bash run_val_transfer_eval.sh（SCALES="1 64"）→ 第一个真实迁移 DES/PDS（val 47/50 覆盖）
         判据：PDS_raw > 0.581（state_lg val 锚点）则迁移路线在真实指标上确认优势
下一步②：test 参考库换 replogle_full（97/100 覆盖）→ 同管线对 test 出分（L2，仅最终确认）
方向质量（cosine ~0.14）是跨细胞系生物学天花板，不再指望数据量；若要突破走 FCN 残差学习（路线 5）
```

### 3.9 Route B：decoder_loss 从 energy_distance 换成 delta MSE（2026-08-10 启动）

**动机**：当前 loss 两部分：
- main_loss = energy_distance(pred_latent_2058, true_latent_2058) — 2058 维 latent 空间，主导
- decoder_loss = energy_distance(decoded_genes_18080, true_counts_18080) — 18080 维基因空间，辅助

比赛指标（PDS/DES）在基因空间计算 delta，但 decoder_loss 用的是分布距离（energy distance）而非 delta MSE，且优化全表达而非扰动效果。**这是 loss 与比赛指标脱钩的直接原因。**

**改动（只改 decoder_loss，其他全不动）**：

```
原来：decoder_loss = energy_distance(gene_decoder(pred), pert_cell_counts)
改成：delta_pred = gene_decoder(pred) − gene_decoder(ctrl_latent.detach())
      delta_true = pert_cell_counts − gene_decoder(ctrl_latent.detach())
      decoder_loss = MSE(delta_pred, delta_true)
```

- main_loss 不动（latent energy distance 保留）
- decoder_loss_weight 保持 1.0
- 不加 PDS/DES proxy（先做最小实验）
- 代码改动位置：`state_transition.py` training_step L628-661、validation_step L716-739
- 新增参数：`model.kwargs.decoder_loss_type="delta_mse"`（默认 "energy" = 原行为）
- **回滚方法**：把 `decoder_loss_type` 改为 "energy" 或删除该行即可恢复原 loss

**训练配置**：
- init_from = `competition/state_emb_all_lg_cs512/checkpoints/best.ckpt`（复用 state_lg 已训好的 main network）
- decoder 随机初始化，从零学 delta MSE 映射
- 脚本：`run_train_delta_mse.sh`
- NAME = `state_emb_all_lg_delta_mse`

**判据**：val cell-eval 上 PDS_raw > 0.581（state_lg 锚点）则 delta MSE 确认有效

---

## 4. 推荐执行顺序

```
Phase 1（立即可做，1-2 天）：
  ├─ 路线 B：统计先验融合（最快见效）
  └─ 路线 D：改为 delta 预测（改 STATE output）

Phase 2（中期，3-5 天）：
  ├─ 路线 A：Pseudo-bulk 训练（最大收益）
  ├─ 路线 C：InfoNCE 对比损失（或 BM_xTVC 式 PDS ranking loss）
  └─ 路线 E：Replogle K562 GWPS 全量（下载已启动 2026-08-07，到位后先跑覆盖率 gate）

Phase 3（长期，1-2 周）：
  └─ 路线 E：iPSC CRISPRi atlas + Replogle 训练（需先算 X_state）
```

**核心策略**：放弃在 STATE 框架内微调，转向 pseudo-bulk + hybrid 的轻量方案（与 Top 队伍一致）。

---

## 5. 关键文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| 项目状态 | `/home/zjh/state-vcc-local/VCC_CHALLENGE_STATUS.md` | 完整项目文档 |
| 结果对比 | `/home/zjh/结果对比.md` | 所有模型 7 指标对比 |
| 指标解释 | `/home/zjh/指标解释_DES_PDS与MAE权衡.md` | 指标细节 |
| 训练数据 | `/home/zjh/state-vcc-local/state/vci_pretrain/*.h5` | 6 个 h5 文件 |
| 测试集 | `/home/zjh/vcc_official/adata_Test.h5ad` | 170K 细胞，101 pert |
| 官方 validation（真值） | `/home/zjh/vcc_official/validation/adata_Validation.h5ad` | 50 pert，98,927 细胞，本地零泄漏验证集（2026-08-07 下载） |
| Replogle GWPS 全量 | `/home/zjh/vcc_official/replogle/replogle_2022_k562_gwps.h5ad` | 8.2 GiB，~10k CRISPRi 扰动（2026-08-07 下载） |
| Baseline（正确） | `competition/vcc_test_eval/baseline/` | 基于 competition_train.h5 |
| state_lg 预测 | `competition/prediction_test_state_lg.h5ad` | 26GB |
| state_lg 6x 放大 | `competition/prediction_test_state_lg_6x.h5ad` | 后处理放大版 |
| PDS 扫描脚本 | `/home/zjh/state-vcc-local/pds_sweep_fast.py` | 快速 PDS 扫描 |
| PDS 放大脚本 | `/home/zjh/state-vcc-local/pds_amplify.py` | 后处理放大工具 |
| 推理脚本 | `/home/zjh/state-vcc-local/infer_local.py` | 模型推理 |
| 评测脚本 | `/home/zjh/state-vcc-local/run_vcc_test_eval.sh` | 完整评测流程 |
| 训练日志 | `/home/zjh/log/` | 所有日志统一放这里 |

---

## 6. Top 队伍策略总结（2026-08-07 官方 wrap-up 核实版）

| 队伍 | 排名 | 核心方法（已核实） |
|------|------|----------|
| BM_xTVC | Score #1 | 改进 scFoundation + ESM-2 + 精选公开扰动数据；DEG 频率/均值表达作显式特征；fused loss(PDS+DES+小MAE)；pseudo-bulk 训练 + **重复输出满足 DES 的 Wilcoxon** |
| XLearning Lab | Score #2 | **FCN + 聚合对照 + ESM-2 + UMI 指示 + 残差学习**；仅用公开 Perturb-seq（含 PerturbAtlas 小 H1 数据）；明确按 "PDS≈2×DES 权重" 分配优化预算 |
| Outlier | Score #3 | 纯统计 TransPert：跨细胞系 pseudo-bulk + Wilcoxon 摘要 + 相似性加权聚合 + **CV 全局缩放**（PDS 0.844） |
| cleopatra (Altos) | Generalist #1 | flow matching 生成模型（U-Net，基因表达空间），~7M 细胞预训练（含内部扰动筛选，无 H1），H1 小数据微调 |

**共同点**：
1. 全部使用 pseudo-bulk 或聚合策略
2. 全部自建网络（没有直接用 STATE）
3. 全部牺牲 MAE 换 PDS/DES（官方确认 MAE 事实退出竞争）
4. Hybrid（AI+统计）优于纯 AI
5. **PDS 对幅度敏感**：方向对之后全局 scaling 是免费分数（Outlier 核心 trick）
