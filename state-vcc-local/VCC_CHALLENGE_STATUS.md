# VCC 2025 打榜项目状态

> **目标**：本地复现并超越 2025 榜单第一名（BM_xTVC，Overall=34.0，DES=0.349，PDS=0.872）。
> **当前阶段**：引入 Replogle GWPS 全量 + 官方 validation split，转向 pseudo-bulk + 跨细胞系迁移路线。
> **最后更新**：2026-08-07
> **注意**：VCC 2025 已闭幕（2025-12-06 公布 winners），2025 榜单为最终状态不再接受提交；VCC 2026 已开赛（见 1.3 节）。

---

## 1. 比赛简介

Arc Institute 举办的 Virtual Cell Challenge 2025，要求模型给定起始细胞状态 + 基因扰动，预测扰动后的基因表达变化。

### 1.1 官方核心任务（准确描述）

> **跨细胞系泛化到 H1**：用其他细胞系（K562、RPE1、Jurkat、HepG2）的公开扰动数据 + H1 的部分已知扰动，预测 H1 细胞系上未见 CRISPRi 扰动的全转录组响应。

- **目标细胞系**：H1 人胚胎干细胞（hESCs）
- **核心挑战**：跨细胞系的分布外泛化（H1 与常见训练细胞系如 K562、A375 等存在分布差异）
- **不是 Few-shot Adaptation**：训练时已经提供 H1 的 151 个已知扰动，任务不是用极少样本去适应新细胞系，而是把多细胞系知识迁移到 H1 的未见扰动上

### 1.2 本地数据与分工

| 数据 | cell_type | 细胞数 | 扰动数 | 作用 |
|---|---|---|---|---|
| `competition_train.h5` | **ARC_H1** | 221,273 | ~151 | H1 训练集（官方提供的目标细胞系训练数据） |
| `k562_gwps.h5` | k562 | 111,605 | ~184 | 跨细胞系辅助数据 |
| `rpe1.h5` | rpe1 | 22,317 | ~69 | 跨细胞系辅助数据 |
| `jurkat.h5` | jurkat | 21,412 | ~54 | 跨细胞系辅助数据 |
| `k562.h5` | k562 | 18,465 | ~54 | 跨细胞系辅助数据 |
| `hepg2.h5` | hepg2 | 9,386 | ~69 | 本地 zeroshot test（不同细胞系，仅用于本地开发验证） |
| `adata_Test.h5ad` | （无列，推测为 H1 held-out） | 170,846 | **101** | **最终测试集 / 榜单提交依据** |

- **测试集**：`/home/zjh/vcc_official/adata_Test.h5ad`（18080 基因，~170K 细胞，101 个 perturbation）
- **训练数据**：`/home/zjh/state-vcc-local/state/vci_pretrain/` 下的 6 个 h5 文件（competition_train + 5 个辅助文件）
- **评测工具**：[cell-eval](https://github.com/ArcInstitute/cell-eval)（Arc Institute 官方）
- **评测页面**：https://virtualcellchallenge.org/evaluation

> **本地评测口径（2026-08-07 确认）**：本地所有测试得分 = 模型对 100 个测试扰动的预测经 cell-eval 对 `adata_Test.h5ad`（官方测试真值）评测得到，与榜单提交同口径。因真值在本地，须遵守**三级评测纪律**：L0 迁移模拟器（`pseudobulk/v2v3_simulator.py`，H1 留出扰动，秒级，用于方法设计）→ L1 官方 validation split（50 扰动，下载中，用于管线/超参选择）→ L2 真实测试集（仅最终确认，尽量减少触碰次数，防过拟合榜单）。

### 1.3 VCC 2026 赛制（2026-08-07 官网一手核实）

- **2025 已闭幕**：2025-12-06 公布 winners（BM_xTVC / XLearning Lab / Outlier，Generalist Prize = Altos Labs），2025 榜单为 final standings，不再接受提交。
- **2026 进行中**：2026-08-04 开放注册；**08-20 开赛**（发布 3 个验证细胞系的 NTC 对照 + 每系 300 个扰动，提交预测上实时榜，每 24h 一次）；10-22 发布 3 个 held-out 测试细胞系（NTC + 每系 300 扰动）；11-05 最终提交截止（每天最多 2 次，只算最后一次）；11 月下旬公布 winners。
- **任务变化**：multi-context generalization + **zero-shot**——官方**不提供新训练集**，只给目标细胞系的 NTC 对照，预测其扰动响应；可用去年 H1 数据 + 任何有权限的公开/私有数据。
- **ML-only 规则**：预测必须完全由 ML 模型生成；实验数据可用于训练/微调，但不能直接当预测或直接修正预测（纯统计方法处于灰色地带，统计组件应做成模型的输入特征）。
- **奖金**：$100k / $50k / $25k（半现金半 NVIDIA Brev credits）；Scoring Rubric 开赛时才公布。

---

## 2. VCC 评测指标

### 2.1 三个核心指标（决定 Overall Score）

| 指标 | 全称 | cell-eval 对应 | 说明 |
|------|------|---------------|------|
| **DES** | Differential Expression Score | `overlap_at_N` | 预测与真值 top-N DE 基因重叠率。按 \|log2FC\| 对 real/pred 显著基因排序后取 top k（k=real 显著基因数），计算 \|real_topk ∩ pred_topk\| / k。 |
| **PDS** | Perturbation Discrimination Score | `discrimination_score_l1` | L1 距离判断每个预测最接近哪个真值扰动 |
| **MAE** | Mean Absolute Error | `mae` | 全转录组（18080 基因）平均绝对误差 |

### 2.2 Overall Score 公式

```
S = (DES_scaled + PDS_scaled + MAE_scaled) / 3 × 100
```

- `*_scaled` 是相对 baseline 归一化后的分数（`cell-eval score` 命令输出）
- **baseline 必须基于 `competition_train.h5` 构建**，不能用测试集真值构建（会导致 baseline 人为过强，压制 scaled scores）

### 2.3 四个 Generalist Prize 指标（附加评测）

| 指标 | cell-eval 对应 | 说明 |
|------|---------------|------|
| **SPEARMAN** | `de_spearman_sig` | 显著 DE 基因效果量的 Spearman 相关 |
| **SPEARMAN_LFC** | `de_spearman_lfc_sig` | log fold change 的 Spearman 相关 |
| **AUPRC** | `pr_auc` | 精确率-召回率曲线下面积 |
| **PEARSON** | `pearson_delta` | delta 表达（扰动 vs 对照）的 Pearson 相关 |

### 2.4 DES 计算方法

cell-eval 的 `overlap_at_N` 就是官方 DES 的对应指标。其计算逻辑为：

```python
# 对每个 perturbation：
# 1. real 侧：取 fdr < 0.05 的基因，按 |log2FC| 降序排列，取前 k 个（k = real 显著基因数）
# 2. pred 侧：取 fdr < 0.05 的基因，按 |log2FC| 降序排列，取前 k 个
# 3. DES = |real_topk ∩ pred_topk| / k
# 4. 最后对所有 perturbation 求平均
```

DE 文件由 `cell-eval run` 生成：`pred_de.csv`（预测侧）和 `real_de.csv`（真值侧）。

> 注意：这与“fdr<0.05 后按上下调分开算交集/min”的手算方式不同，后者尺度在 0.37~0.89 之间，与榜单 DES（0.1~0.3）不兼容；经榜单 Overall 反推验证，官方 DES 对应 `overlap_at_N`。

---

## 3. 当前训练路线与结果

### 3.1 两条训练路线

| | 路线 A：基因空间直训 | 路线 B：SE + decoder |
|---|---|---|
| **output_space** | `gene`（直接预测 HVG 子集） | `all`（预测 2058 latent → decoder 解码 18080） |
| **训练脚本** | `run_train_state_emb.sh`（output_space=gene） | `run_train_state_emb_all.sh`（output_space=all） |
| **checkpoint 目录** | `competition/full_run/` | `competition/state_emb_all_run/` |
| **main network** | 108M params | 108M params（同架构） |
| **decoder** | 无（直接输出基因空间） | 13M params（LatentToGeneDecoder 2058→18080） |
| **训练步数** | 8000 | 8000 |
| **训练耗时** | ~1h | ~1h |

### 3.2 评测结果对比

> **两个模型都在完整 `adata_Test.h5ad`（170,846 细胞，101 perturbation，18080 基因）上评测**，真值相同。
> - full_run best.ckpt：用 `competition_test_template.h5ad` 推理（output_space=gene，直接读 .X）
> - SE+decoder best.ckpt：用 `vci_pretrain/adata_Test.h5ad`（带 X_state 嵌入）推理（output_space=all，读 obsm[X_state]）
> - **两个模型都统一用 `baseline_train`（训练数据 `competition_train.h5` 造的扰动均值 baseline）归一化**，同口径对比

#### raw 分数（未归一化）

| VCC 指标 | 方向 | cell-eval 对应 | full_run best | SE+decoder | baseline_train | cleopatra (Generalist #1) |
|---------|------|---------------|-------------|------------|----------------|---------------------------|
| **DES** | ↑越大越好 | `overlap_at_N` | 0.133 | **0.175** | **0.107** | 0.228 |
| **PDS** | ↑越大越好 | `discrimination_score_l1` | 0.526 | 0.527 | 0.509 | 0.747 |
| **MAE** | ↓越小越好 | `mae` | **0.036** | 0.353 | **0.026** | 0.086 |
| **SPEARMAN** | ↑越大越好 | `de_spearman_sig` | 0.081 | **0.067** | -0.161 | 0.473 |
| **SPEARMAN_LFC** | ↑越大越好 | `de_spearman_lfc_sig` | 0.093 | 0.020 | **0.432** | 0.396 |
| **AUPRC** | ↑越大越好 | `pr_auc` | **0.237** | 0.200 | 0.144 | 0.266 |
| **PEARSON** | ↑越大越好 | `pearson_delta` | **0.091** | 0.042 | **0.148** | 0.203 |

#### 归一化分数（都 vs baseline_train）

> cell-eval 归一化公式：
> - **↓越小越好**（MAE 等）：`scaled = 1 - model/baseline`，若 model > baseline → scaled < 0 → clip to 0
> - **↑越大越好**（DES/PDS/SPEARMAN 等）：`scaled = (model - baseline) / (1 - baseline)`，若 model < baseline → scaled < 0 → clip to 0

| VCC 指标 | full_run scaled | SE+decoder scaled | 为什么是 0 |
|---------|----------------|-------------------|------------|
| **DES** | 0.029 | **0.077** | baseline 0.107，模型仍低于 1.0；SE+decoder 略超 |
| **PDS** | **0.034** | 0.037 | 两模型略超 baseline 0.509 |
| **MAE** | 0.0 | 0.0 | 模型 MAE 0.036/0.353 > baseline 0.026 |
| **SPEARMAN** | **0.209** | 0.190 | baseline 是负的(-0.161)，模型正值都超 |
| **SPEARMAN_LFC** | 0.0 | 0.0 | baseline 0.432 远超模型 0.093/0.020 |
| **AUPRC** | **0.109** | 0.066 | 都超 baseline 0.144 |
| **PEARSON** | 0.0 | 0.0 | baseline 0.148 超模型 0.091/0.042 |
| **VCC Overall Score** | **2.10** | **2.79** | (DES+PDS+MAE)/3×100 |
| cell-eval avg_score | 0.033 | 0.033 | 所有指标均值 |

### 3.2.1 state_lg @ 官方 validation split（2026-08-07 晚，首个 val 锚点）

老路线（SE+decoder，state_lg best.ckpt）在官方 validation（50 扰动，零泄漏）上的端到端评测，管线 `run_val_state_eval.sh`（SE transform 补 X_state → 推理 → cell-eval，baseline 基于 competition_train + val counts）：

| 指标（cell-eval） | state_lg @ val raw | state_lg @ val scaled | 参考：state_lg @ test raw |
|---|---|---|---|
| DES `overlap_at_N` | 0.212 | 0.100 | 0.198 |
| PDS `discrimination_score_l1` | 0.581 | 0.138 | 0.560 |
| MAE `mae` | 0.509 | 0.0 | 0.507 |
| SPEARMAN `de_spearman_sig` | 0.240 | 0.451 | 0.284 |
| SPEARMAN_LFC | 0.174 | 0.0 | 0.093 |
| AUPRC `pr_auc` | 0.229 | 0.090 | 0.237 |
| PEARSON `pearson_delta` | 0.078 | 0.0 | 0.091 |
| **VCC Overall** | — | **7.94** | — |
| cell-eval avg_score | — | **0.0578** | 0.0472 |

结论：老路线在 val 上与 test 同量级（略高，val 扰动可能偏弱效应偏多）→ val 可作为 test 的可靠代理；`de_nsig_counts` pred 12663 vs real 3235，模型预测的显著基因数仍虚高 ~4 倍。结果目录 `competition/val_state_eval/`。

### 3.2.2 valsplit run @ 官方 validation（2026-08-08 晚）

valsplit run（`state_emb_all_lg_cs512_valsplit`：与 state_lg 同数据同配方，仅训练时验证集从 hepg2-holdout 换成官方 validation，8000 步于 08-07 21:41 训完）用 best.ckpt 在同一官方 validation 上评测（管线同 `run_val_state_eval.sh`，PRED/OUT 已参数化，结果目录 `competition/val_valsplit_eval/`）：

| 指标（cell-eval） | state_lg 老 run raw | valsplit raw | valsplit scaled |
|---|---|---|---|
| DES `overlap_at_N` | 0.212 | 0.210 | 0.098 |
| PDS `discrimination_score_l1` | 0.581 | 0.514 | 0.001 |
| SPEARMAN `de_spearman_sig` | 0.240 | **0.307** | **0.500** |
| SPEARMAN_LFC | 0.174 | 0.103 | 0.0 |
| PEARSON `pearson_delta` | 0.078 | 0.026 | 0.0 |
| cell-eval **avg_score** | — | — | **0.0478**（老 run 0.0578） |

结论：
- **换 val 重训没有提升 val 总分**（0.0478 < 0.0578）；且 best.ckpt 由该 val 的 val_loss 选出、带轻微乐观偏差，真实差距只会更大
- 结构上 DE 方向相关性变好（spearman 0.240→0.307），但 PDS（0.581→0.514）与 pearson_delta（0.078→0.026）明显变差——val_loss 与比赛指标（PDS/DES）不一致，checkpoint 选择若只看 val_loss 会把模型推向“整体表达拟合好”而非“扰动可区分”
- valsplit run 的定位 = **扩训实验的同配方对照组**：+replogle_full 将与它唯一变量（训练数据）对比，提升必须来自数据而非重训

### 3.3 分析

#### DES 为什么 scaled 分不高

官方 DES 对应 cell-eval 的 `overlap_at_N`：按 \|log2FC\| 排序后取 real 显著基因数 k，计算 top-k 重叠率。

- baseline_train `overlap_at_N` = 0.107
- full_run = 0.133 → scaled = (0.133-0.107)/(1-0.107) = **0.029**
- SE+decoder = 0.175 → scaled = (0.175-0.107)/(1-0.107) = **0.077**

两个模型都仅略超 baseline，说明在“top-N 差异基因集合”这个任务上，STATE 基线相比扰动均值预测没有明显优势。

#### MAE 归一化为什么是 0

cell-eval 对“越小越好”的指标用 `norm_by_zero` 公式：`scaled = 1 - model/baseline`。
- baseline_train MAE = 0.026（扰动均值预测误差极低）
- full_run MAE = 0.036 → 1 - 0.036/0.026 = -0.38 → clip to 0
- SE+decoder MAE = 0.353 → 1 - 0.353/0.026 = -12.6 → clip to 0

**模型的预测误差比“直接取均值”还大**，所以归一化分 = 0。

#### 核心结论

- **7 个指标中 3 个（MAE、SPEARMAN_LFC、PEARSON）两模型都不如 baseline_train** → STATE 基线模型还没超过“扰动均值预测”的水平
- **VCC Overall Score**：SE+decoder=2.79，full_run=2.10，榜单 Top1=34.0，差距巨大
- DES 上 SE+decoder（0.175）略优于 full_run（0.133），但仍只接近 baseline（0.107）
- baseline_train（扰动均值预测）在 SPEARMAN_LFC=0.432、PEARSON=0.148 上极强，这是均值预测的天然优势

---

## 4. 已知问题

### 4.1 Baseline 构建方式错误（重要）

当前 `run_vcc_test_eval.sh` 的 step 3 用**测试集真值** `adata_Test.h5ad` 构建 baseline：

```bash
# run_vcc_test_eval.sh line 114-115
"${CELL_EVAL[@]}" baseline \
    -a "$TRUTH" \   # ← 这是测试集真值，不是 competition_train.h5！
```

**VCC 规范要求 baseline 基于 `competition_train.h5` 构建**。用测试集真值构建的 baseline 人为过强（MSE≈0），导致 scaled scores 被压制到接近 0。

**修复方向**：将 baseline 的 `-a` 参数改为 `competition_train.h5`，pert_counts 用训练集的扰动计数。

### 4.2 ~~full_run 评测不完整~~（已修复）

full_run best.ckpt 已在完整 `adata_Test.h5ad`（101 pert）上评测完成，结果在 `vcc_test_eval/best/`。早期 hepg2 子集评测结果保留在 `full_run/eval_best.ckpt/hepg2_*.csv`（仅参考）。

### 4.3 沙箱 GPU 限制

- bwrap 沙箱内 `setsid nohup` 后台进程拿不到 GPU（`torch.cuda.is_available()=False`）
- 训练和推理需在用户终端（沙箱外）启动，或用前台进程
- **2026-08-07 验证可行**：Qoder Bash 工具的 `is_background=true`（不加 setsid，进程留在沙箱会话内）可以拿到 GPU，val_state_eval 全程就是这样跑的

### 4.5 长任务进程管理与误杀教训（2026-08-08）

- 沙箱内 `is_background` 终端会吞命令/延迟执行；前台 timeout 直接杀进程（不转后台）。**唯一验证可行的长任务机制：`nohup ... & disown`（不要 setsid，setsid 在沙箱内丢 GPU，见 4.3）**，监控只看日志文件、不碰进程
- DataLoader worker 由主进程 fork，**继承完整 cmdline、RSS 显示 COW 共享值**（看似 90GB 实为共享）。判断“重复实例”必须查 PPID：worker 的 PPID = 主进程 PID。误杀 worker 会让主进程 `RuntimeError: DataLoader worker ... killed`——已因此损失过一次 7313 batch 的 transform 进度
- SearchReplace 写盘后 bash 端可能短暂读到旧内容（写入刷新延迟），重要脚本改完用 `sync; sed -n` 复核再执行

### 4.4 训练 val_loss 的真实口径（2026-08-07 查清）

当前 `starter.toml`（`[fewshot]` 为空）下，cell_load 的 `val_dataloader()` 回退用 test_datasets（hepg2 zeroshot 全系），且 `validation_step` 算的是 **2058 维 latent 空间的 energy distance**（geomloss）——细胞系（hepg2≠H1）、空间（latent≠基因）、度量（energy≠DES/PDS）三重错配，这是 val_loss 与榜单纯指标脱钩的机制原因。best.ckpt 的选择依据就是这个 val_loss。官方 validation 即便接入训练也只是换"在哪算"，不换"算什么"——选型应以 val 上的 cell-eval 指标为准（见 7.评测纪律）。

---

## 5. 关键文件路径

### 5.1 数据

| 文件 | 路径 | 说明 |
|------|------|------|
| 测试集真值 | `/home/zjh/vcc_official/adata_Test.h5ad` | 18080 基因，~170K 细胞，11 GB |
| 扰动计数 | `/home/zjh/vcc_official/pert_counts_Test.csv` | 测试集每个 perturbation 的细胞数 |
| **官方 validation（真值）** | `/home/zjh/vcc_official/validation/adata_Validation.h5ad` | **50 扰动，98,927 细胞，18080 基因，raw counts**；2025-12-16 赛后公开；与 train/test 零重叠 → 本地零泄漏验证集 |
| **Replogle K562 GWPS 全量**（已下载 8.8GB） | `/home/zjh/vcc_official/replogle/replogle_2022_k562_gwps.h5ad` | 199 万细胞 × **9,867 扰动** × 8,248 基因面板（raw counts，对照标签='control'，gene 列对照名 non-targeting，基因名在 var.gene_name）；**覆盖率 gate 已通过（2026-08-07）：test 100 覆盖 97**（缺 AASS/CNN3/COL6A1），val 50 覆盖 47（全量不新增 val 覆盖，val 保持干净） |
| Replogle 全量 + X_state ✅ | `state/vci_pretrain/replogle_full_se.h5ad` | 82.3GB，08-09 10:05 完成（6.26 it/s 全程稳定）；X_state (1989578, 2058) 无 NaN；脚本 `run_transform_replogle_full.sh` |
| **Replogle 训练版** ✅ | `state/vci_pretrain/replogle_full_trainready.h5ad` | 83.6GB，**1,961,299 细胞 × 18080，9,709 扰动（含 non-targeting 对照 75,328 细胞）**；7581/8248 基因映射、log1p、无压缩 csr、X_state 连续；`prep_replogle_for_train.py` 生成 |
| 训练数据 | `/home/zjh/state-vcc-local/state/vci_pretrain/*.h5` | 6 个训练文件 |
| 训练配置 | `/home/zjh/state-vcc-local/state/vci_pretrain/starter_fewshot.toml` | 引用 6 个训练文件 |

### 5.2 训练

| 文件 | 路径 | 说明 |
|------|------|------|
| 基因空间训练脚本 | `/home/zjh/state-vcc-local/run_train_state_emb.sh` | output_space=gene |
| SE+decoder 训练脚本 | `/home/zjh/state-vcc-local/run_train_state_emb_all.sh` | output_space=all |
| full_run checkpoints | `/home/zjh/state-vcc-local/state/competition/full_run/checkpoints/` | best.ckpt 等 |
| valsplit checkpoints | `state/competition/state_emb_all_lg_cs512_valsplit/checkpoints/` | best.ckpt, last.ckpt（8000 步，08-07 21:41 训完） |
| Replogle prep 脚本 | `/home/zjh/state-vcc-local/prep_replogle_for_train.py` | replogle_full_se.h5ad → cell_load 训练格式（argv 可覆盖输入输出做冒烟） |
| SE+decoder checkpoints | `/home/zjh/state-vcc-local/state/competition/state_emb_all_run/checkpoints/` | best.ckpt, step1000-8000 等 |
| 训练日志 | `/home/zjh/log/train_state_emb_all.log` | SE+decoder 训练日志 |

### 5.3 评测

| 文件 | 路径 | 说明 |
|------|------|------|
| 评测脚本 | `/home/zjh/state-vcc-local/run_vcc_test_eval.sh` | 推理 → cell-eval → baseline → score |
| val 迁移评测脚本 | `/home/zjh/state-vcc-local/run_val_transfer_eval.sh` | 统计迁移预测 → cell-eval（validation，SCALES 扫倍数） |
| val 老路线评测脚本 | `/home/zjh/state-vcc-local/run_val_state_eval.sh` | SE transform → state_lg 推理 → cell-eval（validation） |
| state_lg @ val 结果 | `state/competition/val_state_eval/` | agg_results / score_state_lg_vs_trainbase.csv（见 3.2.1） |
| valsplit @ val 结果 | `state/competition/val_valsplit_eval/` | 见 3.2.2；baseline 产物硬链接自 val_state_eval（不重复构建） |
| validation 嵌入版 | `state/vci_pretrain/adata_Validation.h5ad` | 带 obsm[X_state]（SE-600M transform，7.3G） |
| 推理脚本 | `/home/zjh/state-vcc-local/infer_local.py` | 支持 `--embed-key` 参数 |
| SE+decoder 评测结果 | `/home/zjh/state-vcc-local/state/competition/vcc_test_eval/state/` | agg_results, pred_de, real_de, results（全 test 101 pert） |
| SE+decoder 归一化分（vs trainbase） | `vcc_test_eval/score_se_vs_trainbase.csv` | 同口径归一化（baseline_train） |
| full_run 归一化分（vs trainbase） | `vcc_test_eval/score_best_vs_trainbase.csv` | 同口径归一化（baseline_train） |
| full_run best 评测结果 | `/home/zjh/state-vcc-local/state/competition/vcc_test_eval/best/` | agg_results, pred_de, results（全 test 101 pert） |
| full_run hepg2 子集评测 | `/home/zjh/state-vcc-local/state/competition/full_run/eval_best.ckpt/` | hepg2_agg_results, hepg2_pred_de, hepg2_real_de（仅 hepg2） |

### 5.4 评测命令

```bash
# SE+decoder 评测（需在用户终端跑，沙箱 GPU 不稳）
cd /home/zjh/state-vcc-local && \
TEMPLATE=vci_pretrain/adata_Test.h5ad \
EMBED_KEY=X_state \
RUN_DIR=competition/state_emb_all_run \
CKPT=best.ckpt \
setsid nohup bash run_vcc_test_eval.sh \
  > /home/zjh/log/vcc_test_eval_se_all.log 2>&1 < /dev/null &

# 单独重算归一化分（用已有 baseline_train，不需要 GPU）
cd /home/zjh/state-vcc-local/state && uv run --no-sync python -c "
import logging,sys; logging.basicConfig(level=logging.INFO)
sys.argv=['cell-eval','score',
  '-i','competition/vcc_test_eval/state/agg_results.csv',
  '-I','competition/vcc_test_eval/baseline_train/agg_results.csv',
  '-o','competition/vcc_test_eval/score_se_vs_trainbase.csv']
from cell_eval.__main__ import main; main()
"
```

---

## 6. 榜单参考（VCC 2025 Generalist Prize Top 5）

| 排名 | 团队 | OVERALL | DES | PDS | MAE | SPEARMAN | SPEARMAN_LFC | AUPRC | PEARSON |
|------|------|---------|-----|-----|-----|----------|--------------|-------|---------|
| 1 | cleopatra (Altos Labs) | 28.7 | 0.228 | 0.747 | 0.086 | 0.473 | 0.396 | 0.266 | 0.203 |
| 2 | xBio | 31.1 | 0.305 | 0.811 | 0.770 | 0.564 | 0.087 | 0.252 | 0.217 |
| 3 | Mean Predictors | 34.0 | 0.305 | 0.741 | 6.723 | 0.294 | 0.213 | 0.582 | 0.217 |
| 4 | Shippers | 34.4 | 0.354 | 0.699 | 0.231 | 0.000 | 0.227 | 0.576 | 0.123 |
| 5 | Cellock Holmes | 36.4 | 0.356 | 0.679 | 0.239 | 0.000 | 0.238 | 0.576 | 0.125 |

> OVERALL 是各指标排名的平均（越低越好），不是分数。MAE 越低越好，其他指标越高越好。

---

## 7. 下一步建议

> 2026-08-07 起按 VCC_OPTIMIZATION_PLAN.md 的方向全景执行：

1. **Replogle GWPS 覆盖率验证**（gate）：100 个测试基因在 ~10k 扰动中的覆盖数 ≥80 则跨细胞系迁移路线成立
2. **pseudo-bulk 聚合管线**：全部数据集统一成 (cell_line, pert) → delta 向量（对齐 18080 基因面板、统一归一化口径）
3. **TransPert 式统计迁移 baseline**：K562 delta → H1，用 H1 已知扰动做保守性校准（建立 PDS ~0.8 的强下限）
4. **PDS scaling**：方向正确后在 validation split 上 CV 选全局缩放因子（Outlier 核心 trick，见第 8 节）
5. **FCN 残差分支 + hybrid 融合**（XLearning 配方）：pseudo-bulk FCN + ESM-2 + fused loss（PDS+DES），冲击 PDS 0.87+
6. **评测纪律**：一切实验先在 validation split 上验证，再碰 test；最终版 train+val 合训

---

## 8. 官方一手信息核实（2026-08-07，Arc wrap-up + 指标定义原文）

### 8.1 指标机制（官方确认版）

- **PDS**：pseudo-bulk delta（vs NTC 对照）之间的 L1 距离排序，PDS_p = 1 − (r_p−1)/N，随机 ≈ 0.5。**对幅度敏感**：增大预测幅度 PDS 单调上升，渐近"符号余弦相似度"上限（Outlier 团队论文 arXiv 2511.16954）→ 方向对之后全局 scaling 是免费分数。
- **DES**：pred/real 各自做 Wilcoxon（FDR≤0.05，BH 校正），按 |log2FC| 排序取 top-n（n = real 显著基因数）算重叠——与本地使用的 `overlap_at_N` 口径一致。
- **MAE**：pseudo-bulk 层面计算；官方承认其事实退出竞争（几乎所有模型 MAE 不如均值 baseline，Top 队伍 MAE_scaled 全为 0）。
- **最终分**：三指标各自对 perturbation-mean baseline 归一化后取均值；**PDS 实际权重约为 DES 两倍**（XLearning 的 metric-driven 判断，与数值反推一致）。

### 8.2 Winners 配方（官方复盘）

| 队伍 | 方法要点（已核实） |
|---|---|
| BM_xTVC #1 | 改进 scFoundation 预训练 + ESM-2 + 精选公开扰动数据；训练集 DEG 频率/均值表达作**显式输入特征**；fused loss(PDS+DES+小权重MAE)；**pseudo-bulk 训练，推理时重复输出多条充当细胞以满足 DES 的 Wilcoxon 检验** |
| XLearning Lab #2 | **FCN + 聚合对照表达 + ESM-2 + UMI count 指示 + 残差学习（预测 delta）**；MSE+辅助 loss；仅用公开 Perturb-seq 数据（含 PerturbAtlas 的小 H1 数据集） |
| Outlier #3 | **纯统计 TransPert**：跨细胞系 pseudo-bulk + Wilcoxon 摘要 + 相似性加权聚合 + 显著基因预测 + **CV 全局线性缩放优化 PDS**（PDS 0.844） |
| Altos（Generalist） | flow matching 生成模型（定制 U-Net，直接作用于基因表达空间），~7M 细胞预训练（含内部扰动筛选数据，无 H1），再用 H1 小数据微调 |

**对本项目的启示**：hybrid > 纯 AI；pseudo-bulk 是三赢家共识；模型不需要大（FCN 即榜二）；**数据覆盖 > 模型容量**；PDS 优化 = 先修方向（跨细胞系迁移），再收 scaling。
