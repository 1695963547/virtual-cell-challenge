# VCC 指标解释：DES、PDS 与 MAE 的权衡

> 记录 DES 和 PDS 两个核心指标的含义，以及为什么榜单上模型提升 DES/PDS 时 MAE 往往会下降。

---

## 1. DES（Differential Expression Score）

###  leaderboard 口径

在 VCC 官方 score 榜和 `cell-eval` 工具里，**DES 对应 `overlap_at_N`**。

它衡量的是：

> 对每个扰动，模型能不能把**真正差异表达的基因**排进 top-k。

### 计算方式

对单个扰动 \(p\)：

1. 计算每个基因的 log2 fold change：
   \[
   \text{log2FC}_g = \log_2\left(\frac{\text{mean}(\text{perturbed}_g)}{\text{mean}(\text{control}_g)}\right)
   \]
2. 在**真值**侧，取显著差异表达基因（如 fdr < 0.05），按 \(|\text{log2FC}|\) 降序取前 \(k\) 个 → `real_topk`。
3. 在**预测**侧，同样按预测出的 \(|\text{log2FC}|\) 降序取前 \(k\) 个 → `pred_topk`。
4. 计算交集比例：
   \[
   \text{DES}_p = \frac{|\text{real\_topk} \cap \text{pred\_topk}|}{k}
   \]
5. 对所有扰动求平均，得到最终 DES。

### 直观理解

DES 高 = 模型知道“这个扰动让哪些基因上调/下调”。

它**不在乎预测幅度绝对准不准**，只在乎：
- 哪些基因是差异表达的；
- 这些基因的排序对不对。

---

## 2. PDS（Perturbation Discrimination Score）

### leaderboard 口径

在 VCC 官方 score 榜和 `cell-eval` 工具里，**PDS 对应 `discrimination_score_l1`**。

它衡量的是：

> 每个预测的扰动响应，和哪个真实扰动响应最像？是不是它自己？

### 计算方式

对单个扰动 \(p\)：

1. 把预测结果压缩成一个平均表达向量 \(\hat{y}_p\)。
2. 把它和所有真实扰动的平均表达向量 \(y_q\) 做 L1 距离比较：
   \[
   d(p, q) = \|\hat{y}_p - y_q\|_1
   \]
3. 如果 \(q = p\) 时距离最小，算正确；否则算错误。
4. 最后统计正确率：
   \[
   \text{PDS} = \frac{\text{正确匹配的扰动数}}{\text{总扰动数}}
   \]

### 直观理解

PDS 高 = 不同扰动产生的预测模式足够“不一样”，模型没有把一切都预测成同一个“平均扰动响应”。

PDS 低 = 模型塌缩了：不管输入什么扰动，输出都大同小异。

---

## 3. 为什么提升 DES/PDS 时，MAE 会下降？

### 核心原因：优化目标不同

| 指标 | 关注什么 | 对预测的要求 |
|---|---|---|
| **MAE** | 预测值和真值的绝对误差 | 每个基因的数值尽量接近真值 |
| **DES** | 差异基因有没有选对 | 响应模式正确，幅度可以偏 |
| **PDS** | 不同扰动是否可区分 | 扰动间差异要足够大 |

### 一个具体例子

某个扰动的真实响应：

| 基因 | 真实 log2FC |
|---|---|
| A | +2.0 |
| B | -1.0 |
| 其他 | ~0 |

**模型 1（MAE 导向）**：
- A: +0.5
- B: -0.3
- 其他：接近 0

→ MAE 很小，但 A、B 的幅度被压缩，可能进不了 top-k DE 基因列表。
→ **DES 低，PDS 也容易低**（所有扰动都被压得差不多）。

**模型 2（DES/PDS 导向）**：
- A: +5.0
- B: -3.0
- 其他：带一些扰动特异性的小信号

→ MAE 很大，但 A、B 明显是 top DE 基因 → **DES 高**；
→ 不同扰动产生的模式差异大 → **PDS 高**。

### 本质

- **MAE 奖励“保守”**：预测贴近均值/对照，误差小；
- **DES/PDS 奖励“敢于响应”**：模型必须放大扰动信号，才能让差异基因排出来、让不同扰动拉开距离。

这就是两者之间的 fundamental trade-off。

---

## 4. 榜单上的证据

官方 score 榜前三：

| 队伍 | DES | PDS | MAE | Overall |
|---|---|---|---|---|
| **BM_xTVC** | 0.349 | **0.872** | 1.026 | **34.0** |
| **XLearning Lab** | 0.356 | 0.850 | 1.069 | 32.8 |
| **Outlier** | 0.361 | 0.844 | **4.219** | 32.6 |

观察：

1. **MAE 都不低**：BM_xTVC 的 MAE = 1.026，Outlier 甚至到 4.219。它们没有在 MAE 上做到极致。
2. **DES/PDS 都很高**：这是它们 Overall 高的原因。
3. **Outlier 的 MAE 崩了，Overall  still 32.6**：说明 MAE 对 Overall 的边际贡献远低于 PDS/DES。

本地模型对比：

| 模型 | DES | PDS | MAE |
|---|---|---|---|
| full_run | 0.133 | 0.526 | **0.036** |
| cs512 | 0.178 | 0.531 | 0.544 |
| state_lg | **0.198** | **0.560** | 0.507 |

full_run 的 MAE 极低，但 DES/PDS 也低；state_lg 的 MAE 高了 10 倍以上，但 DES/PDS 大幅提升。再次验证 trade-off。

---

## 5. 对优化策略的启示

如果允许牺牲 MAE 换其他指标，策略应该是：

1. **不要过度压缩预测幅度**（magnitude compression），否则 DES/PDS 会塌；
2. **优先保证 DE 基因排序正确**（↑ DES）；
3. **让不同扰动的预测模式拉开**（↑ PDS）；
4. **MAE 只要不跌破 baseline 太多即可**。

这也解释了为什么 VCC 榜单上顶尖队伍普遍选择“放大响应”而不是“压缩响应”。

---

## 6. 补充说明：DES 与 `overlap_at_N` 的区别

> 历史踩坑点，需要注意。

- **DES（严格生物学定义）**：通常指差异表达基因集合的某种交集或相似度。
- **`overlap_at_N`（cell-eval 实现）**：按 \(|\text{log2FC}|\) 取 top-k 后算交集比例。

在 VCC 榜单语境下，官方报告的 DES 对应的是 `overlap_at_N`。但计算 baseline 时不能混淆：

- `overlap_at_N` 的 baseline 约为 0.1065；
- 严格 DES（集合交集）的 baseline 约为 0.8919。

两者曾经混淆，导致 Overall Score 算错。现在统一使用 `overlap_at_N` 作为榜单口径的 DES。

---

## 7. 七个评测指标的计算公式（cell-eval 源码口径）

> 公式对照本地安装的 cell-eval 0.8.1 源码核实（`cell_eval/metrics/_de.py`、`_anndata.py`）。
> 记号约定：\(p\) = 扰动编号，\(g\) = 基因编号；\(\bar{y}^{\text{pert}}_{pg}\) = 扰动组 bulk 平均表达，\(\bar{y}^{\text{ctrl}}_{pg}\) = 对照组 bulk 平均表达；\(G = 18080\) 基因总数，\(N\) = 扰动总数；`log2FC` 和 `fdr` 来自扰动 vs 对照的 Wilcoxon 检验（BH 校正）。

### 7.0 全部 7 个指标一览

| 指标 | cell-eval 名称 | 方向 | 类别 | 一句话 |
|---|---|---|---|---|
| **DES** | `overlap_at_N` | ↑ | DE | top-k 差异基因找对的比例 |
| **PDS** | `discrimination_score_l1` | ↑ | ANNDATA | 预测响应最像哪个真实扰动 |
| **MAE** | `mae` | ↓ | ANNDATA | bulk 表达的平均绝对误差 |
| **SPEARMAN** | `de_spearman_sig` | ↑ | DE | 显著基因数量的跨扰动秩相关 |
| **SPEARMAN_LFC** | `de_spearman_lfc_sig` | ↑ | DE | 显著基因 log2FC 的秩相关 |
| **AUPRC** | `pr_auc` | ↑ | DE | 差异基因找回的 PR 曲线下面积 |
| **PEARSON** | `pearson_delta` | ↑ | ANNDATA | delta 表达向量的 Pearson 相关 |

### 7.1 DES：`overlap_at_N`

对单个扰动 \(p\)：

1. 计算每个基因的 log2FC：
   \[
   \text{log2FC}_{pg} = \log_2\left(\frac{\bar{y}^{\text{pert}}_{pg}}{\bar{y}^{\text{ctrl}}_{pg}}\right)
   \]
2. 真值侧：保留 \(\text{fdr}_{pg} < 0.05\) 的基因，按 \(|\text{log2FC}|\) 降序取前 \(k\) 个 → `real_topk`，其中 **\(k\) = 真值显著基因数**（`overlap_at_N` 的 N 指的就是它；固定 k 版本对应 `overlap_at_50/100/200/500`）。
3. 预测侧：同样按预测 log2FC 取前 \(k\) 个 → `pred_topk`。
4. 交集比例：
   \[
   \text{DES}_p = \frac{|\text{real\_topk} \cap \text{pred\_topk}|}{k}
   \]
5. 任一侧没有显著基因时该扰动记 0；最终对所有扰动求平均。

### 7.2 PDS：`discrimination_score_l1`

对单个扰动 \(p\)：

1. 把预测压缩成 bulk 效应向量（**delta = 扰动均值 − 对照均值**，不是 log2FC）：
   \[
   \hat{\delta}_p = \bar{y}^{\text{pred,pert}}_p - \bar{y}^{\text{pred,ctrl}}_p, \qquad
   \delta_q = \bar{y}^{\text{real,pert}}_q - \bar{y}^{\text{real,ctrl}}_q
   \]
2. 计算预测效应与所有真实效应的 L1 距离（**排除靶基因自身**）：
   \[
   d(p,q) = \sum_{g \neq \text{target}} |\hat{\delta}_{pg} - \delta_{qg}|
   \]
3. 距离升序排序，记真实扰动 \(p\) 的排名为 \(r_p\)（0-based，距离最小 → rank 0）：
   \[
   \text{PDS}_p = 1 - \frac{r_p}{N}
   \]
   - 最像自己：\(r_p = 0\) → \(\text{PDS}_p = 1\)
   - 随机水平：\(r_p \approx N/2\) → \(\text{PDS}_p \approx 0.5\)
4. 对所有扰动求平均。

### 7.3 MAE：`mae`

对单个扰动 \(p\)，直接在 **bulk 平均表达（绝对表达量，非 delta）** 上计算：
\[
\text{MAE}_p = \frac{1}{G} \sum_{g=1}^{G} \left| \bar{y}^{\text{pred}}_{pg} - \bar{y}^{\text{real}}_{pg} \right|
\]
最终对所有扰动求平均。

### 7.4 SPEARMAN：`de_spearman_sig`

⚠️ 名字容易误解：它**不是**“显著基因效果量的相关”，而是**显著基因数量的跨扰动秩相关**（源码 `DESpearmanSignificant`）。

1. 对每个扰动统计显著基因数：
   \[
   n_{\text{real}}(p) = |\{g : \text{fdr}^{\text{real}}_{pg} < 0.05\}|, \qquad
   n_{\text{pred}}(p) = |\{g : \text{fdr}^{\text{pred}}_{pg} < 0.05\}|
   \]
2. 对全体扰动做 Spearman 秩相关（全局唯一一个值）：
   \[
   \text{SPEARMAN} = \rho_S\big(\{n_{\text{real}}(p)\}_{p}, \{n_{\text{pred}}(p)\}_{p}\big)
   \]
3. 两侧都无显著基因时记 1.0。

### 7.5 SPEARMAN_LFC：`de_spearman_lfc_sig`

对单个扰动 \(p\)，在**真值显著基因集合** \(S_p = \{g : \text{fdr}^{\text{real}}_{pg} < 0.05\}\) 上：
\[
\text{SPEARMAN\_LFC}_p = \rho_S\big(\{\text{log2FC}^{\text{real}}_{pg}\}_{g \in S_p}, \{\text{log2FC}^{\text{pred}}_{pg}\}_{g \in S_p}\big)
\]
预测侧缺失的 log2FC 填 0；最终对所有扰动求平均。衡量的是“差异基因的幅度排序对不对”。

### 7.6 AUPRC：`pr_auc`

对单个扰动 \(p\)：
1. 标签 = 真值是否显著：\(\text{label}_{pg} = \mathbb{1}[\text{fdr}^{\text{real}}_{pg} < 0.05]\)
2. 分数 = 预测显著性的负对数：\(\text{score}_{pg} = -\log_{10}(\text{fdr}^{\text{pred}}_{pg})\)（fdr 越小越显著 → 分数越大）
3. 以分数排序计算精确率-召回率曲线下面积（sklearn `average_precision_score`）：
   \[
   \text{AUPRC}_p = \text{AP}(\{\text{label}_{pg}\}, \{\text{score}_{pg}\})
   \]
4. 最终对所有扰动求平均。衡量的是“把真正差异的基因排在前面”的整体排序质量，与 DES 的 top-k 视角互补。

### 7.7 PEARSON：`pearson_delta`

对单个扰动 \(p\)，在 **delta 表达**（扰动均值 − 对照均值）上：
\[
\Delta^{\text{real}}_{pg} = \bar{y}^{\text{real,pert}}_{pg} - \bar{y}^{\text{real,ctrl}}_{pg}, \qquad
\Delta^{\text{pred}}_{pg} = \bar{y}^{\text{pred,pert}}_{pg} - \bar{y}^{\text{pred,ctrl}}_{pg}
\]
\[
\text{PEARSON}_p = \rho_P\big(\{\Delta^{\text{real}}_{pg}\}_g, \{\Delta^{\text{pred}}_{pg}\}_g\big)
\]
最终对所有扰动求平均。衡量的是“响应形状（哪些基因涨、哪些跌）对不对”，对幅度不敏感。

### 7.8 归一化与 Overall Score（为什么 MAE 不重要）

cell-eval `score` 命令把所有指标对 baseline（扰动均值预测）归一化：

- **越小越好**（MAE）：\(\text{scaled} = 1 - \frac{\text{model}}{\text{baseline}}\)，小于 0 截断为 0；
- **越大越好**（DES/PDS/SPEARMAN 等）：\(\text{scaled} = \frac{\text{model} - \text{baseline}}{1 - \text{baseline}}\)，小于 0 截断为 0。

Overall 只取三个核心指标：
\[
S = \frac{\text{DES\_scaled} + \text{PDS\_scaled} + \text{MAE\_scaled}}{3} \times 100
\]

由于均值 baseline 的 MAE 极低（≈0.026），模型几乎不可能超过它 → MAE_scaled 基本为 0，这也是“MAE 退出竞争”的机制原因。

---

## 8. 附录：log2FC、fdr、L1 距离

理解 DES 和 PDS 需要先搞清楚三个基础概念。

### 8.1 log2 fold change（log2FC）

log2FC 衡量的是：**一个基因在扰动后，表达量相对于对照变化了多少倍**。

\[
\text{log2FC}_g = \log_2\left(\frac{\text{mean}(\text{perturbed}_g)}{\text{mean}(\text{control}_g)}\right)
\]

| 基因 | 对照平均表达 | 扰动后平均表达 | fold change | log2FC |
|---|---|---|---|---|
| A | 10 | 40 | 4 | \(\log_2(4) = +2\) |
| B | 40 | 10 | 0.25 | \(\log_2(0.25) = -2\) |
| C | 10 | 10 | 1 | \(\log_2(1) = 0\) |

- log2FC = +2：表达量变成原来的 4 倍（上调）；
- log2FC = -2：表达量变成原来的 1/4（下调）；
- log2FC = 0：没有变化。

取 log2 的好处是**上调和下调对称**：翻倍 = +1，减半 = -1。DES 按 \(|\text{log2FC}|\) 排序，找出变化最显著的基因。

### 8.2 fdr < 0.05

**fdr** = False Discovery Rate，错误发现率。它回答的问题是：

> 在我标记为“显著差异表达”的基因里，大概有多少是假的？

\[
\text{fdr} < 0.05
\]

意思是：**预计只有不到 5% 是被误报的**。

由于有 18080 个基因，即使基因实际上都没变，随机噪声也可能让约 900 个基因“碰巧”显著。fdr 就是控制这种假阳性的比例。DES 只保留 fdr < 0.05 的基因作为真正的差异表达基因。

### 8.3 L1 距离

L1 距离也叫 Manhattan 距离，衡量两个向量“相差多少”：

\[
L1(\hat{y}, y) = \sum_{g=1}^{G} |\hat{y}_g - y_g|
\]

就是**每个基因差的绝对值加起来**。

| 基因 | 预测值 | 真值 | 绝对差 |
|---|---|---|---|
| A | 12 | 10 | 2 |
| B | 8 | 10 | 2 |
| C | 10 | 10 | 0 |

\[
L1 = 2 + 2 + 0 = 4
\]

MAE 其实就是 L1 距离除以基因数。PDS 用 L1 距离判断预测响应和哪个真实扰动最接近。

---

## 9. 一句话总结

> **DES 看“有没有找对差异基因”，PDS 看“不同扰动像不像自己”，MAE 看“数值误差大不大”。提升 DES/PDS 往往需要放大扰动信号，而这会不可避免地提高 MAE。**
