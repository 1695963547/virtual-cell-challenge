# scDFM 复现虚拟细胞挑战赛（VCC）改动记录

> 目的：记录把 ICLR 2026 开源的 scDFM（distributional flow matching，条件生成模型）应用到虚拟细胞挑战赛（VCC, Virtual Cell Challenge）过程中所做的全部改动，便于复现、审计与后续优化。
> 时间跨度：2026-08-12 ~ 2026-08-18

---

## 1. 任务背景

VCC 的任务是：给定 control（non-targeting）细胞 + 一个 perturbation（目标基因敲除），预测扰动后细胞的基因表达分布。官方 test（`adata_Test.h5ad`）的 obs 仅含 `target_gene` / `guide_id` / `batch`（**无细胞系标注**），其中 non-targeting control 共 38,176 细胞，本地推理固定按 H1（`cell_line_id=0`）处理。官方评测 Overall = 1/3 × (DES + PDS + MAE)。

- 官方数据：训练为 **6 个 h5 文件**（`competition_train` / `k562_gwps` / `rpe1` / `jurkat` / `k562` / `hepg2`）、官方 val（50 扰动）、官方 test（100 扰动）
- 本地评测口径：cell-eval 的 `MetricsEvaluator`，输出 7 项指标（DES/PDS/MAE/SPEARMAN/SPEARMAN_LFC/AUPRC/PEARSON）
- 复现硬件：H20 × 4 卡（GPU 0-3），Python venv `/home/zjh/state-vcc-local/state/.venv`

scDFM 的本地 VCC 配置：`d_model=512, batch=256, lr=1e-4, steps=20000`，flow-fusion 模型 + crisper 扰动编码 + differential_perceiver 融合。

---

## 2. 改动总览

### 2.1 新增文件（`vcc/` 目录）

| 文件 | 作用 |
|---|---|
| `vcc/build_scdfm_data.py` | 构建 VCC 训练数据 `vcc_train.h5ad` |
| `vcc/build_scdfm_val.py` | 构建 VCC 验证数据 `vcc_val.h5ad` |
| `vcc/train_cp10k.sh` | 4 卡分布式训练启动脚本（最终配置） |
| `vcc/train_and_eval.sh` | 训练 + 自动选 checkpoint + 测试集推理 流水线 |
| `vcc/test_inference.py` | 测试集推理 + 7 项指标评测脚本 |
| `vcc/smoke_test.sh` | 单卡 3 步冒烟测试 |
| `vcc/auto_launch_inference.sh` | 自动启动推理 |
| `vcc/gpu_watch_train.sh` | 训练期间 GPU 监控 |
| `vcc/bench_*.py` | 性能基准（amp/bs/ddp/mem/train_eval 等） |
| `vcc/sdpa_*.sh`, `vcc/test_sdpa.py` | SDPA 环境问题调试 |

### 2.2 修改的源码文件

| 文件 | 改动点 |
|---|---|
| `config/config_flow.py` | 新增 VCC 相关参数（见 6 节） |
| `src/data_process/data.py` | 数据加载/预处理/cell line 分层采样（见 3 节） |
| `src/models/instantiate_model.py` | 支持 `differential_perceiver` 融合 |
| `src/models/origin/model.py` | 新增 cell line 编码器（见 4 节） |
| `src/models/origin/blocks.py` | 18080 全基因下的显存优化（数学等价改写） |
| `src/tokenizer/` | 新增 `vcc_18080_highly_vocab.json`（18080 基因词表） |
| `src/script/run.py` | bf16 训练、并行 eval、checkpoint 恢复等（见 5 节） |
| `src/utils/utils.py` | `pick_eval_score`、checkpoint 存取适配 |

---

## 3. 数据层改动

### 3.1 数据构建（`build_scdfm_data.py` / `build_scdfm_val.py`）
- 将官方 6 个文件中的 **5 个合并**（`competition_train` / `k562_gwps` / `rpe1` / `jurkat` / `k562`），得到 **4 个细胞系：H1 / K562 / RPE1 / Jurkat**
- **`hepg2` 留出**（不参与训练）：与 state 路线（`starter.toml` 的 `replogle_h1`）一致——state 里 hepg2 整系作为 zeroshot/test，同样不参与训练
- 官方 val 转为 `vcc_val.h5ad`
- 训练数据规模：**39.4 万细胞、197 个扰动 + control（共 198 个 condition）**，分布：H1 221,193（149 扰动）/ K562 129,762（177 扰动）/ RPE1 21,990（58 扰动）/ Jurkat 21,024（56 扰动）
- 与 state 的唯一差异在**拆分方式**：state 为各系内 train/val/test 划分 + hepg2 整系 zeroshot；scDFM 用 `split_method=additive`（按扰动划 train/val/test），hepg2 同样不在训练集内

### 3.2 预处理（`src/data_process/data.py`）
- **归一化口径统一为 log1p(CP10K)**：`normalize_total(1e4) + log1p`，训练/推理完全一致（修复早期 train/test 归一化不一致的问题）
- **全基因模式**：`n_top_genes=18080`，不选 HVG，`highly_variable` 全置 True
- **扰动维度适配**：VCC 是单基因 knockout，`condition` 列替代 norman 的 `Drug1+Drug2` 组合；control 名从 `'control+control'` 改为 `'control'`
- **cell line 分层采样**：训练时 source（control）必须与 target 扰动同 cell line（`TrainSampler`，按 `cell_line_id` 分层），避免跨系错配

### 3.3 词表
- 新建 `vcc_18080_highly_vocab.json`：18080 个基因 + `control` 作为扰动 token

---

## 4. 模型层改动

### 4.1 cell line 条件编码（`src/models/origin/model.py`）
```python
self.cell_line_encoder = BatchLabelEncoder(n_cell_line, d_model)  # VCC: H1/K562/RPE1/Jurkat
```
新增 cell line 标签编码器，把细胞系信息作为条件注入模型。

### 4.2 显存优化（`src/models/origin/blocks.py`）
- 18080 全基因下，显式构造注意力矩阵约需 **~500GB 显存，必然 OOM**
- 利用 matmul 线性性改写为等价实现（不损失精度），使其可在 H20 上运行

### 4.3 融合方式
- 新增 `differential_perceiver` 融合路径（`instantiate_model.py`），替代/补充原版 `differential_transformer`
- 扰动 mask：`mask_fold_0topk_30additive_negative_edge.pt`（topk=30、additive 拆分、负边 mask）

---

## 5. 训练层改动（`src/script/run.py`）

所有改动均带 `# VCC:` 注释标记：

1. **bf16 AMP 训练**：前向用 `torch.autocast('cuda', dtype=torch.bfloat16)`，loss 与 MMD 保持 fp32。H20 的 FP32 算力弱而 BF16 tensor core 强，实测 **加速 2.6-2.8x**
2. **eval 子集监控**（`eval_subset_perturbations=15`）：训练中监控 eval 只跑 15 个等间隔扰动子集，保持代表性同时显著提速
3. **3 卡并行 eval**：扰动按 rank 交错分片，各 rank 只生成/评测自己的分片（原版 3 卡重复跑全量，浪费 3x）
4. **val 单系 control 过滤**：val 为单一 cell line（H1）时，评测基准只用该系 control
5. **source 与 target 同系**：训练/val 的 source control 取与 target 同 cell line 的 control（`TrainSampler` 按 `cell_line_id` 分层采样）；test 推理无系标注，固定按 H1（`cell_line_id=0`）
6. **eval_n_cells**：`N = min(config.eval_n_cells, source.shape[0])`
7. **cell_line_id 截断后构造**：必须在 source 截断后构造，否则长度不匹配
8. **每 rank 独立评测写 rank 后缀文件**，主进程汇总各 rank 分数打印
9. **bf16 eval**（`wrapped_vf`）：模型已是 bf16，浮点输入转 bf16、输出转回 fp32（odeint 需 fp32），开关 `--eval_bf16`
10. **固定随机种子**：`seed + process_index`，避免 DDP 各卡采样完全一致
11. **恢复训练**：`load_checkpoint` 返回 iteration，从断点继续
12. **循环体内主动 break**：VCC epoch 有数千个 batch，`steps` 到了不会自然退出，必须主动 break
13. **训练结束保存最终 checkpoint**：原版只在 `iteration % print_every == 0` 时保存，若 `steps` 不是 `print_every` 倍数会丢掉最后一步的权重

---

## 6. 最终训练配置（`vcc/train_cp10k.sh`）

```bash
torchrun --nproc_per_node=4 src/script/run.py \
  --batch_size=256 --model_type=origin --d_model=512 --ntoken=18084 --lr=1e-4 --steps=20000 --eta_min=1e-06 \
  --data_name=vcc --perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
  --result_path=./result/vcc_cp10k --fusion_method=differential_perceiver \
  --infer_top_gene=1000 --n_top_genes=18080 --gamma=0.5 --split_method=additive \
  --use_mmd_loss --fold=0 --use_negative_edge --topk=30 \
  --eval_n_cells=64 --eval_ode_steps=10 --eval_bf16 --eval_subset_perturbations=15 --seed=42
```

关键超参：`d_model=512, batch=256, lr=1e-4, 20000 步, gamma=0.5, topk=30 负边 mask, fold=0`

---

## 7. 推理/评测层（`vcc/test_inference.py`，独立新增）

- 加载训练好的 flow-fusion 模型 + 18080 词表
- 读官方测试集 `adata_Test.h5ad`（100 扰动），按 `perturbations[rank::world]` DDP 分片
- 每个扰动从 test 内 non-targeting control（38,176 细胞）随机采样 128 个出发（官方 test 无细胞系标注，`cell_line_id` 固定 0=H1），`torchdiffeq.odeint`（RK4, 10 步, atol/rtol=1e-4）生成 **128 个预测细胞**
- 生成后调 `cell_eval.MetricsEvaluator`（32 线程）算 **7 项指标**（DES/PDS/MAE/SPEARMAN/SPEARMAN_LFC/AUPRC/PEARSON），输出 `agg_results.csv`
- bf16 前向 + fp32 输出，保证 ODE 数值精度

### 7.1 推理耗时（已知瓶颈）
- 每扰动 ~461s（40 次模型前向：10 ODE 间隔 × 4 次 RK4 stage；每次前向是 (128, 18084, 512) 的 transformer）
- 3 卡分片 34/33/33，总 ~4.4h；已实测 `torch.compile` 无收益（1.02x，GPU 已满载）
- 最优配置：**从零开始时直接 4 卡**（省 27%）

---

## 8. 环境适配与已知问题

| 问题 | 处理 |
|---|---|
| H20（sm90）FP32 算力弱 | 训练/推理 bf16，实测加速 2.6-2.8x |
| cu130 无 flash-attn wheel | 不装 flash-attn，用标准 MHA（SDPA 已自动走 fused kernel） |
| SDPA 非法访问（早期） | `test_sdpa.py` + `sdpa_workaround_test*.sh` 定位，冒烟测试 `smoke_test.sh` 通过后恢复 |
| 18080 全基因注意力显存 OOM | matmul 线性性改写（数学等价） |
| 训练/推理归一化不一致 | 统一 log1p(CP10K) |
| checkpoint 选择 bug | `train_and_eval.sh` 原 `sort -t_ -k2 -n` 随机选 checkpoint，修复为 `sort -V` 取最大迭代号；本次推理改用训练中 eval 分数最优的 `iteration_15000` |

---

## 9. 评测结果参照

| 模型 | DES | PDS | MAE | SPEARMAN | LFC | AUPRC | PEARSON |
|---|---|---|---|---|---|---|---|
| baseline_train | 0.107 | 0.509 | 0.026 | — | — | — | — |
| cleopatra（Generalist #1, flow matching 同路线） | 0.228 | 0.747 | 0.086 | 0.473 | 0.396 | 0.266 | 0.203 |
| BM_xTVC（官方 #1） | 0.349 | 0.872 | 1.026 | — | — | — | — |
| **state_lg（本地 state 路线最佳，301M）** | **0.198** | **0.560** | 0.507 | **0.284** | 0.146 | 0.213 | 0.051 |
| **scDFM（iter15000，本仓库）** | 0.084 | 0.503 | **0.042** | 0.016 | **0.388** | 0.127 | 0.054 |

> scDFM 最终推理结果（`result/vcc/test_eval_iter15000/agg_results.csv`，100 扰动合并评估，08-18 00:21 完成）：DES 0.0835 / PDS 0.5028 / MAE 0.0419 / SPEARMAN 0.0163 / SPEARMAN_LFC 0.3883 / AUPRC 0.1269 / PEARSON 0.0544。MAE（0.042）与 SPEARMAN_LFC（0.388）为本地各模型中最强，DES（0.084）/ SPEARMAN（0.016）为最大短板（预测显著基因数量膨胀至真值 ~5 倍，扰动特异性不足）。
>
> state_lg 结果（本地 state 路线最佳，`state-vcc-local` 的 `state_emb_all_lg_cs512`，301M 参数、cs512、8000 步）：DES 0.198 / PDS 0.560 / MAE 0.507 / SPEARMAN 0.284 / SPEARMAN_LFC 0.146 / AUPRC 0.213 / PEARSON 0.051，本地 Overall 估算 6.68。state 系列其余配置（cs512 0.178/0.531/0.544/0.233/0.229/0.207/0.099、SE旧 0.175/0.521/0.353/0.175/0.135/0.205/0.073、full_run 0.133/0.526/0.036/0.081/0.093/0.237/0.091）详见 `结果对比.md`。
>
> 两路线对比：scDFM 的 MAE（0.042 vs 0.507）与 SPEARMAN_LFC（0.388 vs 0.146）显著优于 state_lg，但 DES（0.084 vs 0.198）、PDS（0.503 vs 0.560）、SPEARMAN（0.016 vs 0.284）、AUPRC（0.127 vs 0.213）、PEARSON（0.054 vs 0.051，基本持平）落后——state_lg 在 DE 基因检出/排序上更强，scDFM 在表达量幅度与幅度排序上更强。

---

## 10. 可复现性说明

- 所有脚本在 `vcc/` 目录，数据在 `vcc/data/`（`vcc_train.h5ad`、`vcc_val.h5ad`）
- 训练日志：`/home/zjh/log/scdfm_vcc_train_cp10k.log`
- checkpoint：`result/vcc_cp10k/<experiment>/iteration_{5000,10000,15000,20000}/checkpoint.pt`
- 推理日志：`/home/zjh/log/scdfm_vcc_test_eval_best_iter15000.log`
- 结果：`result/vcc/test_eval_iter15000/`（7 项指标 `agg_results.csv`）
