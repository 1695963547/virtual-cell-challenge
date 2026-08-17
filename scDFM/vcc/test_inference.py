"""
VCC 最终评测推理脚本——对 adata_Test.h5ad（100 扰动）生成预测并评测
用法:
    torchrun --nproc_per_node=3 vcc/test_inference.py \
        --checkpoint_path=./result/vcc/<exp>/iteration_20000/checkpoint.pt \
        [--n_cells=128] [--ode_steps=10] [--bf16]
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
import torchdiffeq
from tqdm import trange, tqdm
from accelerate import Accelerator

# 保证 scDFM 模块可导入（解析为 scDFM 根目录）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # 确保相对路径的 mask/vocab 能找到

from config.config_flow import FlowConfig
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.utils.utils import make_lognorm_poisson_noise

from cell_eval import MetricsEvaluator


# ────────────────────── 工具函数 ──────────────────────

def wrapped_vf(target, t, source, perturbation_id, vf, gene_ids, gene_all,
               cell_line_id=None, device=None):
    gene = gene_ids.repeat(source.shape[0], 1).to(device)
    if next(vf.parameters()).dtype == torch.bfloat16:
        target = target.to(torch.bfloat16)
        source = source.to(torch.bfloat16)
        t = t.to(torch.bfloat16)
    predicted_x_t_velocity = vf(gene, target, t, source, perturbation_id,
                                gene_all, cell_line_id=cell_line_id)
    if predicted_x_t_velocity.dtype == torch.bfloat16:
        predicted_x_t_velocity = predicted_x_t_velocity.float()
    return predicted_x_t_velocity


@torch.no_grad()
def generate_sample(source, perturbation_id, vf, gene_ids, gene_all,
                    steps=10, cell_line_id=None, device=None, noise_type='Gaussian'):
    if noise_type == 'Gaussian':
        target_noise = torch.randn(source.shape[0], source.shape[1], device=source.device)
    elif noise_type == 'Poisson':
        target_noise = make_lognorm_poisson_noise(source, alpha=0.8, per_cell_L=1e4)
    traj = torchdiffeq.odeint(
        lambda t, x: wrapped_vf(x, t, source, perturbation_id, vf,
                                gene_ids, gene_all, cell_line_id=cell_line_id, device=device),
        target_noise,
        torch.linspace(0, 1, steps).to(source.device),
        atol=1e-4, rtol=1e-4, method='rk4',
    )
    return torch.clamp(traj[-1], min=0)


# ────────────────────── 主流程 ──────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--test_h5ad', type=str, default='/home/zjh/vcc_official/adata_Test.h5ad')
    parser.add_argument('--output_dir', type=str, default='./result/vcc/test_eval')
    parser.add_argument('--n_cells', type=int, default=128, help='每个扰动生成的预测细胞数')
    parser.add_argument('--ode_steps', type=int, default=10)
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_top_genes', type=int, default=18080)
    parser.add_argument('--ntoken', type=int, default=18084)
    parser.add_argument('--batch_size', type=int, default=128, help='ODE 生成时的批大小')
    parser.add_argument('--mask_path', type=str,
                        default='./data/vcc/mask_fold_0topk_30additive_negative_edge.pt')
    parser.add_argument('--vocab_path', type=str,
                        default='./src/tokenizer/vcc_18080_highly_vocab.json')
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index
    world = accelerator.num_processes
    device = accelerator.device

    # ── 1. 加载模型 ──
    if accelerator.is_main_process:
        print(f"[rank {rank}] Loading model (d_model={args.d_model})...")
    vf = instantiate_model(
        model_type='origin', ntoken=args.ntoken, d_model=args.d_model,
        d_perturbation=args.d_model, fusion_method='differential_perceiver',
        perturbation_function='crisper', mask_path=args.mask_path,
    )
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    vf.load_state_dict(ckpt['model_state_dict'])
    if accelerator.is_main_process:
        print(f"[rank {rank}] Loaded checkpoint at iteration {ckpt.get('iteration', '?')}")
    vf = vf.to(device)
    vf.eval()

    if args.bf16:
        vf = vf.to(torch.bfloat16)

    # ── 2. 加载 vocab ──
    vocab = GeneVocab.from_file(args.vocab_path)

    # ── 3. 加载 & 预处理 test 数据 ──
    if accelerator.is_main_process:
        print(f"[rank {rank}] Loading test data: {args.test_h5ad}")
    adata_test = sc.read_h5ad(args.test_h5ad)
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)

    # 基因 ID（与训练一致的全 18080 基因）
    gene_ids = torch.tensor(vocab.encode(list(adata_test.var_names)),
                            dtype=torch.long, device=device)

    # 扰动列表：100 个官方扰动 + non-targeting（作为 control）
    all_perturbations = sorted(adata_test.obs['target_gene'].unique().tolist())
    control_pert = 'non-targeting'
    test_perturbations = [p for p in all_perturbations if p != control_pert]
    if accelerator.is_main_process:
        print(f"[rank {rank}] Total perturbations: {len(test_perturbations)}, "
              f"control ({control_pert}): "
              f"{(adata_test.obs['target_gene'] == control_pert).sum()} cells")

    # ── 4. DDP 分片 ──
    my_perturbations = test_perturbations[rank::world]
    if accelerator.is_main_process:
        print(f"[rank {rank}] My perturbations: {len(my_perturbations)}")

    # ── 5. 获取 control cells ──
    ctrl_adata = adata_test[adata_test.obs['target_gene'] == control_pert]
    ctrl_X = torch.from_numpy(ctrl_adata.X.toarray()).float()  # (N_ctrl, 18080)

    # ── 6. 逐扰动生成预测 ──
    all_pred = []
    all_real = []
    obs_pred = []
    obs_real = []

    # 每个 rank 的 pred/real 都包含 control cells（评测需要）
    all_pred.append(ctrl_X.numpy())
    all_real.append(ctrl_X.numpy())
    obs_pred.extend([control_pert] * ctrl_X.shape[0])
    obs_real.extend([control_pert] * ctrl_X.shape[0])

    for pert_name in tqdm(my_perturbations, desc=f"[rank {rank}] Generating",
                          disable=not accelerator.is_main_process):
        pert_adata = adata_test[adata_test.obs['target_gene'] == pert_name]
        real_X = pert_adata.X.toarray()

        # source = control cells, 随机采样 n_cells 个
        idx = torch.randperm(ctrl_X.shape[0])[:args.n_cells]
        source = ctrl_X[idx].to(device)

        # crisper 编码：target_gene → vocab token → [n_cells, 1]
        pert_ids = vocab.encode([pert_name])  # e.g. [6360]
        pert_id = torch.tensor(pert_ids, dtype=torch.long, device=device)
        pert_id_batch = pert_id.repeat(source.shape[0], 1)

        # cell_line_id = 0 (H1, 与 val 一致)
        cell_line_id = torch.zeros(source.shape[0], dtype=torch.long, device=device)

        # 分批生成（避免 OOM）
        preds = []
        for i in range(0, source.shape[0], args.batch_size):
            src_batch = source[i:i + args.batch_size]
            pid_batch = pert_id_batch[i:i + args.batch_size]
            cl_batch = cell_line_id[i:i + args.batch_size]
            pred_expr = generate_sample(
                src_batch, pid_batch, vf, gene_ids, gene_ids,
                steps=args.ode_steps, cell_line_id=cl_batch,
                device=device, noise_type='Gaussian',
            )
            preds.append(pred_expr.cpu().numpy())

        pred_X = np.concatenate(preds, axis=0)

        all_pred.append(pred_X)
        all_real.append(real_X)
        obs_pred.extend([pert_name] * pred_X.shape[0])
        obs_real.extend([pert_name] * real_X.shape[0])

    # ── 7. 构建 AnnData ──
    pred_adata = ad.AnnData(
        X=np.concatenate(all_pred, axis=0).astype(np.float32),
        obs=pd.DataFrame({'perturbation': obs_pred}),
    )
    real_adata = ad.AnnData(
        X=np.concatenate(all_real, axis=0).astype(np.float32),
        obs=pd.DataFrame({'perturbation': obs_real}),
    )

    # ── 8. 评测 ──
    os.makedirs(args.output_dir, exist_ok=True)
    evaluator = MetricsEvaluator(
        adata_pred=pred_adata, adata_real=real_adata,
        control_pert=control_pert, pert_col='perturbation',
        num_threads=32,
    )
    results, agg_results = evaluator.compute()

    results.write_csv(os.path.join(args.output_dir, f'results_rank{rank}.csv'))
    agg_results.write_csv(os.path.join(args.output_dir, f'agg_results_rank{rank}.csv'))
    pred_adata.write_h5ad(os.path.join(args.output_dir, f'pred_rank{rank}.h5ad'))
    real_adata.write_h5ad(os.path.join(args.output_dir, f'real_rank{rank}.h5ad'))

    # 打印分数（agg_results 含 count/null_count/mean/std 4 行，取 mean 行）
    df = agg_results.to_pandas()
    df = df[df['statistic'] == 'mean']
    mse = float(df['mse'].iloc[0])
    pearson_delta = float(df['pearson_delta'].iloc[0]) if 'pearson_delta' in df.columns else None
    print(f"[rank {rank}] MSE={mse:.4f}"
          + (f", pearson_delta={pearson_delta:.4f}" if pearson_delta is not None else ""))

    # ── 9. 等待所有 rank 完成，rank 0 合并 ──
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("[main] Merging per-rank outputs...")
        # 合并 pred/real h5ad
        preds = [ad.read_h5ad(os.path.join(args.output_dir, f'pred_rank{r}.h5ad'))
                 for r in range(world)]
        reals = [ad.read_h5ad(os.path.join(args.output_dir, f'real_rank{r}.h5ad'))
                 for r in range(world)]
        merged_pred = ad.concat(preds, axis=0)
        merged_real = ad.concat(reals, axis=0)
        merged_pred.write_h5ad(os.path.join(args.output_dir, 'pred.h5ad'))
        merged_real.write_h5ad(os.path.join(args.output_dir, 'real.h5ad'))

        # 合并 results CSV
        import polars as pl
        all_results = pl.concat([pl.read_csv(os.path.join(args.output_dir, f'results_rank{r}.csv'))
                                 for r in range(world)])
        all_results.write_csv(os.path.join(args.output_dir, 'results.csv'))

        # 用合并后的数据重新算聚合指标
        merged_evaluator = MetricsEvaluator(
            adata_pred=merged_pred, adata_real=merged_real,
            control_pert=control_pert, pert_col='perturbation',
            num_threads=32,
        )
        _, merged_agg = merged_evaluator.compute()
        merged_agg.write_csv(os.path.join(args.output_dir, 'agg_results.csv'))
        merged_df = merged_agg.to_pandas()
        merged_df = merged_df[merged_df['statistic'] == 'mean']
        print(f"\n{'='*60}")
        print(f"Final aggregated metrics ({len(test_perturbations)} perturbations):")
        for col in merged_df.columns:
            if col == 'statistic':
                continue
            print(f"  {col}: {float(merged_df[col].iloc[0]):.4f}")
        print(f"{'='*60}")
        print(f"Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
