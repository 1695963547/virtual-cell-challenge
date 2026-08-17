#!/usr/bin/env python
"""全面 benchmark：参数数 + 显存 + 训练速度 + 多卡效率

用法（单卡实测所有配置）:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python vcc/bench_full.py
"""
import os, sys, gc, time
sys.path.insert(0, '/home/zjh/scDFM')
os.chdir('/home/zjh/scDFM')

import torch
import torch.nn as nn
from src.models.instantiate_model import instantiate_model
from config.config_flow import FlowConfig

device = torch.device('cuda:0')

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_vram_and_speed(d_model, batch_size, infer_top_gene=1000, ntoken=18084, 
                           nlayers_label=None, n_steps=5, warmup=2):
    """实测：显存峰值 + 训练速度"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    mask_path = '/home/zjh/scDFM/data/vcc/mask_fold_0topk_30additive_negative_edge.pt'
    model = instantiate_model(
        'origin',
        ntoken=ntoken, d_model=d_model, 
        fusion_method='differential_perceiver',
        perturbation_function='crisper',
        mask_path=mask_path
    ).to(device)
    
    nlayers = len(model.blocks)
    if nlayers_label is None:
        nlayers_label = nlayers
    
    total_p, train_p = count_params(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 模拟训练数据
    gene_ids = torch.randint(0, infer_top_gene, (batch_size, infer_top_gene), device=device)
    source = torch.randn(batch_size, infer_top_gene, device=device)
    target = torch.randn(batch_size, infer_top_gene, device=device)
    t = torch.rand(batch_size, device=device)
    noise = torch.randn_like(target)
    pert_id = torch.randint(0, 100, (batch_size, 2), device=device)
    cell_line_id = torch.randint(0, 4, (batch_size,), device=device)
    
    # 模拟 forward+backward
    def one_step():
        optimizer.zero_grad()
        # 简化：直接 model forward（跳过 OT sampler / path 等，只测显存和速度）
        pred = model(gene_ids, source, t, noise, perturbation_id=pert_id, 
                     gene_id_all=gene_ids, mode='predict_y', cell_line_id=cell_line_id)
        loss = ((pred - target)**2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # warmup
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    
    # 测速 + 显存
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3  # GB
    model_vram = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    speed = elapsed / n_steps  # s/step
    
    # 清理
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'd_model': d_model,
        'batch_size': batch_size,
        'infer_top_gene': infer_top_gene,
        'nlayers': nlayers,
        'total_params_M': total_p / 1e6,
        'trainable_params_M': train_p / 1e6,
        'model_vram_GB': model_vram,
        'peak_train_vram_GB': peak_vram,
        'activation_vram_GB': peak_vram - model_vram,
        'speed_s_per_step': speed,
    }


def measure_eval_vram(d_model, eval_n_cells, n_genes=18080, batch_eval_size=48):
    """实测 eval 显存（inference mode, bf16）"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    mask_path = '/home/zjh/scDFM/data/vcc/mask_fold_0topk_30additive_negative_edge.pt'
    model = instantiate_model(
        'origin',
        ntoken=18084, d_model=d_model,
        fusion_method='differential_perceiver',
        perturbation_function='crisper',
        mask_path=mask_path
    ).to(device).eval()
    
    # bf16
    model = model.to(torch.bfloat16)
    
    gene_ids = torch.randint(0, 18084, (batch_eval_size, n_genes), device=device)
    source = torch.randn(batch_eval_size, n_genes, device=device, dtype=torch.bfloat16)
    noise = torch.randn_like(source)
    t = torch.rand(batch_eval_size, device=device, dtype=torch.bfloat16)
    pert_id = torch.randint(0, 100, (batch_eval_size, 2), device=device)
    cell_line_id = torch.randint(0, 4, (batch_eval_size,), device=device)
    
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        pred = model(gene_ids, noise, t, source, perturbation_id=pert_id,
                     gene_id_all=gene_ids, mode='predict_y', cell_line_id=cell_line_id)
    torch.cuda.synchronize()
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'd_model': d_model,
        'eval_n_cells': eval_n_cells,
        'n_genes': n_genes,
        'eval_batch': batch_eval_size,
        'peak_eval_vram_GB': peak_vram,
    }


print("=" * 80)
print("scDFM VCC 全面 Benchmark")
print("=" * 80)

# ============================================================
# Part 1: 不同 d_model 的参数数 + 显存 + 速度
# ============================================================
print("\n### Part 1: d_model 对比（batch_size=48, infer_top_gene=1000）")
print("-" * 80)

results = []
for dm in [128, 256, 384, 512]:
    try:
        r = measure_vram_and_speed(dm, batch_size=48, infer_top_gene=1000)
        results.append(r)
        print(f"d_model={dm:>3d} | params={r['total_params_M']:.1f}M | "
              f"model_vram={r['model_vram_GB']:.2f}GB | "
              f"peak_train={r['peak_train_vram_GB']:.2f}GB | "
              f"activation={r['activation_vram_GB']:.2f}GB | "
              f"speed={r['speed_s_per_step']:.2f}s/step")
    except Exception as e:
        print(f"d_model={dm:>3d} | OOM or Error: {e}")

# ============================================================
# Part 2: 不同 batch_size 的显存 + 速度（固定 d_model）
# ============================================================
print("\n### Part 2: batch_size 对比")
print("-" * 80)

for dm in [128, 512]:
    print(f"\n--- d_model={dm} ---")
    for bs in [24, 48, 64, 96, 128]:
        try:
            r = measure_vram_and_speed(dm, batch_size=bs, infer_top_gene=1000)
            print(f"  bs={bs:>3d} | peak={r['peak_train_vram_GB']:.2f}GB | "
                  f"activation={r['activation_vram_GB']:.2f}GB | "
                  f"speed={r['speed_s_per_step']:.2f}s/step")
        except Exception as e:
            print(f"  bs={bs:>3d} | OOM or Error: {e}")

# ============================================================
# Part 3: infer_top_gene 对比
# ============================================================
print("\n### Part 3: infer_top_gene 对比（d_model=128, bs=48）")
print("-" * 80)

for topg in [500, 1000, 2000, 3000]:
    try:
        r = measure_vram_and_speed(128, batch_size=48, infer_top_gene=topg)
        print(f"  top_gene={topg:>4d} | peak={r['peak_train_vram_GB']:.2f}GB | "
              f"activation={r['activation_vram_GB']:.2f}GB | "
              f"speed={r['speed_s_per_step']:.2f}s/step")
    except Exception as e:
        print(f"  top_gene={topg:>4d} | OOM or Error: {e}")

# ============================================================
# Part 4: Eval 显存（bf16, 全基因 18080）
# ============================================================
print("\n### Part 4: Eval 显存（bf16, 18080 全基因）")
print("-" * 80)

for dm in [128, 256, 512]:
    for ncells in [32, 48, 64, 128]:
        try:
            r = measure_eval_vram(dm, eval_n_cells=ncells, batch_eval_size=min(ncells, 48))
            print(f"  d_model={dm:>3d}, eval_cells={ncells:>3d} | "
                  f"peak_eval={r['peak_eval_vram_GB']:.2f}GB")
        except Exception as e:
            print(f"  d_model={dm:>3d}, eval_cells={ncells:>3d} | OOM or Error: {e}")

print("\n" + "=" * 80)
print("Benchmark 完成")
print("=" * 80)
