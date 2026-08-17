#!/usr/bin/env python
"""混合精度对比 + eval 前向耗时实测（H20 关键数据）"""
import os, sys, gc, time
sys.path.insert(0, '/home/zjh/scDFM')
os.chdir('/home/zjh/scDFM')

import torch
from src.models.instantiate_model import instantiate_model

device = torch.device('cuda:0')
MASK = '/home/zjh/scDFM/data/vcc/mask_fold_0topk_30additive_negative_edge.pt'

def train_step_bench(d_model, batch_size, use_amp, n_steps=8, warmup=3, top_gene=1000):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = instantiate_model('origin', ntoken=18084, d_model=d_model,
                              fusion_method='differential_perceiver',
                              perturbation_function='crisper', mask_path=MASK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    gene_ids = torch.randint(0, top_gene, (batch_size, top_gene), device=device)
    source = torch.randn(batch_size, top_gene, device=device)
    target = torch.randn(batch_size, top_gene, device=device)
    t = torch.rand(batch_size, device=device)
    noise = torch.randn_like(target)
    pert_id = torch.randint(0, 100, (batch_size, 2), device=device)
    cl_id = torch.randint(0, 4, (batch_size,), device=device)

    def one_step():
        opt.zero_grad()
        if use_amp:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(gene_ids, source, t, noise, perturbation_id=pert_id,
                             gene_id_all=gene_ids, mode='predict_y', cell_line_id=cl_id)
        else:
            pred = model(gene_ids, source, t, noise, perturbation_id=pert_id,
                         gene_id_all=gene_ids, mode='predict_y', cell_line_id=cl_id)
        loss = ((pred.float() - target)**2).mean()
        loss.backward()
        opt.step()
        return loss.item()

    losses = []
    for _ in range(warmup):
        losses.append(one_step())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_steps):
        losses.append(one_step())
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    label = 'bf16-AMP' if use_amp else 'fp32'
    print(f"  {label:>9s} d_model={d_model} bs={batch_size:>3d} | speed={elapsed/n_steps:.2f}s/step "
          f"| peak={peak:.2f}GB | loss={losses[-1]:.4f}")
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return elapsed / n_steps

def eval_forward_bench(d_model, n_cells, top_gene=18080, use_bf16=True):
    """eval 全基因前向单次耗时（bf16）"""
    gc.collect(); torch.cuda.empty_cache()
    model = instantiate_model('origin', ntoken=18084, d_model=d_model,
                              fusion_method='differential_perceiver',
                              perturbation_function='crisper', mask_path=MASK).to(device).eval()
    if use_bf16:
        model = model.to(torch.bfloat16)
    gene_ids = torch.randint(0, 18084, (n_cells, top_gene), device=device)
    source = torch.randn(n_cells, top_gene, device=device,
                         dtype=torch.bfloat16 if use_bf16 else torch.float32)
    noise = torch.randn_like(source)
    t = torch.rand(n_cells, device=device, dtype=torch.bfloat16 if use_bf16 else torch.float32)
    pert_id = torch.randint(0, 100, (n_cells, 2), device=device)
    cl_id = torch.randint(0, 4, (n_cells,), device=device)

    with torch.inference_mode():
        # warmup
        for _ in range(2):
            model(gene_ids, noise, t, source, perturbation_id=pert_id,
                  gene_id_all=gene_ids, mode='predict_y', cell_line_id=cl_id)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            model(gene_ids, noise, t, source, perturbation_id=pert_id,
                  gene_id_all=gene_ids, mode='predict_y', cell_line_id=cl_id)
        torch.cuda.synchronize()
    per_call = (time.time() - t0) / 5
    dtype = 'bf16' if use_bf16 else 'fp32'
    print(f"  eval fwd {dtype} d_model={d_model} cells={n_cells:>3d} x {top_gene} genes | {per_call:.2f}s/call")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return per_call

print("=" * 70)
print("Part 1: 混合精度训练对比（fp32 vs bf16-AMP）")
print("=" * 70)
t_fp32_256 = train_step_bench(256, 48, use_amp=False)
t_bf16_256 = train_step_bench(256, 48, use_amp=True)
print(f"  => d_model=256 加速比: {t_fp32_256/t_bf16_256:.2f}x")
t_fp32_512 = train_step_bench(512, 64, use_amp=False)
t_bf16_512 = train_step_bench(512, 64, use_amp=True)
print(f"  => d_model=512 加速比: {t_fp32_512/t_bf16_512:.2f}x")

print()
print("=" * 70)
print("Part 2: eval 全基因(18080)前向单次耗时")
print("=" * 70)
c_256 = eval_forward_bench(256, 48)
c_512 = eval_forward_bench(512, 48)
print()
print("估算（每扰动 = ode_steps x ceil(cells/48) 次前向）：")
print(f"  d_model=256, 32cells x 10steps = 10 次前向 x {c_256:.2f}s = {10*c_256:.1f}s/扰动")
print(f"  d_model=256, 128cells x 20steps = 60 次前向 x {c_256:.2f}s = {60*c_256:.1f}s/扰动")
print(f"  d_model=512, 32cells x 10steps = 10 次前向 x {c_512:.2f}s = {10*c_512:.1f}s/扰动")
print(f"  d_model=512, 128cells x 20steps = 60 次前向 x {c_512:.2f}s = {60*c_512:.1f}s/扰动")
print(f"  50 扰动 x d_model=512 严格(128x20) = {50*60*c_512/3600:.1f} 小时")
print(f"  50 扰动 x d_model=512 监控(32x10) = {50*10*c_512/3600:.1f} 小时")
print("=" * 70)
print("完成")
