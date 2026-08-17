#!/usr/bin/env python
"""单卡极限 batch 实测：d_model=512 下 bs=160/192 能否塞进 142GB"""
import os, sys, gc, time
sys.path.insert(0, '/home/zjh/scDFM')
os.chdir('/home/zjh/scDFM')

import torch
from src.models.instantiate_model import instantiate_model

device = torch.device('cuda:0')

def measure(d_model, batch_size, infer_top_gene=1000, n_steps=4, warmup=2):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mask_path = '/home/zjh/scDFM/data/vcc/mask_fold_0topk_30additive_negative_edge.pt'
    model = instantiate_model('origin', ntoken=18084, d_model=d_model,
                              fusion_method='differential_perceiver',
                              perturbation_function='crisper', mask_path=mask_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    gene_ids = torch.randint(0, infer_top_gene, (batch_size, infer_top_gene), device=device)
    source = torch.randn(batch_size, infer_top_gene, device=device)
    target = torch.randn(batch_size, infer_top_gene, device=device)
    t = torch.rand(batch_size, device=device)
    noise = torch.randn_like(target)
    pert_id = torch.randint(0, 100, (batch_size, 2), device=device)
    cl_id = torch.randint(0, 4, (batch_size,), device=device)

    def one_step():
        optimizer.zero_grad()
        pred = model(gene_ids, source, t, noise, perturbation_id=pert_id,
                     gene_id_all=gene_ids, mode='predict_y', cell_line_id=cl_id)
        loss = ((pred - target)**2).mean()
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  d_model={d_model} bs={batch_size:>3d} | peak={peak:.2f}GB | speed={elapsed/n_steps:.2f}s/step")
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

print("=== 单卡 142GB 极限测试（d_model=512, infer_top_gene=1000）===")
for bs in [128, 160, 192]:
    try:
        measure(512, bs)
    except Exception as e:
        print(f"  d_model=512 bs={bs:>3d} | FAIL: {type(e).__name__}: {str(e)[:120]}")
print("=== 完成 ===")
