#!/usr/bin/env python
"""bench_mem.py: 实测 bf16 + checkpoint 下 HVG 基因数 x batch 的显存/耗时"""
import sys
import time
import torch

sys.path.insert(0, "/home/zjh/scDFM")

from src.models.origin.model import model as OriginModel


def bench(n_genes, batch_size, use_checkpoint):
    try:
        model = OriginModel(
            fusion_method="differential_perceiver",
            perturbation_function="crisper",
            ntoken=20000,
            d_model=512,
            use_perturbation_interaction=False,
            mask_path=None,
        ).cuda()
        model.train()
        gene_id = torch.randint(0, 19000, (batch_size, n_genes)).cuda()
        c1 = torch.randn(batch_size, n_genes).cuda()
        c2 = torch.randn(batch_size, n_genes).cuda()
        t = torch.rand(batch_size).cuda()
        pid = torch.randint(0, 19000, (batch_size, 2)).cuda()

        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(gene_id, c1, t, c2, perturbation_id=pid, use_checkpoint=use_checkpoint)
            loss = out.float().mean()
            loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(gene_id, c1, t, c2, perturbation_id=pid, use_checkpoint=use_checkpoint)
                loss = out.float().mean()
                loss.backward()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        tag = "ckpt" if use_checkpoint else "no-ckpt"
        print(f"bf16+{tag} n_genes={n_genes:>5} batch={batch_size:>2} | peak_mem={peak:6.1f} GB | step={dt:5.2f} s", flush=True)
        del model, gene_id, c1, c2, t, pid, out, loss
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        tag = "ckpt" if use_checkpoint else "no-ckpt"
        print(f"bf16+{tag} n_genes={n_genes:>5} batch={batch_size:>2} | OOM", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for n, b in [(5000, 8), (5000, 16), (5000, 32), (8000, 8), (8000, 16)]:
        bench(n, b, use_checkpoint=True)
