#!/usr/bin/env python
"""多卡 DDP 效率测试（单配置，不同卡数）"""
import os, sys, time, argparse
sys.path.insert(0, '/home/zjh/scDFM')
os.chdir('/home/zjh/scDFM')

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from src.models.instantiate_model import instantiate_model

def run_benchmark(rank, world_size, d_model, batch_size, n_steps=20, warmup=5):
    dist.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    mask_path = '/home/zjh/scDFM/data/vcc/mask_fold_0topk_30additive_negative_edge.pt'
    model = instantiate_model(
        'origin', ntoken=18084, d_model=d_model,
        fusion_method='differential_perceiver',
        perturbation_function='crisper', mask_path=mask_path
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 每个 rank 处理自己的 batch
    gene_ids = torch.randint(0, 1000, (batch_size, 1000), device=device)
    source = torch.randn(batch_size, 1000, device=device)
    target = torch.randn(batch_size, 1000, device=device)
    t = torch.rand(batch_size, device=device)
    noise = torch.randn_like(target)
    pert_id = torch.randint(0, 100, (batch_size, 2), device=device)
    cell_line_id = torch.randint(0, 4, (batch_size,), device=device)
    
    def one_step():
        optimizer.zero_grad()
        pred = model(gene_ids, source, t, noise, perturbation_id=pert_id,
                     gene_id_all=gene_ids, mode='predict_y', cell_line_id=cell_line_id)
        loss = ((pred - target)**2).mean()
        loss.backward()
        optimizer.step()
    
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()
    
    t0 = time.time()
    for _ in range(n_steps):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.time() - t0
    
    speed = elapsed / n_steps
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    
    if rank == 0:
        print(f"  {world_size} GPU | speed={speed:.2f}s/step | peak_vram={peak_vram:.2f}GB/GPU | "
              f"throughput={world_size/speed:.1f} samples/s (total {world_size}*{batch_size})")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=48)
    args = parser.parse_args()
    
    rank = int(os.environ['RANK'])
    run_benchmark(rank, args.world_size, args.d_model, args.batch_size)
