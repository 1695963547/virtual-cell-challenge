#!/usr/bin/env python
"""正式训练/评测速度实测（独占 GPU）——用于训练规模估算

用法:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python vcc/bench_train_eval.py \
      --train_steps=10 --eval_perturbations=1 --eval_n_cells=128 --eval_ode_steps=20
"""
import os, sys, time, argparse
sys.path.insert(0, '/home/zjh/scDFM')
os.chdir('/home/zjh/scDFM')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from config.config_flow import FlowConfig
from src.data_process.data import Data, PerturbationDataset
from src.models.instantiate_model import instantiate_model
from src.utils.utils import process_vocab, set_requires_grad_for_p_only
from accelerate import Accelerator, DistributedDataParallelKwargs

# 加载 run.py（有 __main__ 保护，只取函数复用）
import importlib.util
_spec = importlib.util.spec_from_file_location('run_mod', '/home/zjh/scDFM/src/script/run.py')
run_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=1)
    ap.add_argument('--train_steps', type=int, default=10)
    ap.add_argument('--eval_perturbations', type=int, default=1)
    ap.add_argument('--eval_n_cells', type=int, default=128)
    ap.add_argument('--eval_ode_steps', type=int, default=20)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    config = FlowConfig(
        model_type='origin', batch_size=48, ntoken=18084, d_model=128,
        lr=5e-5, steps=100, eta_min=1e-6, data_name='vcc',
        perturbation_function='crisper', noise_type='Gaussian',
        fusion_method='differential_perceiver', infer_top_gene=1000,
        n_top_genes=18080, mode='predict_y', result_path='./result/vcc',
        gamma=0.5, split_method='additive', use_mmd_loss=True, fold=0,
        use_negative_edge=True, topk=30, num_workers=0,
        eval_n_cells=args.eval_n_cells, eval_ode_steps=args.eval_ode_steps,
        eval_bf16=True,
    )
    run_mod.config = config  # train_step/test/generate_sample 用模块级 config

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    run_mod.device = device

    t0 = time.time()
    data_manager = Data('./data')
    data_manager.load_data('vcc')
    data_manager.process_data(n_top_genes=18080, split_method='additive',
                              fold=0, use_negative_edge=True, k=30)
    train_sampler, valid_sampler, _ = data_manager.load_flow_data(batch_size=48)
    print(f'[data] load+process {time.time()-t0:.1f}s', flush=True)

    mask_path = os.path.join(data_manager.data_path, 'vcc',
        f'mask_fold_{config.fold}topk_{config.topk}{config.split_method}_negative_edge.pt')
    vf = instantiate_model(config.model_type, ntoken=config.ntoken, d_model=config.d_model,
        d_perturbation=config.d_model, fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function, mask_path=mask_path)

    vocab = process_vocab(data_manager, config)
    gene_ids = torch.tensor(vocab.encode(list(data_manager.adata.var_names)),
                            dtype=torch.long, device=device)
    run_mod.gene_ids = gene_ids
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}

    # ========== 1) 训练速度 ==========
    train_dataset = PerturbationDataset(train_sampler, config.batch_size)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(vf.parameters(), lr=config.lr)
    vf, optimizer, dataloader = accelerator.prepare(vf, optimizer, dataloader)
    vf.train()
    criterion = nn.MSELoss()
    it = iter(dataloader)

    def one_train_step(batch):
        source = batch['src_cell_data'].squeeze(0).to(device)
        target = batch['tgt_cell_data'].squeeze(0).to(device)
        perturbation_id = batch['condition_id'].squeeze(0).to(device)
        cell_line_id = batch['cell_line_id'].squeeze(0).to(device)
        pname = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
        perturbation_id = torch.tensor(vocab.encode(pname), dtype=torch.long, device=device)
        perturbation_id = perturbation_id.repeat(source.shape[0], 1)
        set_requires_grad_for_p_only(vf, p_only=config.mode)
        loss = run_mod.train_step(source, target, perturbation_id, vf, criterion, accelerator,
                                  noise_type=config.noise_type, mode=config.mode,
                                  cell_line_id=cell_line_id)
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        optimizer.step()
        torch.cuda.synchronize()

    for _ in range(2):
        one_train_step(next(it))
    times = []
    for _ in range(args.train_steps):
        b = next(it)
        t1 = time.time()
        one_train_step(b)
        times.append(time.time() - t1)
    tps = float(np.mean(times))
    print(f'[train] {args.train_steps} 步平均 {tps:.2f}s/step（bsz=48, 1000基因, fp32, 独占GPU）', flush=True)
    for k in (10000, 50000, 200000):
        print(f'[train] {k} 步 ≈ {tps*k/3600:.1f}h（单卡）; 3卡DDP ≈ {tps*k/3600/3:.1f}h', flush=True)

    # ========== 2) eval 速度（严格配置 bf16）==========
    vf.eval()
    vf.to(torch.bfloat16)
    control_data = valid_sampler.get_control_data()
    if 'cell_line_id' in control_data:
        ref_cl = int(control_data['cell_line_id'][0])
        ref_mask = control_data['cell_line_id'] == ref_cl
        control_data = {k: control_data[k][ref_mask] for k in control_data}
    plist = valid_sampler._perturbation_covariates[:args.eval_perturbations]
    per_pert_times = []
    for pname in plist:
        pdata = valid_sampler.get_perturbation_data(pname)
        pid = pdata['condition_id'].to(device)
        cl = pdata.get('cell_line_id')
        source = control_data['src_cell_data']
        if cl is not None:
            source = source[control_data['cell_line_id'] == cl]
        source = source.to(device)
        if config.perturbation_function == 'crisper':
            pc = [inverse_dict[int(x)] for x in pid[0].cpu().numpy()]
            pid = torch.tensor(vocab.encode(pc), dtype=torch.long, device=device)
            pid = pid.repeat(source.shape[0], 1)
        idx = torch.randperm(source.shape[0])
        source = source[idx]
        N = min(config.eval_n_cells, source.shape[0])
        source = source[:N]
        cl_t = None if cl is None else torch.tensor([cl], dtype=torch.long, device=device).repeat(N)
        t1 = time.time()
        preds = []
        for i in range(0, N, config.batch_size):
            bp = pid[0].repeat(source[i:i+config.batch_size].shape[0], 1).to(device)
            pe = run_mod.generate_sample(
                run_mod.wrapped_vf, source[i:i+config.batch_size], bp, vf,
                gene_ids=gene_ids, gene_all=gene_ids, steps=config.eval_ode_steps,
                cell_line_id=None if cl_t is None else cl_t[i:i+config.batch_size])
            preds.append(pe)
        torch.cuda.synchronize()
        dt = time.time() - t1
        per_pert_times.append(dt)
        print(f'[eval] 扰动 {pname}: {dt:.1f}s（{N} cells, {config.eval_ode_steps} ode步 rk4, bf16）', flush=True)
    tpp = float(np.mean(per_pert_times))
    print(f'[eval] 平均单扰动 {tpp:.1f}s', flush=True)
    print(f'[eval] 50 扰动生成 ≈ {tpp*50/60:.1f} 分钟', flush=True)
    print(f'[eval] 完整 val 评测（含 cell-eval 指标 ~10min）≈ {tpp*50/60 + 10:.1f} 分钟', flush=True)


if __name__ == '__main__':
    main()
