#!/usr/bin/env python
"""诊断 vocab/索引越界问题"""
import sys
sys.path.insert(0, "/home/zjh/scDFM")
import torch
from src.data_process.data import Data
from config.config_flow import FlowConfig
from src.utils.utils import process_vocab

dm = Data('/home/zjh/scDFM/data')
dm.load_data('vcc')
dm.process_data(n_top_genes=18080, split_method='additive', fold=0, use_negative_edge=True, k=30)
print('adata_train:', dm.adata_train.shape, 'var_names[:5]:', list(dm.adata_train.var_names[:5]))

cfg = FlowConfig(data_name='vcc', n_top_genes=18080, ntoken=18084)
vocab = process_vocab(dm, cfg)
print('vocab size:', len(vocab))
print('vocab[:8]:', [vocab.itos[i] for i in range(8)])
print('vocab[-3:]:', [vocab.itos[i] for i in range(len(vocab)-3, len(vocab))])

gene_ids = vocab.encode(list(dm.adata.var_names))
print('gene_ids: len=%d min=%d max=%d' % (len(gene_ids), min(gene_ids), max(gene_ids)))

# perturbation 名 encode 检查
inverse_dict = {v: str(k) for k, v in dm.perturbation_dict.items()}
bad = []
for pid, name in inverse_dict.items():
    if name == 'control':
        continue
    eid = vocab.encode([name])[0]
    if not (0 <= eid < 18084):
        bad.append((name, eid))
print('perturbation encode bad:', bad[:10])

# TrainSampler 构造 + PerturbationDataset 采样检查（cell line 逻辑）
train_sampler, valid_sampler, _ = dm.load_flow_data(batch_size=48)
from src.data_process.data import PerturbationDataset
ds = PerturbationDataset(train_sampler, 48)
b = ds[0]
print('sample keys:', list(b.keys()))
print('cell_line_id unique:', torch.unique(b['cell_line_id']).tolist() if 'cell_line_id' in b else 'N/A')
print('condition_id shape:', b['condition_id'].shape, 'value:', b['condition_id'][0].tolist())

# valid sampler 检查
ctrl = valid_sampler.get_control_data()
print('val control cells:', ctrl['src_cell_data'].shape, 'cell_line_id unique:', torch.unique(ctrl['cell_line_id']).tolist() if 'cell_line_id' in ctrl else 'N/A')
p0 = valid_sampler._perturbation_covariates[0]
pd0 = valid_sampler.get_perturbation_data(p0)
print('val pert[0]:', p0, 'shape:', pd0['tgt_cell_data'].shape, 'cell_line_id:', pd0.get('cell_line_id'))
