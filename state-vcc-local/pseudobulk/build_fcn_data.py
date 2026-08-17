"""构建 FCN pseudo-bulk 训练数据。

对每个细胞系的每个扰动，构建 (ctrl_mean, ESM-2 embedding) → delta 样本。

输入：
  - h5 文件（state/vci_pretrain/*.h5，log1p 表达）
  - ESM-2 蛋白嵌入（state/competition_support_set/ESM2_pert_features.pt）
  - delta 库（pseudobulk/deltas_*.npz）

输出：
  - pseudobulk/fcn_traindata.npz（inputs, targets, perts, sources, ctrl_means_dict）
  - pseudobulk/ctrl_mean_*.npy（每个细胞系的对照均值）

用法：
  cd /home/zjh/state-vcc-local/state && uv run --no-sync python ../pseudobulk/build_fcn_data.py
"""
import os
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

BASE = "/home/zjh/state-vcc-local/state/vci_pretrain"
ESM_FILE = "/home/zjh/state-vcc-local/state/competition_support_set/ESM2_pert_features.pt"
DELTA_DIR = "/home/zjh/state-vcc-local/pseudobulk"
OUT = "/home/zjh/state-vcc-local/pseudobulk"
CTRL = "non-targeting"

H5_FILES = {
    "H1": "competition_train.h5",
    "k562_gwps": "k562_gwps.h5",
    "rpe1": "rpe1.h5",
    "jurkat": "jurkat.h5",
    "k562": "k562.h5",
    "hepg2": "hepg2.h5",
}


def compute_ctrl_means():
    """从 h5 文件计算每个细胞系的 non-targeting 对照均值。"""
    means = {}
    for name, fname in H5_FILES.items():
        path = os.path.join(BASE, fname)
        a = ad.read_h5ad(path)
        tg = a.obs["target_gene"].astype(str).to_numpy()
        mask = tg == CTRL
        X = a.X[mask]
        if sp.issparse(X):
            X = np.asarray(X.mean(axis=0)).ravel()
        else:
            X = np.asarray(X.mean(axis=0)).ravel()
        means[name] = X.astype(np.float32)
        n_ctrl = int(mask.sum())
        print(f"  {name}: {n_ctrl} ctrl cells, ctrl_mean norm={np.linalg.norm(means[name]):.2f}")
        np.save(os.path.join(OUT, f"ctrl_mean_{name}.npy"), means[name])
    return means


def load_esm2():
    """加载 ESM-2 蛋白嵌入 dict。"""
    esm = torch.load(ESM_FILE, map_location="cpu", weights_only=True)
    return {k: v.numpy().astype(np.float32) for k, v in esm.items()}


def build_training_data(ctrl_means, esm2):
    """从 delta 库构建 (ctrl_mean + ESM-2) → delta 训练样本。"""
    esm_dim = 5120
    zero_esm = np.zeros(esm_dim, dtype=np.float32)

    all_inputs = []
    all_targets = []
    all_perts = []
    all_sources = []
    n_missing_esm = 0

    for name in H5_FILES:
        d = np.load(os.path.join(DELTA_DIR, f"deltas_{name}.npz"), allow_pickle=False)
        ctrl = ctrl_means[name]
        for i, p in enumerate(d["perts"]):
            p_str = str(p)
            esm = esm2.get(p_str, None)
            if esm is None:
                esm = zero_esm
                n_missing_esm += 1
            inp = np.concatenate([ctrl, esm]).astype(np.float32)
            all_inputs.append(inp)
            all_targets.append(d["deltas"][i])
            all_perts.append(p_str)
            all_sources.append(name)

    inputs = np.stack(all_inputs)
    targets = np.stack(all_targets)
    print(f"  Training samples: {inputs.shape[0]}")
    print(f"  Input dim: {inputs.shape[1]} (ctrl {ctrl.shape[0]} + esm {esm_dim})")
    print(f"  Target dim: {targets.shape[1]}")
    print(f"  Missing ESM-2: {n_missing_esm}/{len(all_perts)}")
    return inputs, targets, all_perts, all_sources


if __name__ == "__main__":
    t0 = time.time()
    print("=== 计算 ctrl_mean ===")
    ctrl_means = compute_ctrl_means()

    print("\n=== 加载 ESM-2 ===")
    esm2 = load_esm2()
    print(f"  ESM-2 genes: {len(esm2)}")

    print("\n=== 构建训练数据 ===")
    inputs, targets, perts, sources = build_training_data(ctrl_means, esm2)

    # 保存
    np.savez_compressed(
        os.path.join(OUT, "fcn_traindata.npz"),
        inputs=inputs,
        targets=targets,
        perts=np.array(perts),
        sources=np.array(sources),
    )
    print(f"\n保存: fcn_traindata.npz ({time.time()-t0:.0f}s)")
    print(f"  Unique perturbations: {len(set(perts))}")
    print(f"  Sources: { {s: perts.count(s) for s in set(sources)} }")
