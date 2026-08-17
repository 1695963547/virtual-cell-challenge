"""FCN pseudo-bulk 模型训练。

参照 XLearning Lab (VCC 2025 #2) 方案：
  - 输入：ctrl_mean (18080) + ESM-2 (5120) = 23200
  - 输出：delta (18080)，残差学习
  - Loss：MSE + direction_loss（cosine similarity）
  - 优化器：AdamW + CosineAnnealing
  - 正则化：dropout + weight_decay

用法：
  cd /home/zjh/state-vcc-local/state && uv run --no-sync python ../pseudobulk/train_fcn.py \
      [--epochs 500] [--lr 1e-3] [--hidden 2048] [--wd 0.05] [--batch_size 64]
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_FILE = "/home/zjh/state-vcc-local/pseudobulk/fcn_traindata.npz"
CKPT_DIR = "/home/zjh/state-vcc-local/pseudobulk"

GENE_DIM = 18080
ESM_DIM = 5120


class PerturbFCN(nn.Module):
    """Pseudo-bulk 扰动预测 FCN（XLearning 式）。

    输入：ctrl_mean (gene_dim) + ESM-2 (esm_dim)
    输出：delta (gene_dim)
    """

    def __init__(self, gene_dim=GENE_DIM, esm_dim=ESM_DIM, hidden=2048,
                 n_layers=3, dropout=0.15):
        super().__init__()
        self.gene_dim = gene_dim
        self.esm_dim = esm_dim
        input_dim = gene_dim + esm_dim

        layers = []
        in_d = input_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden),
            ])
            in_d = hidden
        layers.append(nn.Linear(in_d, gene_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ctrl_mean, esm_emb):
        x = torch.cat([ctrl_mean, esm_emb], dim=-1)
        return self.net(x)


def direction_loss(pred, target):
    """1 - cosine similarity，衡量方向一致性。"""
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()


def train(args):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    data = np.load(DATA_FILE, allow_pickle=False)
    inputs = torch.from_numpy(data["inputs"]).float()
    targets = torch.from_numpy(data["targets"]).float()
    perts = data["perts"]
    sources = data["sources"]
    n_samples = inputs.shape[0]
    print(f"Training data: {n_samples} samples, input_dim={inputs.shape[1]}, target_dim={targets.shape[1]}")

    # 拆分 ctrl_mean 和 ESM-2
    ctrl_all = inputs[:, :GENE_DIM]
    esm_all = inputs[:, GENE_DIM:]

    # 训练/验证划分：留出 20% 作为验证（按细胞系分层）
    rng = np.random.RandomState(42)
    n_val = max(1, n_samples // 5)
    # 分层采样：确保每个细胞系都有验证样本
    val_idx = []
    train_idx = []
    for src in set(sources):
        idx = np.where(sources == src)[0]
        rng.shuffle(idx)
        n_v = max(1, len(idx) // 5)
        val_idx.extend(idx[:n_v].tolist())
        train_idx.extend(idx[n_v:].tolist())
    val_idx = np.array(val_idx)
    train_idx = np.array(train_idx)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 模型
    model = PerturbFCN(
        gene_dim=GENE_DIM,
        esm_dim=ESM_DIM,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M (hidden={args.hidden}, layers={args.n_layers})")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_val_loss = float("inf")
    best_epoch = 0
    batch_size = min(args.batch_size, len(train_idx))

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_idx)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_dir = 0.0
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i + batch_size]
            ctrl = ctrl_all[batch_idx].to(device)
            esm = esm_all[batch_idx].to(device)
            tgt = targets[batch_idx].to(device)

            pred = model(ctrl, esm)
            mse = F.mse_loss(pred, tgt)
            dir_l = direction_loss(pred, tgt)
            loss = mse + args.dir_weight * dir_l

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_dir += dir_l.item()
            n_batches += 1

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            ctrl_v = ctrl_all[val_idx].to(device)
            esm_v = esm_all[val_idx].to(device)
            tgt_v = targets[val_idx].to(device)
            pred_v = model(ctrl_v, esm_v)
            val_mse = F.mse_loss(pred_v, tgt_v).item()
            val_dir = direction_loss(pred_v, tgt_v).item()
            val_loss = val_mse + args.dir_weight * val_dir

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs}: "
                  f"train_loss={epoch_loss/n_batches:.4f} mse={epoch_mse/n_batches:.4f} dir={epoch_dir/n_batches:.4f} | "
                  f"val_loss={val_loss:.4f} val_mse={val_mse:.4f} val_dir={val_dir:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_path = os.path.join(CKPT_DIR, "fcn_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_args": {
                    "gene_dim": GENE_DIM,
                    "esm_dim": ESM_DIM,
                    "hidden": args.hidden,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                },
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_mse": val_mse,
                "val_dir": val_dir,
            }, ckpt_path)

    print(f"\n训练完成 ({time.time()-t0:.0f}s)")
    print(f"Best epoch: {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(CKPT_DIR, 'fcn_best.pt')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dir_weight", type=float, default=0.1,
                    help="direction_loss 权重")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
