"""Fused PDS/DES proxy loss for gene-space perturbation prediction.

Designed for models that bypass the SE latent bottleneck and predict directly
in gene space (e.g., PseudobulkPerturbationModel with embed_key=None).

Three differentiable components approximate the VCC competition metrics:
  1. weighted_mse  → DES proxy: emphasizes top DE genes (large |delta|)
  2. contrastive   → PDS proxy: each pred delta closest to its own true delta
  3. direction     → cosine similarity on delta direction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedPDSDESLoss(nn.Module):
    """
    Fused loss approximating VCC competition metrics in gene space.

    Args:
        w_mse:    weight for weighted MSE (DES proxy)
        w_pds:    weight for contrastive loss (PDS proxy)
        w_dir:    weight for direction cosine loss
        alpha:    amplification factor for gene importance weighting
                  weight_g = 1 + alpha * |delta_true| / mean(|delta_true|)
        tau:      temperature for contrastive softmax (higher = softer)
        eps:      numerical stability floor for cosine similarity
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_pds: float = 0.1,
        w_dir: float = 0.1,
        alpha: float = 5.0,
        tau: float = 10.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_pds = w_pds
        self.w_dir = w_dir
        self.alpha = alpha
        self.tau = tau
        self.eps = eps

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
    ):
        """
        Args:
            delta_pred: predicted delta, [B, G]  (B = batch perturbations, G = genes)
            delta_true: true delta,      [B, G]

        Returns:
            (total_loss, component_dict)
        """
        B, G = delta_pred.shape

        # ---- Component 1: Weighted MSE (DES proxy) ----
        # Emphasize genes with large |delta| → these are the DE genes DES cares about
        abs_true = delta_true.abs().detach()
        gene_weight = 1.0 + self.alpha * abs_true / (abs_true.mean(dim=-1, keepdim=True) + self.eps)
        # gene_weight shape: [B, G], mean ≈ 1 + alpha for average genes, higher for DE genes

        weighted_mse = (gene_weight * (delta_pred - delta_true) ** 2).mean()

        # ---- Component 2: Contrastive loss (PDS proxy) ----
        # L1 distance matrix between all (pred_i, true_j) pairs
        # PDS wants: for each pred_i, the closest true delta should be true_i
        if B > 1:
            # dist_matrix[i][j] = L1_distance(delta_pred_i, delta_true_j)
            dist_matrix = torch.cdist(
                delta_pred.unsqueeze(0), delta_true.unsqueeze(0), p=1
            ).squeeze(0)  # [B, B]
            logits = -dist_matrix / self.tau
            labels = torch.arange(B, device=delta_pred.device)
            pds_loss = F.cross_entropy(logits, labels)
        else:
            pds_loss = torch.tensor(0.0, device=delta_pred.device)

        # ---- Component 3: Direction loss ----
        # Cosine similarity ensures delta direction is correct
        # (plain MSE is insensitive to direction errors on small-delta genes)
        pred_norm = F.normalize(delta_pred, dim=-1, eps=self.eps)
        true_norm = F.normalize(delta_true, dim=-1, eps=self.eps)
        direction_loss = (1.0 - (pred_norm * true_norm).sum(dim=-1)).mean()

        # ---- Total ----
        total = (
            self.w_mse * weighted_mse
            + self.w_pds * pds_loss
            + self.w_dir * direction_loss
        )

        components = {
            "weighted_mse": weighted_mse.detach(),
            "pds_loss": pds_loss.detach(),
            "direction_loss": direction_loss.detach(),
        }

        return total, components
