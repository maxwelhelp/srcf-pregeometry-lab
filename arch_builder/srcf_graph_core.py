from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRCFGraphConfig:
    """Task-agnostic graph/relation SRCF core.

    Contract:
      frontend builds R [B,N,N,C]
      SRCFGraphCore closes relation state H with learned triangle-aware moves
      task head reads H* into logits/targets

    The core does not know task recipes. It only learns latent graph moves under
    task loss + generic closure pressure.
    """

    rel_dim: int
    out_dim: int = 3
    dim: int = 64
    hidden: int = 128
    action_count: int = 8
    action_emb_dim: int = 16
    layers: int = 2
    micro_steps: int = 3
    dropout: float = 0.0
    noise_std: float = 0.03
    action_temperature: float = 1.0
    use_triangle: bool = True
    use_reverse_triangle: bool = False
    use_rel_skip: bool = True


@dataclass
class SRCFGraphLossWeights:
    fixed: float = 0.05
    recovery: float = 0.05
    contract: float = 0.05
    far_keep: float = 0.02
    state_var: float = 0.02
    move_band: float = 0.02
    action_entropy: float = 0.0
    edge_sparsity: float = 0.0
    far_margin: float = 0.25
    state_var_floor: float = 0.005
    move_min: float = 0.005
    move_max: float = 2.0


def _pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros((0,))
    dist = torch.cdist(z.float(), z.float(), p=2)
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return dist[mask]


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    return (x * m).sum() / m.sum().clamp_min(1.0)


class SRCFGraphClosureLayer(nn.Module):
    """Triangle-aware learned closure transition.

    Core update:
      tri[i,j,c] = sum_k h[i,k,c] * h[k,j,c] / N

    This is the missing path-composition operation: relation consistency should
    explicitly know how a relation i->j is supported by all intermediate k.
    """

    def __init__(self, cfg: SRCFGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        h = cfg.hidden
        a = cfg.action_count
        e = cfg.action_emb_dim

        # h,row,col,global,row_std,col_std,tri,(optional reverse tri),rel_skip
        pieces = 6 + int(cfg.use_triangle) + int(cfg.use_reverse_triangle)
        feat_dim = d * pieces
        if cfg.use_rel_skip:
            feat_dim += cfg.rel_dim

        self.action_emb = nn.Parameter(torch.randn(a, e) * 0.02)
        self.context_norm = nn.LayerNorm(feat_dim)
        self.edge_head = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.SiLU(),
            nn.Linear(h, a),
        )
        self.move_net = nn.Sequential(
            nn.LayerNorm(feat_dim + e),
            nn.Linear(feat_dim + e, h),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, d),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.SiLU(),
            nn.Linear(h, d),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d)

    @staticmethod
    def triangle_update(h: torch.Tensor) -> torch.Tensor:
        """Path composition: tri[i,j,c] = sum_k h[i,k,c] * h[k,j,c] / N."""
        b, n, _, d = h.shape
        hc = h.permute(0, 3, 1, 2).contiguous().reshape(b * d, n, n)
        tri = torch.bmm(hc, hc) / max(float(n), 1.0)
        return tri.reshape(b, d, n, n).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def reverse_triangle_update(h: torch.Tensor) -> torch.Tensor:
        """Reverse support proxy: sum_k h[k,i,c] * h[j,k,c] / N."""
        b, n, _, d = h.shape
        hc = h.permute(0, 3, 1, 2).contiguous().reshape(b * d, n, n)
        tri = torch.bmm(hc.transpose(1, 2), hc.transpose(1, 2)) / max(float(n), 1.0)
        return tri.reshape(b, d, n, n).permute(0, 2, 3, 1).contiguous()

    def build_context(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        row = h.mean(dim=2, keepdim=True).expand_as(h)
        col = h.mean(dim=1, keepdim=True).expand_as(h)
        glob = h.mean(dim=(1, 2), keepdim=True).expand_as(h)
        row_std = h.std(dim=2, keepdim=True, unbiased=False).expand_as(h)
        col_std = h.std(dim=1, keepdim=True, unbiased=False).expand_as(h)
        pieces = [h, row, col, glob, row_std, col_std]
        if self.cfg.use_triangle:
            pieces.append(self.triangle_update(h))
        if self.cfg.use_reverse_triangle:
            pieces.append(self.reverse_triangle_update(h))
        if self.cfg.use_rel_skip:
            pieces.append(rel)
        return torch.cat(pieces, dim=-1)

    def step_once(self, h: torch.Tensor, rel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ctx = self.context_norm(self.build_context(h, rel))
        edge = torch.sigmoid(self.edge_head(ctx))
        logits = self.action_head(ctx) / max(float(self.cfg.action_temperature), 1e-4)
        action_prob = torch.softmax(logits, dim=-1)

        b, n, _, fd = ctx.shape
        ctx_a = ctx.unsqueeze(3).expand(b, n, n, self.cfg.action_count, fd)
        emb = self.action_emb.view(1, 1, 1, self.cfg.action_count, self.cfg.action_emb_dim).expand(
            b, n, n, self.cfg.action_count, self.cfg.action_emb_dim
        )
        moves = self.move_net(torch.cat([ctx_a, emb], dim=-1))
        move = (action_prob.unsqueeze(-1) * moves).sum(dim=3)
        gate = self.update_gate(ctx)
        h_next = self.norm(h + edge * gate * move)

        entropy = (-(action_prob.clamp_min(1e-8) * action_prob.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        metrics = {
            "edge_mass": edge.mean(),
            "action_entropy": entropy,
            "update_gate_mean": gate.mean(),
        }
        return h_next, metrics

    def forward(self, h: torch.Tensor, rel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        start = h
        curves = []
        edge_masses = []
        entropies = []
        gates = []
        prev = h
        for _ in range(max(1, int(self.cfg.micro_steps))):
            h, m = self.step_once(h, rel)
            curves.append((h - prev).pow(2).mean().sqrt())
            edge_masses.append(m["edge_mass"])
            entropies.append(m["action_entropy"])
            gates.append(m["update_gate_mean"])
            prev = h

        h_fixed, _ = self.step_once(h, rel)
        if self.training and self.cfg.noise_std > 0:
            h_noisy = h + torch.randn_like(h) * float(self.cfg.noise_std)
        else:
            h_noisy = h
        h_recovered, _ = self.step_once(h_noisy, rel)

        diag = {
            "fixed": F.mse_loss(h_fixed, h),
            "recovery": F.mse_loss(h_recovered, h.detach()),
            "move": (h - start).pow(2).mean(dim=-1).sqrt().mean(),
            "curve_start": curves[0].detach() if curves else h.new_zeros(()),
            "curve_end": curves[-1].detach() if curves else h.new_zeros(()),
            "edge_mass": torch.stack(edge_masses).mean() if edge_masses else h.new_zeros(()),
            "action_entropy": torch.stack(entropies).mean() if entropies else h.new_zeros(()),
            "update_gate_mean": torch.stack(gates).mean() if gates else h.new_zeros(()),
        }
        return h, diag


class SRCFGraphCore(nn.Module):
    """Universal graph closure core.

    Input is relation tensor R [B,N,N,C]. The output head is a simple pair head by
    default, but users can replace it. The core itself is task-agnostic.
    """

    def __init__(self, cfg: SRCFGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.enc = nn.Sequential(
            nn.Linear(cfg.rel_dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.layers = nn.ModuleList([SRCFGraphClosureLayer(cfg) for _ in range(cfg.layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.out_dim),
        )

    def descriptor(self, h: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h [B,N,N,D], active_mask [B,N,N] optional
        if active_mask is None:
            return h.mean(dim=(1, 2))
        m = active_mask.to(dtype=h.dtype).unsqueeze(-1)
        return (h * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)

    def forward(
        self,
        rel: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        return_h: bool = False,
    ) -> Dict[str, object]:
        if rel.ndim != 4:
            raise ValueError(f"rel must be [B,N,N,C], got {tuple(rel.shape)}")
        h0 = self.enc(rel)
        h = h0
        layer_diags = []
        for layer in self.layers:
            h, diag = layer(h, rel)
            layer_diags.append(diag)
        logits = self.head(h)
        desc0 = self.descriptor(h0, active_mask)
        desc = self.descriptor(h, active_mask)

        diagnostics: Dict[str, torch.Tensor] = {
            "descriptor": desc,
            "descriptor_start": desc0,
            "state_var": desc.var(dim=0, unbiased=False).mean() if desc.shape[0] > 1 else desc.var(unbiased=False),
            "move": (desc - desc0).pow(2).mean(dim=-1).sqrt().mean(),
        }
        if layer_diags:
            for key in layer_diags[0].keys():
                diagnostics[key] = torch.stack([d[key] for d in layer_diags]).mean()
            diagnostics["curve_ratio"] = diagnostics["curve_end"] / diagnostics["curve_start"].clamp_min(1e-8)

        out: Dict[str, object] = {
            "logits": logits,
            "diagnostics": diagnostics,
            "descriptor": desc,
            "layer_diagnostics": layer_diags,
        }
        if return_h:
            out["h0"] = h0
            out["h"] = h
        return out


def srcf_graph_closure_loss(
    output: Dict[str, object],
    weights: SRCFGraphLossWeights = SRCFGraphLossWeights(),
    peer_output: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    diagnostics = output["diagnostics"]
    descriptor = output["descriptor"]
    loss = descriptor.new_zeros(())

    fixed = diagnostics.get("fixed", descriptor.new_zeros(()))
    recovery = diagnostics.get("recovery", descriptor.new_zeros(()))
    move = diagnostics.get("move", descriptor.new_zeros(()))
    state_var = diagnostics.get("state_var", descriptor.new_zeros(()))
    action_entropy = diagnostics.get("action_entropy", descriptor.new_zeros(()))
    edge_mass = diagnostics.get("edge_mass", descriptor.new_zeros(()))

    loss = loss + weights.fixed * fixed
    loss = loss + weights.recovery * recovery
    loss = loss + weights.state_var * F.relu(descriptor.new_tensor(weights.state_var_floor) - state_var)
    loss = loss + weights.move_band * (
        F.relu(descriptor.new_tensor(weights.move_min) - move).pow(2)
        + F.relu(move - descriptor.new_tensor(weights.move_max)).pow(2)
    )

    far_dist = _pairwise_distances(F.normalize(descriptor.float(), dim=-1, eps=1e-6))
    if far_dist.numel() > 0:
        far_keep_loss = F.relu(descriptor.new_tensor(weights.far_margin) - far_dist.to(descriptor.device)).mean()
        loss = loss + weights.far_keep * far_keep_loss
    else:
        far_keep_loss = descriptor.new_zeros(())

    if peer_output is not None:
        contract = F.mse_loss(descriptor, peer_output["descriptor"])
        loss = loss + weights.contract * contract
    else:
        contract = descriptor.new_zeros(())

    if weights.action_entropy != 0.0:
        loss = loss - weights.action_entropy * action_entropy
    if weights.edge_sparsity != 0.0:
        loss = loss + weights.edge_sparsity * edge_mass

    metrics = {
        "srcf_graph_loss": float(loss.detach().cpu()),
        "srcf_graph_fixed": float(fixed.detach().cpu()),
        "srcf_graph_recovery": float(recovery.detach().cpu()),
        "srcf_graph_contract": float(contract.detach().cpu()),
        "srcf_graph_far_keep_loss": float(far_keep_loss.detach().cpu()),
        "srcf_graph_move": float(move.detach().cpu()),
        "srcf_graph_state_var": float(state_var.detach().cpu()),
        "srcf_graph_action_entropy": float(action_entropy.detach().cpu()),
        "srcf_graph_edge_mass": float(edge_mass.detach().cpu()),
        "srcf_graph_curve_ratio": float(diagnostics.get("curve_ratio", descriptor.new_zeros(())).detach().cpu()),
    }
    return loss, metrics
