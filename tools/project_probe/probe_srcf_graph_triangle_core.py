from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import networkx as nx
except Exception as exc:
    raise SystemExit("networkx is required") from exc

try:
    from sklearn.metrics import roc_auc_score
except Exception as exc:
    raise SystemExit("scikit-learn is required") from exc

from arch_builder.srcf_graph_core import (
    SRCFGraphConfig,
    SRCFGraphCore,
    SRCFGraphLossWeights,
    srcf_graph_closure_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def upper_mask(n: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def offdiag_active_mask(active: torch.Tensor) -> torch.Tensor:
    b, n = active.shape
    m = active[:, :, None] & active[:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=active.device)[None]
    return m & (~eye)


def safe_auc(target: torch.Tensor, score: torch.Tensor, mask: torch.Tensor) -> float:
    y = target[mask].detach().cpu().numpy().astype(np.int32)
    s = score[mask].detach().cpu().numpy().astype(np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def simple_graph(name: str) -> nx.Graph:
    if name == "karate":
        g = nx.karate_club_graph()
    elif name == "lesmis":
        g = nx.les_miserables_graph()
    elif name == "florentine":
        g = nx.florentine_families_graph()
    elif name == "davis":
        g = nx.davis_southern_women_graph()
    else:
        raise ValueError(f"unknown graph {name}")
    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(g.edges())
    h.remove_edges_from(nx.selfloop_edges(h))
    comps = sorted(nx.connected_components(h), key=len, reverse=True)
    h = h.subgraph(comps[0]).copy()
    return nx.convert_node_labels_to_integers(h)


def graph_to_adj(g: nx.Graph, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if g.number_of_nodes() > n:
        start = int(rng.integers(0, g.number_of_nodes()))
        seen = [start]
        seen_set = {start}
        q = [start]
        while q and len(seen) < n:
            u = q.pop(0)
            nbrs = list(g.neighbors(u))
            rng.shuffle(nbrs)
            for v in nbrs:
                if v not in seen_set:
                    seen.append(v)
                    seen_set.add(v)
                    q.append(v)
                    if len(seen) >= n:
                        break
        if len(seen) < n:
            rest = [x for x in g.nodes() if x not in seen_set]
            rng.shuffle(rest)
            seen.extend(rest[: n - len(seen)])
        h = g.subgraph(seen[:n]).copy()
    else:
        h = g.copy()
    h = nx.convert_node_labels_to_integers(h)
    a = nx.to_numpy_array(h, dtype=np.float32)
    active = np.zeros(n, dtype=np.bool_)
    active[: a.shape[0]] = True
    out = np.zeros((n, n), dtype=np.float32)
    out[: a.shape[0], : a.shape[1]] = (a > 0).astype(np.float32)
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out, active


def communities_matrix(a: np.ndarray, active: np.ndarray) -> np.ndarray:
    g = nx.from_numpy_array((a > 0).astype(np.float32))
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(g))
    except Exception:
        comms = list(nx.connected_components(g))
    labels = np.full(a.shape[0], -1, dtype=np.int64)
    for ci, nodes in enumerate(comms):
        for node in nodes:
            if active[node]:
                labels[node] = ci
    same = (labels[:, None] == labels[None, :]).astype(np.float32)
    invalid = (labels[:, None] < 0) | (labels[None, :] < 0)
    same[invalid] = 0.0
    np.fill_diagonal(same, 0.0)
    return same


def targets_from_clean(a: np.ndarray, active: np.ndarray) -> np.ndarray:
    edge = (a > 0).astype(np.float32)
    path2 = ((edge @ edge) > 0).astype(np.float32)
    path2 = np.maximum(path2, edge)
    np.fill_diagonal(path2, 0.0)
    comm = communities_matrix(edge, active)
    return np.stack([edge, path2, comm], axis=-1).astype(np.float32)


def corrupt_adj(a: np.ndarray, active: np.ndarray, drop_p: float, add_p: float, rng: np.random.Generator) -> np.ndarray:
    n = a.shape[0]
    c = a.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if not (active[i] and active[j]):
                continue
            if a[i, j] > 0.5 and rng.random() < drop_p:
                c[i, j] = c[j, i] = 0.0
            elif a[i, j] < 0.5 and rng.random() < add_p:
                c[i, j] = c[j, i] = 1.0
    np.fill_diagonal(c, 0.0)
    return c


def relation_features(c: np.ndarray, active: np.ndarray, rel_mode: str) -> np.ndarray:
    c = (c > 0).astype(np.float32)
    n = c.shape[0]
    deg = c.sum(-1) / max(n - 1, 1)
    cn = (c @ c) / max(n, 1)
    two = (cn > 0).astype(np.float32)
    union = deg[:, None] + deg[None, :] - cn
    jac = cn / (union + 1e-6)
    deg_diff = np.abs(deg[:, None] - deg[None, :])
    deg_prod = deg[:, None] * deg[None, :]
    row = c.mean(-1)[:, None].repeat(n, axis=1)
    miss = (1.0 - c) * cn
    if rel_mode == "raw":
        feats = [c]
    else:
        feats = [c, cn, jac, two, deg_diff, deg_prod, row, miss]
    r = np.stack(feats, axis=-1).astype(np.float32)
    r[~(active[:, None] & active[None, :])] = 0.0
    return r


@dataclass
class Batch:
    rel: torch.Tensor
    target: torch.Tensor
    active: torch.Tensor
    corrupt: torch.Tensor


def make_batch(graphs, args, rng: np.random.Generator, device: torch.device, hard: bool = False) -> Batch:
    rels, targets, actives, corrupts = [], [], [], []
    drop = args.hard_drop_p if hard else args.drop_p
    add = args.hard_add_p if hard else args.add_p
    for _ in range(args.batch):
        g = graphs[int(rng.integers(0, len(graphs)))]
        clean, active = graph_to_adj(g, args.n, rng)
        corrupt = corrupt_adj(clean, active, drop, add, rng)
        rels.append(relation_features(corrupt, active, args.rel_mode))
        targets.append(targets_from_clean(clean, active))
        actives.append(active.astype(np.float32))
        corrupts.append(corrupt)
    return Batch(
        torch.tensor(np.stack(rels), device=device),
        torch.tensor(np.stack(targets), device=device),
        torch.tensor(np.stack(actives), dtype=torch.bool, device=device),
        torch.tensor(np.stack(corrupts), device=device),
    )


def masked_bce(logits: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    mask = offdiag_active_mask(active)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss[mask].mean()


@torch.no_grad()
def evaluate(model: SRCFGraphCore, graphs, args, rng, device, hard: bool = False) -> Dict[str, float]:
    model.eval()
    vals = {"edge": [], "path2": [], "comm": [], "raw_edge": []}
    for _ in range(args.eval_batches):
        b = make_batch(graphs, args, rng, device, hard=hard)
        mask = offdiag_active_mask(b.active)
        out = model(b.rel, active_mask=mask)
        score = torch.sigmoid(out["logits"])
        vals["edge"].append(safe_auc(b.target[..., 0], score[..., 0], mask))
        vals["path2"].append(safe_auc(b.target[..., 1], score[..., 1], mask))
        vals["comm"].append(safe_auc(b.target[..., 2], score[..., 2], mask))
        vals["raw_edge"].append(safe_auc(b.target[..., 0], b.corrupt, mask))
    model.train()
    return {k: float(np.nanmean(v)) for k, v in vals.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--graphs", default="karate,lesmis,florentine,davis")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--rel-mode", default="raw", choices=["raw", "full"])
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--micro-steps", type=int, default=3)
    p.add_argument("--actions", type=int, default=8)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--eval-batches", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--drop-p", type=float, default=0.25)
    p.add_argument("--add-p", type=float, default=0.06)
    p.add_argument("--hard-drop-p", type=float, default=0.45)
    p.add_argument("--hard-add-p", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--closure-w", type=float, default=0.10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    names = [x.strip() for x in args.graphs.split(",") if x.strip()]
    graphs = [simple_graph(n) for n in names]
    rng = np.random.default_rng(args.seed)
    rel_dim = 1 if args.rel_mode == "raw" else 8

    cfg = SRCFGraphConfig(
        rel_dim=rel_dim,
        out_dim=3,
        dim=args.dim,
        hidden=args.hidden,
        layers=args.layers,
        micro_steps=args.micro_steps,
        action_count=args.actions,
        use_triangle=True,
        use_rel_skip=True,
        noise_std=0.02,
    )
    model = SRCFGraphCore(cfg).to(device)
    weights = SRCFGraphLossWeights(
        fixed=0.04,
        recovery=0.04,
        contract=0.05,
        far_keep=0.02,
        state_var=0.02,
        move_band=0.02,
        far_margin=0.20,
        state_var_floor=0.003,
        move_min=0.003,
        move_max=2.5,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("SRCF_GRAPH_TRIANGLE_CORE_PROBE")
    print(f"device={device} graphs={names} rel_mode={args.rel_mode} params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    last_metrics = {}
    for step in range(1, args.steps + 1):
        b = make_batch(graphs, args, rng, device, hard=False)
        # peer = same clean graph distribution under independent corruption; no labels leak into closure loss
        peer = make_batch(graphs, args, rng, device, hard=False)
        mask = offdiag_active_mask(b.active)
        out = model(b.rel, active_mask=mask)
        out_peer = model(peer.rel, active_mask=offdiag_active_mask(peer.active))
        task = masked_bce(out["logits"], b.target, b.active)
        closs, cm = srcf_graph_closure_loss(out, weights, peer_output=out_peer)
        loss = task + args.closure_w * closs
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm}")
        opt.step()
        last_metrics = cm
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ein = evaluate(model, graphs, args, rng, device, hard=False)
            eood = evaluate(model, graphs, args, rng, device, hard=True)
            print(
                f"step={step:04d} loss={float(loss.detach().cpu()):.4f} task={float(task.detach().cpu()):.4f} "
                f"srcf={cm['srcf_graph_loss']:.4f} fixed={cm['srcf_graph_fixed']:.4f} rec={cm['srcf_graph_recovery']:.4f} "
                f"move={cm['srcf_graph_move']:.4f} var={cm['srcf_graph_state_var']:.4f} curve_ratio={cm['srcf_graph_curve_ratio']:.4f}"
            )
            print(f"  IN : edge={ein['edge']:.3f} path2={ein['path2']:.3f} comm={ein['comm']:.3f} raw_edge={ein['raw_edge']:.3f}")
            print(f"  OOD: edge={eood['edge']:.3f} path2={eood['path2']:.3f} comm={eood['comm']:.3f} raw_edge={eood['raw_edge']:.3f}")

    for k, v in last_metrics.items():
        if not math.isfinite(float(v)):
            raise RuntimeError(f"bad metric {k}={v}")
    print("SRCF_GRAPH_TRIANGLE_CORE_PASS")


if __name__ == "__main__":
    main()
