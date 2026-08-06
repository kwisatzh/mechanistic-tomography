#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Belief-state tomography minimal harness.

Synthetic wind tunnel:
  - Two independent binary HMMs are emitted as a 4-token stream.
  - HMM #1 is the controlled/belief variable z1.
  - HMM #2 is a nuisance variable z2 used to measure collateral movement.
  - True Bayesian posteriors are analytic via the forward filter.

Experiment:
  1. Train a tiny causal transformer to predict next token.
  2. Probe each residual layer for z1 and z2.
  3. Choose an actuation direction for z1.
  4. Hold a P controller fixed and sweep observer quality.
  5. Plot observer RMSE vs closed-loop control loss.

This is intentionally small: it should run on an Apple M-series laptop for quick
iteration and on Colab H100 for larger sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def logit(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def sigmoid_from_logit(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def bern_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log1p(-p) - torch.log1p(-q))


# -----------------------------
# Synthetic HMM wind tunnel
# -----------------------------


@dataclass
class HMMParams:
    p_stay1: float = 0.96
    p_emit1: float = 0.85
    p_stay2: float = 0.92
    p_emit2: float = 0.80


def _generate_one_hmm(
    batch_size: int,
    seq_len: int,
    p_stay: float,
    p_emit: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate hidden states, observed bits, and analytic posterior log-odds.

    hidden[t] in {0,1}; obs[t] is a noisy emission of hidden[t].
    posterior z[t] = logit P(hidden[t]=1 | obs[0:t]).
    """
    hidden = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    obs = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)

    hidden[:, 0] = torch.bernoulli(torch.full((batch_size,), 0.5, device=device)).long()
    for t in range(1, seq_len):
        stay = torch.bernoulli(torch.full((batch_size,), p_stay, device=device)).long()
        hidden[:, t] = torch.where(stay.bool(), hidden[:, t - 1], 1 - hidden[:, t - 1])

    # emission: P(obs=hidden)=p_emit
    correct = torch.bernoulli(torch.full((batch_size, seq_len), p_emit, device=device)).long()
    obs = torch.where(correct.bool(), hidden, 1 - hidden)

    posterior_p = torch.empty((batch_size, seq_len), dtype=torch.float32, device=device)
    prior = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)

    for t in range(seq_len):
        o = obs[:, t].float()
        # likelihoods for hidden state 1 and 0
        like1 = torch.where(o > 0.5, torch.tensor(p_emit, device=device), torch.tensor(1 - p_emit, device=device))
        like0 = torch.where(o > 0.5, torch.tensor(1 - p_emit, device=device), torch.tensor(p_emit, device=device))
        numer = prior * like1
        denom = numer + (1 - prior) * like0
        post = numer / denom.clamp_min(1e-8)
        posterior_p[:, t] = post
        # predict next hidden state prior
        prior = p_stay * post + (1 - p_stay) * (1 - post)

    return hidden, obs, logit(posterior_p)


def generate_batch(
    batch_size: int,
    seq_len: int,
    params: HMMParams,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    h1, o1, z1 = _generate_one_hmm(batch_size, seq_len, params.p_stay1, params.p_emit1, device)
    h2, o2, z2 = _generate_one_hmm(batch_size, seq_len, params.p_stay2, params.p_emit2, device)
    # Four-token vocabulary: bit0 = observed evidence for controlled HMM, bit1 = nuisance observation.
    tokens = o1 + 2 * o2
    return {"tokens": tokens, "h1": h1, "h2": h2, "o1": o1, "o2": o2, "z1": z1, "z2": z2}


def next_obs_prob_from_current_p(p: torch.Tensor, p_stay: float, p_emit: float) -> torch.Tensor:
    """P(observation_{t+1}=1 | posterior P(hidden_t=1)=p)."""
    p_next_state1 = p_stay * p + (1 - p_stay) * (1 - p)
    return (1 - p_emit) + (2 * p_emit - 1) * p_next_state1


def invert_next_obs_prob_to_current_p(q_obs1: torch.Tensor, p_stay: float, p_emit: float) -> torch.Tensor:
    """Invert q = P(obs_{t+1}=1) to implied P(hidden_t=1)."""
    denom_emit = 2 * p_emit - 1
    denom_trans = 2 * p_stay - 1
    p_next_state1 = (q_obs1 - (1 - p_emit)) / denom_emit
    p_current = (p_next_state1 - (1 - p_stay)) / denom_trans
    return p_current.clamp(1e-5, 1 - 1e-5)


def marginal_probs_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return predicted next-observation marginals for HMM #1 and #2.

    logits shape: [B, T, 4]. token bits: token % 2 is o1, token // 2 is o2.
    """
    probs = F.softmax(logits, dim=-1)
    # tokens 1 and 3 have bit0=1; tokens 2 and 3 have bit1=1
    q1 = probs[..., 1] + probs[..., 3]
    q2 = probs[..., 2] + probs[..., 3]
    return q1, q2


# -----------------------------
# Tiny causal transformer
# -----------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4,
        seq_len: int = 64,
        d_model: int = 96,
        n_layers: int = 4,
        n_heads: int = 4,
        d_mlp: int = 384,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_mlp, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        return_acts: bool = False,
        control: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        B, T = idx.shape
        assert T <= self.seq_len, f"input length {T} exceeds configured seq_len {self.seq_len}"
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        acts: List[torch.Tensor] = []
        for li, block in enumerate(self.blocks):
            x = block(x)
            if control is not None and li == int(control["layer"]):
                strengths = control["strengths"].to(x.device, x.dtype)  # [B,T]
                direction = control["direction"].to(x.device, x.dtype)  # [C]
                x = x + strengths.unsqueeze(-1) * direction.view(1, 1, -1)
            if return_acts:
                acts.append(x)
        logits = self.head(self.ln_f(x))
        return logits, acts if return_acts else None


# -----------------------------
# Probe fitting and evaluation
# -----------------------------


@dataclass
class RidgeProbe:
    w: torch.Tensor
    b: torch.Tensor
    layer: int
    target_name: str
    train_mean: Optional[torch.Tensor] = None
    train_std: Optional[torch.Tensor] = None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [..., C]
        if self.train_mean is not None:
            x = (x - self.train_mean.to(x.device)) / self.train_std.to(x.device).clamp_min(1e-6)
        return x @ self.w.to(x.device) + self.b.to(x.device)


def fit_ridge_probe(X: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3, standardize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form ridge regression with intercept.

    Returns w, b, mean, std such that y_hat = ((X-mean)/std) @ w + b.
    """
    X = X.float()
    y = y.float().view(-1, 1)
    if standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xs = (X - mean) / std
    else:
        mean = torch.zeros((1, X.shape[1]), device=X.device)
        std = torch.ones((1, X.shape[1]), device=X.device)
        Xs = X
    ones = torch.ones((Xs.shape[0], 1), device=X.device, dtype=X.dtype)
    Xa = torch.cat([Xs, ones], dim=1)
    d = Xa.shape[1]
    reg = ridge * torch.eye(d, device=X.device, dtype=X.dtype)
    reg[-1, -1] = 0.0
    beta = torch.linalg.solve((Xa.T @ Xa) / Xa.shape[0] + reg, (Xa.T @ y) / Xa.shape[0])
    w = beta[:-1, 0].detach()
    b = beta[-1, 0].detach()
    return w, b, mean.squeeze(0).detach(), std.squeeze(0).detach()


def r2_rmse(y: torch.Tensor, yhat: torch.Tensor) -> Tuple[float, float]:
    y = y.float().view(-1)
    yhat = yhat.float().view(-1)
    mse = torch.mean((y - yhat) ** 2)
    var = torch.var(y, unbiased=False).clamp_min(1e-8)
    r2 = 1.0 - mse / var
    return float(r2.detach().cpu()), float(torch.sqrt(mse).detach().cpu())


def collect_activations(
    model: TinyCausalTransformer,
    params: HMMParams,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    seq_len: int,
    max_samples: int,
) -> Dict[str, List[torch.Tensor]]:
    model.eval()
    X_by_layer: List[List[torch.Tensor]] = [[] for _ in range(model.n_layers)]
    y1_list: List[torch.Tensor] = []
    y2_list: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="collect activations", leave=False):
            batch = generate_batch(batch_size, seq_len, params, device)
            idx = batch["tokens"][:, :-1]
            z1 = batch["z1"][:, :-1]
            z2 = batch["z2"][:, :-1]
            _, acts = model(idx, return_acts=True)
            assert acts is not None
            B, T = idx.shape
            flat_idx = torch.randperm(B * T, device=device)[: min(max_samples // n_batches + 1, B * T)]
            for li, a in enumerate(acts):
                X_by_layer[li].append(a.reshape(B * T, -1)[flat_idx].detach().cpu())
            y1_list.append(z1.reshape(B * T)[flat_idx].detach().cpu())
            y2_list.append(z2.reshape(B * T)[flat_idx].detach().cpu())
    X_layers = [torch.cat(chunks, dim=0)[:max_samples] for chunks in X_by_layer]
    y1 = torch.cat(y1_list, dim=0)[:max_samples]
    y2 = torch.cat(y2_list, dim=0)[:max_samples]
    return {"X_layers": X_layers, "z1": y1, "z2": y2}


# -----------------------------
# Training and experiments
# -----------------------------


@dataclass
class ExperimentConfig:
    seed: int = 7
    seq_len: int = 64
    batch_size: int = 256
    steps: int = 1500
    eval_every: int = 100
    lr: float = 3e-4
    d_model: int = 96
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 384
    dropout: float = 0.0
    probe_batches: int = 40
    probe_samples: int = 250_000
    control_batches: int = 20
    control_batch_size: int = 256
    target_z: float = 2.0
    controller_gain: float = 0.8
    max_strength: float = 8.0
    ridge: float = 1e-3
    aux_posterior_weight: float = 0.0
    device: str = "auto"
    outdir: str = "runs/hmm_control"


def evaluate_next_token(model: TinyCausalTransformer, params: HMMParams, device: torch.device, batch_size: int, seq_len: int, n_batches: int = 10) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    bayes_losses: List[float] = []
    with torch.no_grad():
        for _ in range(n_batches):
            batch = generate_batch(batch_size, seq_len, params, device)
            idx = batch["tokens"][:, :-1]
            target = batch["tokens"][:, 1:]
            logits, _ = model(idx)
            loss = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1))
            losses.append(float(loss.detach().cpu()))

            # Bayes-optimal joint next-token distribution factorizes across the two independent HMMs.
            p1 = sigmoid_from_logit(batch["z1"][:, :-1])
            p2 = sigmoid_from_logit(batch["z2"][:, :-1])
            q1 = next_obs_prob_from_current_p(p1, params.p_stay1, params.p_emit1)
            q2 = next_obs_prob_from_current_p(p2, params.p_stay2, params.p_emit2)
            target_o1 = (target % 2).float()
            target_o2 = (target // 2).float()
            nll1 = -target_o1 * torch.log(q1.clamp_min(1e-6)) - (1 - target_o1) * torch.log1p(-q1.clamp_max(1 - 1e-6))
            nll2 = -target_o2 * torch.log(q2.clamp_min(1e-6)) - (1 - target_o2) * torch.log1p(-q2.clamp_max(1 - 1e-6))
            bayes_losses.append(float((nll1 + nll2).mean().detach().cpu()))
    return {"model_ce": float(np.mean(losses)), "bayes_ce": float(np.mean(bayes_losses)), "bayes_gap": float(np.mean(losses) - np.mean(bayes_losses))}


def train_model(cfg: ExperimentConfig, params: HMMParams, device: torch.device, outdir: Path) -> TinyCausalTransformer:
    model = TinyCausalTransformer(
        vocab_size=4,
        seq_len=cfg.seq_len - 1,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    aux_head = nn.Linear(cfg.d_model, 2).to(device) if cfg.aux_posterior_weight > 0 else None
    if aux_head is not None:
        opt = torch.optim.AdamW(list(model.parameters()) + list(aux_head.parameters()), lr=cfg.lr, weight_decay=0.01)

    rows = []
    pbar = tqdm(range(1, cfg.steps + 1), desc="train")
    for step in pbar:
        model.train()
        batch = generate_batch(cfg.batch_size, cfg.seq_len, params, device)
        idx = batch["tokens"][:, :-1]
        target = batch["tokens"][:, 1:]
        logits, acts = model(idx, return_acts=(aux_head is not None))
        loss = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1))
        if aux_head is not None and acts is not None:
            z_targets = torch.stack([batch["z1"][:, :-1], batch["z2"][:, :-1]], dim=-1)
            pred_z = aux_head(acts[-1])
            aux_loss = F.mse_loss(pred_z, z_targets.clamp(-6, 6))
            loss = loss + cfg.aux_posterior_weight * aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pbar.set_postfix(loss=float(loss.detach().cpu()))

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate_next_token(model, params, device, cfg.batch_size, cfg.seq_len, n_batches=5)
            row = {"step": step, "train_loss": float(loss.detach().cpu()), **metrics}
            rows.append(row)
            pd.DataFrame(rows).to_csv(outdir / "learning_curve.csv", index=False)

    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "params": asdict(params)}, outdir / "model.pt")
    return model


def train_probes(model: TinyCausalTransformer, cfg: ExperimentConfig, params: HMMParams, device: torch.device, outdir: Path) -> Dict[str, object]:
    data_train = collect_activations(model, params, device, cfg.probe_batches, cfg.batch_size, cfg.seq_len, cfg.probe_samples)
    data_val = collect_activations(model, params, device, max(5, cfg.probe_batches // 4), cfg.batch_size, cfg.seq_len, max(50_000, cfg.probe_samples // 5))

    rows = []
    probes_z1: List[RidgeProbe] = []
    probes_z2: List[RidgeProbe] = []
    for li in range(model.n_layers):
        Xtr = data_train["X_layers"][li]
        Xva = data_val["X_layers"][li]
        for name, probes, ytr, yva in [
            ("z1", probes_z1, data_train["z1"], data_val["z1"]),
            ("z2", probes_z2, data_train["z2"], data_val["z2"]),
        ]:
            w, b, mean, std = fit_ridge_probe(Xtr, ytr, ridge=cfg.ridge)
            probe = RidgeProbe(w=w, b=b, layer=li, target_name=name, train_mean=mean, train_std=std)
            yhat_tr = probe.predict(Xtr)
            yhat_va = probe.predict(Xva)
            r2_tr, rmse_tr = r2_rmse(ytr, yhat_tr)
            r2_va, rmse_va = r2_rmse(yva, yhat_va)
            rows.append({"layer": li, "target": name, "r2_train": r2_tr, "rmse_train": rmse_tr, "r2_val": r2_va, "rmse_val": rmse_va})
            probes.append(probe)

    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(outdir / "probe_by_layer.csv", index=False)
    best_row = probe_df[probe_df["target"] == "z1"].sort_values("rmse_val").iloc[0]
    best_layer = int(best_row["layer"])
    best_z1 = probes_z1[best_layer]
    best_z2 = probes_z2[best_layer]

    # Actuation direction: high-z1 minus low-z1 mean at best layer, not the same vector as the probe weights.
    X = data_train["X_layers"][best_layer]
    y1 = data_train["z1"]
    lo = torch.quantile(y1, 0.20)
    hi = torch.quantile(y1, 0.80)
    direction = X[y1 >= hi].mean(dim=0) - X[y1 <= lo].mean(dim=0)
    direction = direction / direction.norm().clamp_min(1e-8)
    # Gain: how much the z1 probe readout changes per unit of this actuation direction.
    dir_standardized = direction / best_z1.train_std.cpu().clamp_min(1e-6)
    gain = float((best_z1.w.cpu() * dir_standardized).sum().item())
    if abs(gain) < 1e-6:
        gain = 1e-6 if gain >= 0 else -1e-6

    torch.save(
        {
            "probes_z1": probes_z1,
            "probes_z2": probes_z2,
            "best_layer": best_layer,
            "direction": direction,
            "gain": gain,
            "probe_df": probe_df,
        },
        outdir / "probes.pt",
    )
    plot_probe_layers(probe_df, outdir)
    return {"probes_z1": probes_z1, "probes_z2": probes_z2, "best_layer": best_layer, "direction": direction, "gain": gain, "probe_df": probe_df}


def observer_predictions(
    observer_name: str,
    z_true: torch.Tensor,
    z2_true: torch.Tensor,
    act: torch.Tensor,
    probe: RidgeProbe,
    sigma: float = 0.0,
) -> torch.Tensor:
    if observer_name == "oracle":
        return z_true
    if observer_name == "noisy_oracle":
        return z_true + sigma * torch.randn_like(z_true)
    if observer_name == "linear_probe":
        return probe.predict(act)
    if observer_name == "last_obs_proxy":
        # A simple causal-but-incomplete observer: use only the sign/magnitude of the last observation's evidence.
        return 1.5 * torch.sign(z_true)  # intentionally coarse; correlated with z but not calibrated
    if observer_name == "entangled_bad":
        # A confounded observer: controlled belief contaminated by nuisance belief.
        return z_true + 0.9 * z2_true
    raise ValueError(f"unknown observer {observer_name}")


def run_control_eval(
    model: TinyCausalTransformer,
    cfg: ExperimentConfig,
    params: HMMParams,
    device: torch.device,
    outdir: Path,
    probe_pack: Dict[str, object],
) -> pd.DataFrame:
    model.eval()
    best_layer: int = int(probe_pack["best_layer"])
    direction = probe_pack["direction"].to(device).float()
    gain = float(probe_pack["gain"])
    probe_z1: RidgeProbe = probe_pack["probes_z1"][best_layer]
    probe_z2: RidgeProbe = probe_pack["probes_z2"][best_layer]

    observers: List[Tuple[str, float]] = [
        ("oracle", 0.0),
        ("linear_probe", 0.0),
        ("last_obs_proxy", 0.0),
        ("entangled_bad", 0.0),
    ]
    for sigma in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        observers.append(("noisy_oracle", sigma))

    rows = []
    for observer_name, sigma in tqdm(observers, desc="control observers"):
        metrics_accum: Dict[str, List[float]] = {k: [] for k in [
            "observer_rmse", "base_target_mse", "control_target_mse", "control_target_kl",
            "natural_ce_base", "natural_ce_control", "collateral_q2_abs", "collateral_q2_kl", "mean_abs_strength"
        ]}
        for _ in range(cfg.control_batches):
            batch = generate_batch(cfg.control_batch_size, cfg.seq_len, params, device)
            idx = batch["tokens"][:, :-1]
            target = batch["tokens"][:, 1:]
            z1_true = batch["z1"][:, :-1]
            z2_true = batch["z2"][:, :-1]
            target_z = torch.full_like(z1_true, cfg.target_z)
            p_target = torch.sigmoid(target_z)
            q1_target = next_obs_prob_from_current_p(p_target, params.p_stay1, params.p_emit1)

            with torch.no_grad():
                base_logits, acts = model(idx, return_acts=True)
                assert acts is not None
                act = acts[best_layer]
                zhat = observer_predictions(observer_name, z1_true, z2_true, act, probe_z1, sigma=sigma)
                observer_rmse = torch.sqrt(torch.mean((zhat - z1_true) ** 2))

                # P controller: amplitude scaled by the fixed z-readout gain of the actuation direction.
                strengths = cfg.controller_gain * (target_z - zhat) / gain
                strengths = strengths.clamp(-cfg.max_strength, cfg.max_strength)
                control = {"layer": best_layer, "direction": direction, "strengths": strengths}
                controlled_logits, controlled_acts = model(idx, return_acts=True, control=control)
                assert controlled_acts is not None

                q1_base, q2_base = marginal_probs_from_logits(base_logits)
                q1_ctrl, q2_ctrl = marginal_probs_from_logits(controlled_logits)
                p1_base_implied = invert_next_obs_prob_to_current_p(q1_base, params.p_stay1, params.p_emit1)
                p1_ctrl_implied = invert_next_obs_prob_to_current_p(q1_ctrl, params.p_stay1, params.p_emit1)
                z1_base_implied = logit(p1_base_implied)
                z1_ctrl_implied = logit(p1_ctrl_implied)

                base_target_mse = torch.mean((z1_base_implied - target_z) ** 2)
                control_target_mse = torch.mean((z1_ctrl_implied - target_z) ** 2)
                control_target_kl = torch.mean(bern_kl(q1_target, q1_ctrl))
                ce_base = F.cross_entropy(base_logits.reshape(-1, 4), target.reshape(-1))
                ce_ctrl = F.cross_entropy(controlled_logits.reshape(-1, 4), target.reshape(-1))
                collateral_abs = torch.mean(torch.abs(q2_ctrl - q2_base))
                collateral_kl = torch.mean(bern_kl(q2_base, q2_ctrl))
                mean_abs_strength = torch.mean(torch.abs(strengths))

            for key, value in [
                ("observer_rmse", observer_rmse),
                ("base_target_mse", base_target_mse),
                ("control_target_mse", control_target_mse),
                ("control_target_kl", control_target_kl),
                ("natural_ce_base", ce_base),
                ("natural_ce_control", ce_ctrl),
                ("collateral_q2_abs", collateral_abs),
                ("collateral_q2_kl", collateral_kl),
                ("mean_abs_strength", mean_abs_strength),
            ]:
                metrics_accum[key].append(float(value.detach().cpu()))
        row = {"observer": observer_name, "sigma": sigma, "layer": best_layer, "gain": gain}
        row.update({k: float(np.mean(v)) for k, v in metrics_accum.items()})
        row["control_improvement_mse"] = row["base_target_mse"] - row["control_target_mse"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "observer_control.csv", index=False)
    plot_control(df, outdir)
    return df


# -----------------------------
# Plotting
# -----------------------------


def plot_learning_curve(outdir: Path) -> None:
    path = outdir / "learning_curve.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["model_ce"], marker="o", label="model CE")
    plt.plot(df["step"], df["bayes_ce"], marker="o", label="Bayes-optimal CE")
    plt.xlabel("training step")
    plt.ylabel("next-token cross entropy")
    plt.title("Training vs analytic Bayes baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve.png", dpi=160)
    plt.close()


def plot_probe_layers(probe_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(7, 4))
    for target in sorted(probe_df["target"].unique()):
        sub = probe_df[probe_df["target"] == target]
        plt.plot(sub["layer"], sub["r2_val"], marker="o", label=target)
    plt.xlabel("residual layer")
    plt.ylabel("validation R²")
    plt.title("Linear decodability of analytic posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "probe_by_layer.png", dpi=160)
    plt.close()


def plot_control(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(7, 5))
    # Primary scatter: observer error vs controlled target loss.
    for _, row in df.iterrows():
        label = row["observer"] if row["observer"] != "noisy_oracle" else f"noisy σ={row['sigma']}"
        plt.scatter(row["observer_rmse"], row["control_target_mse"])
        plt.text(row["observer_rmse"] + 0.01, row["control_target_mse"], label, fontsize=8)
    plt.xlabel("observer RMSE against true z₁")
    plt.ylabel("closed-loop target MSE")
    plt.title("Figure 1: Does control quality track observer quality?")
    plt.tight_layout()
    plt.savefig(outdir / "observer_rmse_vs_control_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for _, row in df.iterrows():
        label = row["observer"] if row["observer"] != "noisy_oracle" else f"noisy σ={row['sigma']}"
        plt.scatter(row["control_target_mse"], row["collateral_q2_abs"])
        plt.text(row["control_target_mse"] + 0.01, row["collateral_q2_abs"], label, fontsize=8)
    plt.xlabel("closed-loop target MSE on controlled belief z₁")
    plt.ylabel("collateral movement in nuisance marginal q₂")
    plt.title("Specificity: target control vs nuisance damage")
    plt.tight_layout()
    plt.savefig(outdir / "control_loss_vs_collateral.png", dpi=180)
    plt.close()


def save_summary(outdir: Path, cfg: ExperimentConfig, params: HMMParams, probe_pack: Dict[str, object], control_df: pd.DataFrame) -> None:
    best_probe = probe_pack["probe_df"][(probe_pack["probe_df"]["target"] == "z1")].sort_values("rmse_val").iloc[0].to_dict()
    spearman = float(control_df[["observer_rmse", "control_target_mse"]].corr(method="spearman").iloc[0, 1])
    pearson = float(control_df[["observer_rmse", "control_target_mse"]].corr(method="pearson").iloc[0, 1])
    summary = {
        "config": asdict(cfg),
        "hmm_params": asdict(params),
        "best_z1_probe": best_probe,
        "best_layer": int(probe_pack["best_layer"]),
        "actuation_gain": float(probe_pack["gain"]),
        "observer_rmse_vs_control_loss_spearman": spearman,
        "observer_rmse_vs_control_loss_pearson": pearson,
        "control_results": control_df.to_dict(orient="records"),
        "interpretation_hint": (
            "If observer_rmse_vs_control_loss is monotone and oracle/noisy/probe separate, "
            "observer fidelity is binding in this task. If points collapse, this task/controller/actuator "
            "did not make observer quality binding."
        ),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def run_all(cfg: ExperimentConfig) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    params = HMMParams()
    with open(outdir / "config.json", "w") as f:
        json.dump({"cfg": asdict(cfg), "params": asdict(params), "device": str(device)}, f, indent=2)
    print(f"Using device: {device}")
    print(f"Writing results to: {outdir.resolve()}")

    model = train_model(cfg, params, device, outdir)
    plot_learning_curve(outdir)
    probe_pack = train_probes(model, cfg, params, device, outdir)
    control_df = run_control_eval(model, cfg, params, device, outdir, probe_pack)
    save_summary(outdir, cfg, params, probe_pack, control_df)
    print("\nDone. Key files:")
    for name in [
        "learning_curve.png",
        "probe_by_layer.png",
        "observer_rmse_vs_control_loss.png",
        "control_loss_vs_collateral.png",
        "summary.json",
    ]:
        print(f"  {outdir / name}")
    print("\nControl results:")
    print(control_df.sort_values("observer_rmse")[["observer", "sigma", "observer_rmse", "control_target_mse", "control_improvement_mse", "collateral_q2_abs", "mean_abs_strength"]].to_string(index=False))


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="HMM observer-fidelity vs control-quality wind tunnel")
    p.add_argument("--outdir", type=str, default="runs/hmm_control")
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-mlp", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--probe-batches", type=int, default=40)
    p.add_argument("--probe-samples", type=int, default=250000)
    p.add_argument("--control-batches", type=int, default=20)
    p.add_argument("--control-batch-size", type=int, default=256)
    p.add_argument("--target-z", type=float, default=2.0)
    p.add_argument("--controller-gain", type=float, default=0.8)
    p.add_argument("--max-strength", type=float, default=8.0)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--aux-posterior-weight", type=float, default=0.0, help="debug only; default keeps training next-token-only")
    p.add_argument("--quick", action="store_true", help="cheap smoke-test settings")
    p.add_argument("--h100", action="store_true", help="larger Colab/H100 settings")
    args = p.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_every=args.eval_every,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
        probe_batches=args.probe_batches,
        probe_samples=args.probe_samples,
        control_batches=args.control_batches,
        control_batch_size=args.control_batch_size,
        target_z=args.target_z,
        controller_gain=args.controller_gain,
        max_strength=args.max_strength,
        ridge=args.ridge,
        aux_posterior_weight=args.aux_posterior_weight,
        device=args.device,
        outdir=args.outdir,
    )
    if args.quick:
        cfg.steps = min(cfg.steps, 350)
        cfg.batch_size = min(cfg.batch_size, 128)
        cfg.d_model = min(cfg.d_model, 64)
        cfg.n_layers = min(cfg.n_layers, 3)
        cfg.n_heads = min(cfg.n_heads, 4)
        cfg.d_mlp = min(cfg.d_mlp, 256)
        cfg.probe_batches = min(cfg.probe_batches, 12)
        cfg.probe_samples = min(cfg.probe_samples, 60_000)
        cfg.control_batches = min(cfg.control_batches, 6)
        cfg.control_batch_size = min(cfg.control_batch_size, 128)
        cfg.eval_every = min(cfg.eval_every, 50)
    if args.h100:
        cfg.steps = max(cfg.steps, 6000)
        cfg.batch_size = max(cfg.batch_size, 1024)
        cfg.d_model = max(cfg.d_model, 160)
        cfg.n_layers = max(cfg.n_layers, 6)
        cfg.n_heads = max(cfg.n_heads, 8)
        cfg.d_mlp = max(cfg.d_mlp, 640)
        cfg.probe_batches = max(cfg.probe_batches, 80)
        cfg.probe_samples = max(cfg.probe_samples, 700_000)
        cfg.control_batches = max(cfg.control_batches, 50)
        cfg.control_batch_size = max(cfg.control_batch_size, 512)
        cfg.eval_every = max(cfg.eval_every, 200)
    return cfg


if __name__ == "__main__":
    run_all(parse_args())
