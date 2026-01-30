#!/usr/bin/env python
# coding: utf-8

import os
import random
from glob import glob
from itertools import product
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import treams
from joblib import Parallel, delayed

from helper import (
    ensure_stackable_T,
    wl_to_1d_nm,
    geom_key_str,
    grouped_train_val_test_stratified,
    create_r,
    load_or_compute_Rs,
    set_seed,
    Rvec_from_h5_tr,
    plot_curves,
)
print("hash('abc') =", hash("abc"))

SEED = 777
set_seed(SEED)

# ============================================================
# 1) Small helpers
# ============================================================

def compute_T_errors(T_true, T_pred):
    """
    T_true, T_pred : complex arrays [B, N, N]
    Returns dict with per-sample norms + errors.
    """
    diff = T_pred - T_true

    norm_true = np.linalg.norm(T_true, axis=(1, 2))
    norm_pred = np.linalg.norm(T_pred, axis=(1, 2))
    norm_diff = np.linalg.norm(diff,   axis=(1, 2))

    err_abs = norm_diff
    err_rel_sym = 2 * norm_diff / (norm_true + norm_pred + 1e-12)

    return dict(
        norm_true=norm_true,
        norm_pred=norm_pred,
        norm_diff=norm_diff,
        err_abs=err_abs,
        err_rel_sym=err_rel_sym,
    )

def build_reciprocal_map_fast(l, m, p, l_out=None, m_out=None, p_out=None, device="cpu"):
    """
    Build (src_idx, tgt_idx, signs, N) for reciprocity mapping.

    Returns:
      src_idx: (S,) LongTensor of linear indices i*N + j
      tgt_idx: (S,) LongTensor of linear indices a*N + b
      signs  : (S,) ComplexTensor with values ±1
      N      : int
    """
    def tonp(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x).reshape(-1)

    l = tonp(l).astype(np.int64)
    m = tonp(m).astype(np.int64)
    p = tonp(p)

    if l_out is None:
        l_out, m_out, p_out = l.copy(), m.copy(), p.copy()
    else:
        l_out = tonp(l_out).astype(np.int64)
        m_out = tonp(m_out).astype(np.int64)
        p_out = tonp(p_out)

    N = l.shape[0]
    assert l.shape == m.shape == p.shape == (N,)
    assert l_out.shape == m_out.shape == p_out.shape == (N,)

    p = (p == p[0]).astype(np.int64)
    p_out = (p_out == p_out[0]).astype(np.int64)

    key_in = list(zip(l, m, p))
    lut_in = {k: i for i, k in enumerate(key_in)}

    key_out = list(zip(l_out, m_out, p_out))
    lut_out = {k: j for j, k in enumerate(key_out)}

    a_of_j = np.full(N, -1, dtype=np.int64)
    for j in range(N):
        a_of_j[j] = lut_in.get((l_out[j], -m_out[j], p_out[j]), -1)

    b_of_i = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        b_of_i[i] = lut_out.get((l[i], -m[i], p[i]), -1)

    I = np.arange(N, dtype=np.int64)[:, None]
    J = np.arange(N, dtype=np.int64)[None, :]

    src_flat = (I * N + J).ravel()
    tgt_flat = (a_of_j[None, :] * N + b_of_i[:, None]).ravel()

    valid = ((a_of_j[None, :] >= 0) & (b_of_i[:, None] >= 0)).ravel()

    signs_mat = ((m[:, None] + m_out[None, :]) & 1)
    signs_np = np.where(signs_mat.ravel() == 0, 1.0, -1.0)

    src_idx = torch.from_numpy(src_flat[valid]).to(device=device, dtype=torch.long)
    tgt_idx = torch.from_numpy(tgt_flat[valid]).to(device=device, dtype=torch.long)
    signs = torch.from_numpy(signs_np[valid]).to(device=device, dtype=torch.complex64)
    return src_idx, tgt_idx, signs, N


def sym_frob_loss_flat(pred_flat, true_flat, eps=1e-12):
    half = pred_flat.shape[1] // 2
    pred = pred_flat[:, :half] + 1j * pred_flat[:, half:]
    true = true_flat[:, :half] + 1j * true_flat[:, half:]

    num = torch.sum(torch.abs(pred - true)**2, dim=1)
    den = torch.sum(torch.abs(pred)**2 + torch.abs(true)**2, dim=1)
    return torch.mean(num / (den + eps))


def reciprocity_loss_flat(pred_flat, src_idx, tgt_idx, signs, N, eps=1e-12):
    """
    pred_flat: [B, 2*N*N] = [Re vec, Im vec]
    """
    B = pred_flat.shape[0]
    half = pred_flat.shape[1] // 2

    pred_c = pred_flat[:, :half] + 1j * pred_flat[:, half:]
    pred_c = pred_c.view(B, N*N)

    mapped = pred_c[:, src_idx] * signs.unsqueeze(0)
    reordered = torch.zeros_like(pred_c)
    reordered.scatter_(1, tgt_idx.unsqueeze(0).expand(B, -1), mapped)

    num = torch.sum(torch.abs(reordered - pred_c) ** 2, dim=1)
    den = torch.sum(torch.abs(reordered) ** 2 + torch.abs(pred_c) ** 2, dim=1)
    return torch.mean(num / (den + eps))


@torch.no_grad()
def per_class_T_mse(model, dataloader, device="cpu"):
    """
    Returns {class_id: (mse, count)} over flattened T per sample.
    Batches: (theta, lambda, y_oh, T_flat).
    """
    model.eval().to(device)
    sums = {}
    for th, lam, y_oh, T in dataloader:
        th, lam, y_oh, T = th.to(device), lam.to(device), y_oh.to(device), T.to(device)
        T_hat = model(th, lam, y_oh)
        err = (T_hat - T).pow(2).mean(dim=1).cpu().numpy()
        cls = y_oh.argmax(1).cpu().numpy()
        for e, c in zip(err, cls):
            s = sums.setdefault(int(c), [0.0, 0])
            s[0] += float(e)
            s[1] += 1

    out = {}
    for c, (s, n) in sums.items():
        out[c] = (s / max(1, n), n)
    return out

def save_results_h5(
    path,
    *,
    te_gid, y_te, wl_te,
    R_true_te, R_pred_te,
    T_true_te, T_pred_te,
    err_T, err_R,
    R_ref, wl_ref,
    class_names=None,
    angles=None, pols=None,
    gid_spec=None, i_T=None
):
    te_gid = np.asarray(te_gid)
    y_te = np.asarray(y_te, dtype=np.int64)
    wl_te = np.asarray(wl_te, dtype=np.float32)

    R_true_te = np.asarray(R_true_te, dtype=np.float32)
    R_pred_te = np.asarray(R_pred_te, dtype=np.float32)

    T_true_te = np.asarray(T_true_te, dtype=np.complex64)
    T_pred_te = np.asarray(T_pred_te, dtype=np.complex64)

    err_T = np.asarray(err_T, dtype=np.float32)
    err_R = np.asarray(err_R, dtype=np.float32)

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        if angles is not None:
            meta.create_dataset("angles", data=np.asarray(angles, dtype=np.float32))
        if pols is not None:
            meta.create_dataset("pols", data=np.asarray(pols, dtype=np.int64))
        if gid_spec is not None:
            meta.attrs["gid_spec"] = str(gid_spec)
        if i_T is not None:
            meta.attrs["i_T"] = int(i_T)
        if class_names is not None:
            meta.create_dataset("class_names", data=np.array(class_names, dtype=object), dtype=str_dt)

        d = f.create_group("data")
        d.create_dataset("te_gid", data=te_gid.astype(str_dt), compression="gzip")
        d.create_dataset("y_te",   data=y_te, compression="gzip")
        d.create_dataset("wl_te",  data=wl_te, compression="gzip")

        d.create_dataset("R_true_te", data=R_true_te, compression="gzip")
        d.create_dataset("R_pred_te", data=R_pred_te, compression="gzip")
        d.create_dataset("R_ref",     data=R_ref, compression="gzip")
        d.create_dataset("wl_ref",    data=wl_ref, compression="gzip")

        d.create_dataset("T_true_te", data=T_true_te, compression="gzip")
        d.create_dataset("T_pred_te", data=T_pred_te, compression="gzip")

        d.create_dataset("err_T", data=err_T, compression="gzip")
        d.create_dataset("err_R", data=err_R, compression="gzip")


# ============================================================
# 2) Model definitions
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0, res_scale=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        self.res_scale = res_scale

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.res_scale * h


class ResBlockLN(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0, res_scale_init=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hid = mult * dim
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)
        self.dropout = dropout
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        return x + self.res_scale * h


class Theta2TResNet(nn.Module):
    """
    Forward model: (theta, lambda, one-hot class) -> flattened complex T-matrix.
    Output format: concat[Re(T).ravel(), Im(T).ravel()]
    """
    def __init__(self, theta_dim: int, y_dim: int, out_dim: int, width: int = 512, num_blocks: int = 4):
        super().__init__()
        in_dim = theta_dim + 1 + y_dim
        self.in_layer = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.out_layer = nn.Linear(width, out_dim)

    def forward(self, theta_n, lambda_n, y_oh):
        x = torch.cat([theta_n, lambda_n, y_oh], dim=1)
        h = F.gelu(self.in_layer(x))
        h = self.blocks(h)
        return self.out_layer(h)


class Theta2TResNet_ClassHead(nn.Module):
    def __init__(self, theta_dim, y_dim, out_dim, width=512, num_blocks=4):
        super().__init__()
        self.y_dim = y_dim
        in_dim = theta_dim + 1 + y_dim
        self.in_layer = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlockLN(width) for _ in range(num_blocks)])
        self.heads = nn.ModuleList([nn.Linear(width, out_dim) for _ in range(y_dim)])

    def forward(self, theta_n, lambda_n, y_oh):
        x = torch.cat([theta_n, lambda_n, y_oh], dim=1)
        h = F.gelu(self.in_layer(x))
        h = self.blocks(h)
        y_int = y_oh.argmax(1)
        out = torch.zeros(h.size(0), self.heads[0].out_features, device=h.device, dtype=h.dtype)
        for k in range(self.y_dim):
            m = (y_int == k)
            if m.any():
                out[m] = self.heads[k](h[m])
        return out


# ============================================================
# 3) Config
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

lmax = 5
maxlen = 2 * lmax * (lmax + 2)
class_total = 3000

library = {0: "cone", 1: "cuboid", 2: "cylinder"}

folders = {
    0: "cones/",
    1: "cuboids/",
    2: "cylinders/",
} # MODIFY PATH

    
WL_MIN_NM = 700.0
WL_MAX_NM = 1000.0
GAP_NM = 200.0

rmax_coef = 3.0
pols = (0, 1)
poltype = "parity"
treams.config.POLTYPE = poltype

angles = np.array([0.0, np.pi / 6, np.pi / 4])


# ============================================================
# 4) Load dataset (T, labels, params, wl, embedding, groups)
# ============================================================

t_matrices_full = []
true_labels_full = []
true_params_full = []
wls_full = []
emb_full = []
groups = []
circ_spheres = []
counts = {0: 0, 1: 0, 2: 0}

for cls, folder in folders.items():
    files = sorted(glob(os.path.join(folder, "*.h5")))
    for fp in files:
        if counts[cls] >= class_total:
            break
        with h5py.File(fp, "r") as f:
            T = np.array(f["tmatrix"][...]) 
            T = ensure_stackable_T(T, maxlen)
            wl = wl_to_1d_nm(f["vacuum_wavelength"][...], T.shape[0])

            sel = np.arange(T.shape[0])
            mask = (wl >= WL_MIN_NM) & (wl <= WL_MAX_NM)
            sel = sel[mask]
            if sel.size == 0:
                continue

            remaining = class_total - counts[cls]
            if remaining <= 0:
                continue
            sel = sel[:remaining]

            g = f["scatterer/geometry"]
            if cls == 0:
                rtop = float(g["radius_top"][()])
                rbot = float(g["radius_bottom"][()])
                h = float(g["height"][()])
                base_p = [rtop, rbot, h]
                rmax = max(rtop, rbot)
                a = np.sqrt(rmax**2 + (0.5 * h)**2)
            elif cls == 1:
                Lx = float(g["lengthx"][()])
                Ly = float(g["lengthy"][()])
                Lz = float(g["lengthz"][()])
                base_p = [Lx, Ly, Lz]
                a = 0.5 * np.sqrt(Lx**2 + Ly**2 + Lz**2)
            else:
                r = float(g["radius"][()])
                h = float(g["height"][()])
                base_p = [r, 0.0, h]
                a = np.sqrt(r**2 + (0.5 * h)**2)

            emb = f["embedding/relative_permittivity"][...]

            T_sel = T[sel]
            wl_sel = wl[sel]
            k = sel.size
            P_sel = np.repeat(np.array(base_p, float)[None, :], k, axis=0)

            circ_spheres.extend([float(a)] * k)
            t_matrices_full.extend(list(T_sel))
            true_labels_full.extend([cls] * k)
            emb_full.extend([emb] * k)
            true_params_full.extend(list(P_sel))
            wls_full.extend(list(wl_sel))
            counts[cls] += k

            gkey = geom_key_str(cls, base_p, emb, nd=4)
            groups.extend([gkey] * k)

circ_spheres = np.array(circ_spheres)
t_matrices_full = np.asarray(t_matrices_full)
true_labels_full = np.asarray(true_labels_full, int)
true_params_full = np.asarray(true_params_full, float)
wls_full = np.asarray(wls_full, float)
emb_full = np.array(emb_full)
groups = np.array(groups)

uniq_g, cnt_g = np.unique(groups, return_counts=True)
count_map = dict(zip(uniq_g, cnt_g))
min_wls = 5
keep = np.array([count_map[g] >= min_wls for g in groups], dtype=bool)

t_matrices_full = t_matrices_full[keep]
true_labels_full = true_labels_full[keep]
true_params_full = true_params_full[keep]
wls_full = wls_full[keep]
emb_full = emb_full[keep]
circ_spheres = circ_spheres[keep]
groups = groups[keep]

print("Loaded:")
print("  T:", t_matrices_full.shape, "(complex)")
print("  labels:", true_labels_full.shape)
print("  params:", true_params_full.shape)
print("  wl:", wls_full.shape)
print("  groups:", groups.shape)

B, N_modes, _ = t_matrices_full.shape

ps = 2.0 * circ_spheres + GAP_NM
k0s = 2 * np.pi / wls_full



# ============================================================
# 5)  Rs from true T
# ============================================================

title = (
    f"angles_pol_{pols[0]}_{pols[1]}_{angles[0]}_{angles[-1]}_{len(angles)}"
    f"_gap_{GAP_NM}_rmax_{rmax_coef}_lmax_{lmax}_min_wls_{min_wls}"
)
print("TITLE", title)

Rs = load_or_compute_Rs(title, t_matrices_full, k0s, emb_full, ps, angles, pols, rmax_coef, poltype, lmax)

# ============================================================
# 6) Prepare tensors (theta, lambda, y, T_flat)
# ============================================================

t_matrices_trunc = t_matrices_full
T_real = t_matrices_trunc.real.reshape(B, -1)
T_imag = t_matrices_trunc.imag.reshape(B, -1)
T_flat = np.concatenate([T_real, T_imag], axis=1).astype(np.float32)
out_dim_T = T_flat.shape[1]
print("Flattened T:", T_flat.shape, "(out_dim_T =", out_dim_T, ")")

y = true_labels_full.copy()
theta_phys = true_params_full.astype(np.float32)
wl_nm = wls_full.astype(np.float32)

theta_size = theta_phys  # optional scaling by wl

tr_idx, va_idx, te_idx = grouped_train_val_test_stratified(
    y, groups, test_size=0.15, val_size=0.15
)
te_gid = groups[te_idx]
print("Split sizes: train", tr_idx.size, "val", va_idx.size, "test", te_idx.size)

wl_tr = wl_nm[tr_idx]
lam_mu = wl_tr.mean()
lam_sd = wl_tr.std() + 1e-9
lam_all = (wl_nm - lam_mu) / lam_sd

theta_tr, theta_va, theta_te = theta_size[tr_idx], theta_size[va_idx], theta_size[te_idx]
lam_tr, lam_va, lam_te = lam_all[tr_idx, None], lam_all[va_idx, None], lam_all[te_idx, None]
T_tr, T_va, T_te = T_flat[tr_idx], T_flat[va_idx], T_flat[te_idx]
y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
R_tr, R_va, R_te = Rs[tr_idx], Rs[va_idx], Rs[te_idx]

K = int(y.max()) + 1
for i in np.unique(y):
    print("class number", int(i), int(np.sum(y == i)))

mu_th_c, sd_th_c = {}, {}
for k in range(K):
    m = (y_tr == k)
    mu_th_c[k] = theta_tr[m].mean(axis=0, keepdims=True)
    sd_th_c[k] = theta_tr[m].std(axis=0, keepdims=True) + 1e-6


def norm_theta_classwise(theta_arr, y_int, mu_dict, sd_dict):
    out = np.empty_like(theta_arr, dtype=np.float32)
    for k in range(len(mu_dict)):
        m = (y_int == k)
        out[m] = (theta_arr[m] - mu_dict[k]) / sd_dict[k]
    return out


th_tr_n = norm_theta_classwise(theta_tr, y_tr, mu_th_c, sd_th_c)
th_va_n = norm_theta_classwise(theta_va, y_va, mu_th_c, sd_th_c)
th_te_n = norm_theta_classwise(theta_te, y_te, mu_th_c, sd_th_c)

ytr_oh = F.one_hot(torch.from_numpy(y_tr).long(), num_classes=K).float()
yva_oh = F.one_hot(torch.from_numpy(y_va).long(), num_classes=K).float()
yte_oh = F.one_hot(torch.from_numpy(y_te).long(), num_classes=K).float()

th_tr_t = torch.from_numpy(th_tr_n).float()
th_va_t = torch.from_numpy(th_va_n).float()
th_te_t = torch.from_numpy(th_te_n).float()

lam_tr_t = torch.from_numpy(lam_tr).float()
lam_va_t = torch.from_numpy(lam_va).float()
lam_te_t = torch.from_numpy(lam_te).float()

T_tr_t = torch.from_numpy(T_tr).float()
T_va_t = torch.from_numpy(T_va).float()
T_te_t = torch.from_numpy(T_te).float()

ds_tr_T = TensorDataset(th_tr_t, lam_tr_t, ytr_oh, T_tr_t)
ds_va_T = TensorDataset(th_va_t, lam_va_t, yva_oh, T_va_t)
ds_te_T = TensorDataset(th_te_t, lam_te_t, yte_oh, T_te_t)


g = torch.Generator().manual_seed(SEED)

dl_tr_T = DataLoader(ds_tr_T, batch_size=256, shuffle=True, generator=g)
dl_va_T = DataLoader(ds_va_T, batch_size=512, shuffle=False, generator=g)
dl_te_T = DataLoader(ds_te_T, batch_size=512, shuffle=False, generator=g)


# ============================================================
# 8) Train param,->T 
# ============================================================

num_blocks = 4
width = 1024
lr = 1e-3
alpha = 0.0
steps = 150

model_T = Theta2TResNet(
    theta_dim=th_tr_n.shape[1],
    y_dim=K,
    out_dim=out_dim_T,
    width=width,
    num_blocks=num_blocks,
).to(device)

opt_T = torch.optim.AdamW(model_T.parameters(), lr=lr, weight_decay=1e-4)
sched_T = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_T, mode="min", factor=0.5, patience=5
)

hist = {
    "ep": [],
    "train_data": [],
    "train_total": [],
    "val_data": [],
    "val_total": [],
}

best_val = float("inf")
best_state_T = None

n_params = sum(p.numel() for p in model_T.parameters() if p.requires_grad)
print("Trainable parameters (DoF):", n_params)

for ep in range(steps):
    model_T.train()
    td_sum = tr_sum = tt_sum = n = 0

    for thb, lb, yb, Tb in dl_tr_T:
        thb, lb, yb, Tb = thb.to(device), lb.to(device), yb.to(device), Tb.to(device)
        pred = model_T(thb, lb, yb)

        loss_data = sym_frob_loss_flat(pred, Tb)
        
        loss = loss_data #+ alpha * loss_rec

        bs = Tb.size(0)
        opt_T.zero_grad()
        loss.backward()
        opt_T.step()

        td_sum += loss_data.item() * bs
        
        tt_sum += loss.item() * bs
        n += bs

    train_data = td_sum / max(1, n)
    train_total = tt_sum / max(1, n)

    model_T.eval()
    vd_sum = vr_sum = sum_total = n_va = 0
    with torch.no_grad():
        for thb, lb, yb, Tb in dl_va_T:
            thb, lb, yb, Tb = thb.to(device), lb.to(device), yb.to(device), Tb.to(device)
            pred = model_T(thb, lb, yb)

            loss_data = sym_frob_loss_flat(pred, Tb)
            loss = loss_data #+ alpha * loss_rec

            bs = Tb.size(0)
            vd_sum += loss_data.item() * bs
            sum_total += loss.item() * bs
            n_va += bs

    val_data = vd_sum / max(1, n_va)
    val_total = sum_total / max(1, n_va)

    sched_T.step(val_total)
    if val_total < best_val:
        best_val = val_total
        best_state_T = {k: v.detach().cpu().clone() for k, v in model_T.state_dict().items()}

    hist["ep"].append(ep)
    hist["train_data"].append(train_data)
    hist["train_total"].append(train_total)
    hist["val_data"].append(val_data)
    hist["val_total"].append(val_total)

    if ep % 10 == 0 or ep == 1:
        print(f"[T] ep{ep:03d}  train={train_total:.4e}  val={val_total:.4e}")

model_T.load_state_dict(best_state_T)
model_T.to(device).eval()
plot_curves(hist, title=title, keys=["loss"], save_path="figs/paramtot_curves")

# ============================================================
# 9)  param->T on test and T to R 
# ============================================================

preds_T, trues_T = [], []
with torch.no_grad():
    for thb, lb, yb, Tb in dl_te_T:
        thb, lb, yb = thb.to(device), lb.to(device), yb.to(device)
        pn = model_T(thb, lb, yb).cpu().numpy()
        preds_T.append(pn)
        trues_T.append(Tb.numpy())

T_hat_te = np.concatenate(preds_T, axis=0)
T_true_te = np.concatenate(trues_T, axis=0)

global_mse = np.mean((T_hat_te - T_true_te) ** 2)
print("Test MSE on flattened T:", global_mse)

per_cls = per_class_T_mse(model_T, dl_te_T, device=device)
for c, (mse_c, n_c) in per_cls.items():
    print(f"class {c} ({library[c]}): MSE={mse_c:.4e}, num={n_c}")

B_te = T_hat_te.shape[0]
half = T_hat_te.shape[1] // 2

T_true_real = T_true_te[:, :half].reshape(B_te, N_modes, N_modes)
T_true_imag = T_true_te[:, half:].reshape(B_te, N_modes, N_modes)
T_hat_real = T_hat_te[:, :half].reshape(B_te, N_modes, N_modes)
T_hat_imag = T_hat_te[:, half:].reshape(B_te, N_modes, N_modes)

T_true_c = T_true_real + 1j * T_true_imag
T_hat_c = T_hat_real + 1j * T_hat_imag

num = np.linalg.norm(T_hat_c - T_true_c, axis=(1, 2))
den = np.linalg.norm(T_true_c, axis=(1, 2)) + 1e-12
rel_err = num / den
print("Mean relative Frobenius error on T:", float(rel_err.mean()))
print("Median relative error:", float(np.median(rel_err)))

idx_best = int(np.argmin(rel_err))
idx_worst = int(np.argmax(rel_err))
print("Best sample index:", idx_best, "err=", float(rel_err[idx_best]))
print("Worst sample index:", idx_worst, "err=", float(rel_err[idx_worst]))

n_te = len(te_idx)
n_kpar = len(angles)
n_pols = len(pols)

ijk_test = list(product(range(n_te), range(n_kpar), range(n_pols)))

k0s_te = k0s[te_idx]
ps_te = ps[te_idx]
emb_te = [emb_full[g] for g in te_idx]

rs_list_pred = Parallel(n_jobs=-1, backend="loky")(
    delayed(create_r)(
        i, j, p,
        T_hat_c,
        k0s_te,
        emb_te,
        ps_te,
        angles,
        pols, 
        rmax_coef, 
        poltype,
        lmax=lmax,
    )
    for (i, j, p) in ijk_test
)

Rs_pred_te_flat = np.array(rs_list_pred)
Rs_pred_te = Rs_pred_te_flat.reshape(n_te, n_kpar * n_pols)
print("Rs_pred_te", Rs_pred_te.shape, R_te.shape)

mae_r = np.mean(np.abs(Rs_pred_te - R_te), axis=1)
rmse = float(np.sqrt(np.mean((Rs_pred_te - R_te) ** 2)))

wl_te = wl_nm[te_idx]

uniq = np.unique(te_gid)
var_geom = np.array([R_te[te_gid == g].std() for g in uniq])

gid_spec = uniq[np.argsort(var_geom)[len(var_geom) // 2]]
print("avg", gid_spec)
gid_spec = uniq[np.argsort(var_geom)[-1]]
print("worst", gid_spec)

reffile = "cylinder_si_r_110.0_h_190.0_l_5_wls_7.000000000000001e-07_1.0000000000000002e-06_61_msl_2_3_domain_500_500.tmat.h5" #MODIFY PATH

R_ref, wl_ref = Rvec_from_h5_tr(reffile, angles=angles, GAP_NM=GAP_NM, rmax_coef=rmax_coef)

eps = 1e-12
diff = T_hat_c - T_true_c
num = np.linalg.norm(diff, axis=(1, 2))
den = np.linalg.norm(T_true_c, axis=(1, 2)) + np.linalg.norm(T_hat_c, axis=(1, 2)) + eps
err_T = 2.0 * num / den

mask = (te_gid == gid_spec)
idx_candidates = np.where(mask)[0]
i_T = int(idx_candidates[np.argmax(err_T[mask])])

print("i_T", i_T)
Tt = T_true_c[i_T]
Tp = T_hat_c[i_T]
print("max|T_true|", float(np.abs(Tt).max()), "median|T_true|", float(np.median(np.abs(Tt))))
print("max|T_pred|", float(np.abs(Tp).max()), "median|T_pred|", float(np.median(np.abs(Tp))))
print("median|pred-true|", float(np.median(np.abs(Tp - Tt))))
print("grid spec", gid_spec)
filename = "results/paramtot_results.h5"
save_results_h5(
    filename,
    y_te=y_te,
    te_gid=te_gid,
    wl_te=wl_te,
    R_true_te=R_te,
    R_pred_te=Rs_pred_te,
    T_true_te=T_true_c,
    T_pred_te=T_hat_c,
    err_T=err_T,
    err_R=mae_r,
    R_ref=R_ref,
    wl_ref=wl_ref,
    angles=angles,
    pols=pols,
    class_names=list(library.values()),
    gid_spec=gid_spec,
    i_T=i_T,
)

print("Saved", filename)
