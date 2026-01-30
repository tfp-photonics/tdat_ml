#!/usr/bin/env python3
# coding: utf-8

# ============================================================
# Imports
# ============================================================

import os
from glob import glob

import h5py
import numpy as np
import treams

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from helper import  (
    ensure_stackable_T,
    wl_to_1d_nm, geom_key_str,
    grouped_train_val_test_stratified,
    load_or_compute_Rs, set_seed, plot_curves, plot_per_class_mae
)

# ============================================================
# Reproducibility 
# ============================================================
SEED = 777
set_seed(SEED)

# ============================================================
# Basic utilities
# ============================================================

def select_n_per_class(y_int, n_per_class=300, seed=0, replace_if_needed=False):
    """
    Returns indices with (about) n_per_class samples per class.

    - replace_if_needed=False: takes min(n_per_class, available) per class (no replacement)
    - replace_if_needed=True : always takes exactly n_per_class per class (uses replacement if needed)
    """
    y = np.asarray(y_int, dtype=int).ravel()
    rng = np.random.default_rng(seed)

    idx_keep = []
    classes = np.unique(y)

    for c in classes:
        idx_c = np.flatnonzero(y == c)
        if idx_c.size == 0:
            continue

        if replace_if_needed:
            take = rng.choice(idx_c, size=n_per_class, replace=(idx_c.size < n_per_class))
        else:
            take = rng.choice(idx_c, size=min(n_per_class, idx_c.size), replace=False)

        idx_keep.append(take)

    if not idx_keep:
        return np.array([], dtype=int)

    idx_keep = np.concatenate(idx_keep)
    rng.shuffle(idx_keep)

    print(
        "Selected:", idx_keep.size,
        "counts:", np.bincount(y[idx_keep], minlength=int(y.max()) + 1)
    )
    return idx_keep




def save_abcd_figdata_h5(path, *, theta_true, theta_pred, y_int,
                         class_names, parity_spec, dims_by_class):
    theta_true = np.asarray(theta_true, dtype=np.float32)
    theta_pred = np.asarray(theta_pred, dtype=np.float32)
    y_int = np.asarray(y_int, dtype=np.int64).ravel()

    if theta_true.shape != theta_pred.shape:
        raise ValueError(f"theta_true {theta_true.shape} != theta_pred {theta_pred.shape}")

    str_dt = h5py.string_dtype("utf-8")

    with h5py.File(path, "w") as f:
        f.create_dataset("theta_true", data=theta_true, compression="gzip", compression_opts=4)
        f.create_dataset("theta_pred", data=theta_pred, compression="gzip", compression_opts=4)
        f.create_dataset("y_int", data=y_int, compression="gzip", compression_opts=4)

        f.create_dataset("class_names", data=np.array(list(class_names), dtype=object), dtype=str_dt)

        gp = f.create_group("parity_spec")
        for c, spec in parity_spec.items():
            c = int(c)
            dims = np.array([int(d) for d, _ in spec], dtype=np.int32)
            names = np.array([str(n) for _, n in spec], dtype=object)

            gc = gp.create_group(f"class_{c}")
            gc.create_dataset("dims", data=dims)
            gc.create_dataset("names", data=names, dtype=str_dt)

        gd = f.create_group("dims_by_class")
        for c, dims in dims_by_class.items():
            c = int(c)
            gd.create_dataset(f"class_{c}", data=np.array(dims, dtype=np.int32))


# ============================================================
# Model + evaluation helpers
# ============================================================

@torch.no_grad()
def per_class_mae(model, dataloader, device="cpu"):
    model.eval().to(device)
    sums = {}
    for th, lam, y_oh, R in dataloader:
        th, lam, y_oh, R = th.to(device), lam.to(device), y_oh.to(device), R.to(device)
        Rhat = model(th, lam, y_oh).clamp(0, 1)
        err = (Rhat - R).abs().mean(dim=1).cpu().numpy()
        cls = y_oh.argmax(1).cpu().numpy()
        for e, c in zip(err, cls):
            s = sums.setdefault(int(c), [0.0, 0])
            s[0] += float(e)
            s[1] += 1

    out = {}
    for c, (s, n) in sums.items():
        out[c] = (s / max(1, n), n)
    return out


class CondResMLPBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)

        hidden = mult * width
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.dropout = dropout

        self.film = nn.Sequential(
            nn.Linear(cond_dim, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, 2 * width),
        )
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.film(cond)
        gamma, beta = gb.chunk(2, dim=1)

        x = self.norm(h)
        x = x * (1.0 + gamma) + beta

        x = F.silu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return h + self.res_scale * x


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        return x + h


class R2ThetaResNetClassHead(nn.Module):
    def __init__(self, n_angles: int, y_dim: int, theta_dim: int = 3,
                 width: int = 512, num_blocks: int = 4):
        super().__init__()
        self.y_dim = y_dim
        in_dim = n_angles + 1 + y_dim

        self.in_layer = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])

        self.out_all = nn.Linear(width, y_dim * theta_dim)
        self.theta_dim = theta_dim

    def forward(self, R_vec, lambda_n, y_oh):
        x = torch.cat([R_vec, lambda_n, y_oh], dim=1)
        h = F.gelu(self.in_layer(x))
        h = self.blocks(h)

        all_heads = self.out_all(h).view(h.size(0), self.y_dim, self.theta_dim)
        theta_hat = (all_heads * y_oh.unsqueeze(-1)).sum(dim=1)
        return theta_hat


# ============================================================
# Loss helpers
# ============================================================


def masked_huber(th_pred, th_true, y_int, dims_by_class, beta=1.0):
    loss_sum = th_pred.new_tensor(0.0)
    n_el = 0
    for k, dims in dims_by_class.items():
        m = (y_int == k)
        if m.any():
            a = th_pred[m][:, dims]
            b = th_true[m][:, dims]
            loss_sum = loss_sum + F.smooth_l1_loss(a, b, reduction="sum", beta=beta)
            n_el += a.numel()
    return loss_sum / max(1, n_el)


def norm_theta_classwise(theta, y_int, mu_dict, sd_dict):
    out = np.empty_like(theta, dtype=np.float32)
    for k in range(len(mu_dict)):
        m = (y_int == k)
        out[m] = (theta[m] - mu_dict[k]) / sd_dict[k]
    return out

def main():
    # ============================================================
    #  config
    # ============================================================

    lmax = 5
    maxlen = 2 * lmax * (lmax + 2)
    class_total = 3000
    dims_by_class = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 2]}
    library = {0: "cone", 1: "cuboid", 2: "cylinder"}
    GAP_NM = 200.0

    folders = {
        0: "/scratch/local/nasadova/data_ml/nn/sshfsmountpoint/Cone/Si/",
        1: "/scratch/local/nasadova/tmatrix_data_format/jcmsuite/cuboid_si/results",
        2: "/scratch/local/nasadova/data_ml/data_for_ml/cyl_in_air/new_files/",
    }

    WL_MIN_NM = 700.0
    WL_MAX_NM = 1000.0

    poltype = "parity"
    treams.config.POLTYPE = poltype

    rmax_coef = 3.0
    pols = [0, 1]
    angles = np.array([0.0, np.pi / 6, np.pi / 4])


    # ============================================================
    # Data loading from H5
    # ============================================================

    t_matrices_full = []
    true_labels_full = []
    true_params_full = []
    wls_full = []
    circ_spheres = []
    emb_full = []
    groups = []

    class_counts = {0: 0, 1: 0, 2: 0}

    for cls, folder in folders.items():
        files = sorted(glob(os.path.join(folder, "*.h5")))
        for fp in files:
            if class_counts[cls] >= class_total:
                break

            with h5py.File(fp, "r") as f:
                T = ensure_stackable_T(f["tmatrix"][...], maxlen)
                wl = wl_to_1d_nm(f["vacuum_wavelength"][...], T.shape[0])

                sel = np.arange(T.shape[0])
                mask = (wl >= WL_MIN_NM) & (wl <= WL_MAX_NM)
                sel = sel[mask]
                if sel.size == 0:
                    continue

                remaining = class_total - class_counts[cls]
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
                    a = np.sqrt(rmax ** 2 + (0.5 * h) ** 2)
                elif cls == 1:
                    Lx = float(g["lengthx"][()])
                    Ly = float(g["lengthy"][()])
                    Lz = float(g["lengthz"][()])
                    base_p = [Lx, Ly, Lz]
                    a = 0.5 * np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)
                else:
                    r = float(g["radius"][()])
                    h = float(g["height"][()])
                    base_p = [r, 0.0, h]
                    a = np.sqrt(r ** 2 + (0.5 * h) ** 2)

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
                class_counts[cls] += k

                gkey = geom_key_str(cls, base_p, emb, nd=4)
                groups.extend([gkey] * k)

    circ_spheres = np.array(circ_spheres)
    t_matrices_full = np.asarray(t_matrices_full)
    true_labels_full = np.asarray(true_labels_full, int)
    true_params_full = np.asarray(true_params_full, float)
    wls_full = np.asarray(wls_full, float)
    emb_full = np.array(emb_full)
    groups = np.array(groups)

    uniq, counts = np.unique(groups, return_counts=True)
    min_wls = 5
    count_map = dict(zip(uniq, counts))
    keep = np.array([count_map[g] >= min_wls for g in groups], dtype=bool)

    t_matrices_full = t_matrices_full[keep]
    true_labels_full = true_labels_full[keep]
    true_params_full = true_params_full[keep]
    wls_full = wls_full[keep]
    emb_full = emb_full[keep]
    circ_spheres = circ_spheres[keep]
    groups = groups[keep]

    print("CYLINDERS TOTAL", np.sum(true_labels_full == 2))
    print("CUBOID TOTAL", np.sum(true_labels_full == 1))
    print("CONES TOTAL", np.sum(true_labels_full == 0))

    ps = 2.0 * circ_spheres + GAP_NM
    k0s = 2 * np.pi / wls_full


    # ============================================================
    # Spectra (R) generation via treams + joblib
    # ============================================================

    def create_r(i, j, p_idx):
        k0 = k0s[i]
        kpar = np.array([k0s[i] * np.sin(angles[j]), 0.0])
        eps = treams.Material(emb_full[i])

        tm = treams.TMatrix(
            t_matrices_full[i],
            basis=treams.SphericalWaveBasis.default(lmax),
            k0=k0,
            material=eps,
            poltype=poltype,
        )

        lattice = treams.Lattice.square(ps[i])
        metasurf_t = tm.latticeinteraction.solve(lattice, kpar)

        pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, rmax_coef * k0)
        pol = pols[p_idx]
        illu = treams.plane_wave(kpar, pol, k0=k0, basis=pwb, material=eps, poltype=poltype)

        metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)
        r = metasurf_s.tr(illu)[1]
        return r

    n_pols   = len(pols)
    n_struct = len(t_matrices_full)
    n_angles = len(angles)

    title = (
        f"angles_pol_{pols[0]}_{pols[1]}_{angles[0]}_{angles[-1]}_{len(angles)}"
        f"_gap_{GAP_NM}_rmax_{rmax_coef}_lmax_{lmax}_min_wls_{min_wls}"
    )
    Rs = load_or_compute_Rs(title, t_matrices_full, k0s, emb_full, ps, angles, pols, rmax_coef, poltype, lmax)

    # ============================================================
    # Build dataset (R, lambda, class) -> param
    # ============================================================

    y = np.asarray(true_labels_full, dtype=np.int64)
    groups = np.asarray(groups)
    theta = np.asarray(true_params_full, dtype=np.float32)
    wls_full = np.asarray(wls_full, dtype=np.float32).ravel()

    K = int(y.max()) + 1
    N_samples, N_ang = Rs.shape

    tr_idx, va_idx, te_idx = grouped_train_val_test_stratified(y, groups, test_size=0.15, val_size=0.15)

    print("train", tr_idx.shape)
    print("val ", va_idx.shape)
    print("test ", te_idx.shape)

    mu_lam = wls_full[tr_idx].mean()
    sd_lam = wls_full[tr_idx].std() + 1e-6
    lam_n = (wls_full - mu_lam) / sd_lam

    theta_tr, theta_va, theta_te = theta[tr_idx], theta[va_idx], theta[te_idx]
    lam_tr_n, lam_va_n, lam_te_n = lam_n[tr_idx, None], lam_n[va_idx, None], lam_n[te_idx, None]
    R_tr, R_va, R_te = Rs[tr_idx], Rs[va_idx], Rs[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    device = "cpu"

    mu_th_c, sd_th_c = {}, {}
    for k in range(K):
        m = (y_tr == k)
        mu_th_c[k] = theta_tr[m].mean(axis=0, keepdims=True)
        sd_th_c[k] = theta_tr[m].std(axis=0, keepdims=True) + 1e-6

    th_tr_n = norm_theta_classwise(theta_tr, y_tr, mu_th_c, sd_th_c)
    th_va_n = norm_theta_classwise(theta_va, y_va, mu_th_c, sd_th_c)
    th_te_n = norm_theta_classwise(theta_te, y_te, mu_th_c, sd_th_c)

    ytr_oh = F.one_hot(torch.from_numpy(y_tr).long(), num_classes=K).float()
    yva_oh = F.one_hot(torch.from_numpy(y_va).long(), num_classes=K).float()
    yte_oh = F.one_hot(torch.from_numpy(y_te).long(), num_classes=K).float()

    th_tr_t = torch.from_numpy(th_tr_n).float()
    th_va_t = torch.from_numpy(th_va_n).float()
    th_te_t = torch.from_numpy(th_te_n).float()

    lam_tr_t = torch.from_numpy(lam_tr_n).float()
    lam_va_t = torch.from_numpy(lam_va_n).float()
    lam_te_t = torch.from_numpy(lam_te_n).float()

    R_tr_t = torch.from_numpy(R_tr).float()
    R_va_t = torch.from_numpy(R_va).float()
    R_te_t = torch.from_numpy(R_te).float()

    ds_tr_inv = TensorDataset(R_tr_t, lam_tr_t, ytr_oh, th_tr_t)
    ds_va_inv = TensorDataset(R_va_t, lam_va_t, yva_oh, th_va_t)
    ds_te_inv = TensorDataset(R_te_t, lam_te_t, yte_oh, th_te_t)

    g = torch.Generator().manual_seed(SEED)

    dl_tr_inv = DataLoader(ds_tr_inv, batch_size=512, shuffle=True, generator=g)
    dl_va_inv = DataLoader(ds_va_inv, batch_size=1024, shuffle=False, generator=g)
    dl_te_inv = DataLoader(ds_te_inv, batch_size=1024, shuffle=False, generator=g)


    # ============================================================
    # Train inverse model
    # ============================================================

    hist_inv = {"ep": [], "train_loss": [], "val_loss": []}
    best_val_inv = float("inf")
    best_state_inv = None

    inv_model = R2ThetaResNetClassHead(
        n_angles=n_angles * n_pols,
        y_dim=K,
        theta_dim=theta.shape[1],
        width=1024,
        num_blocks=5,
    )

    opt_inv = torch.optim.AdamW(inv_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_inv = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_inv, mode="min", factor=0.5, patience=5)

    epochs = 150
    for ep in range(1, epochs):
        inv_model.train()
        tr_loss = 0.0
        n = 0

        for Rb, lb, yb, thb in dl_tr_inv:
            Rb, lb, yb, thb = Rb.to(device), lb.to(device), yb.to(device), thb.to(device)

            th_pred = inv_model(Rb, lb, yb)
            y_int = yb.argmax(1)
            loss = masked_huber(th_pred, thb, y_int, dims_by_class)

            opt_inv.zero_grad()
            loss.backward()
            opt_inv.step()

            bs = thb.size(0)
            tr_loss += loss.item() * bs
            n += bs

        tr_loss /= max(n, 1)

        inv_model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for Rb, lb, yb, thb in dl_va_inv:
                Rb, lb, yb, thb = Rb.to(device), lb.to(device), yb.to(device), thb.to(device)
                th_pred = inv_model(Rb, lb, yb)
                y_int = yb.argmax(1)
                loss = masked_huber(th_pred, thb, y_int, dims_by_class)

                va_loss += loss.item() * thb.size(0)
                vn += thb.size(0)

        va_loss /= max(vn, 1)
        sched_inv.step(va_loss)

        if va_loss < best_val_inv:
            best_val_inv = va_loss
            best_state_inv = {k: v.cpu().clone() for k, v in inv_model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"[INV] ep{ep:03d}  train={tr_loss:.4f}  val={va_loss:.4f}")

        hist_inv["ep"].append(ep)
        hist_inv["train_loss"].append(tr_loss)
        hist_inv["val_loss"].append(va_loss)

    inv_model.load_state_dict(best_state_inv)
    inv_model.to(device).eval()
    
    plot_curves(hist_inv, keys=("loss",), save_path="figs/rtoparam_curves")
    plot_per_class_mae(hist_inv, split="tr", save_prefix="rtoparam_tr")
    plot_per_class_mae(hist_inv, split="va", save_prefix="rtoparam_va")


    # ============================================================
    # Test evaluation + unnormalize param
    # ============================================================

    inv_model.eval()
    theta_pred_n_list = []
    theta_true_n_list = []
    y_te_list = []

    with torch.no_grad():
        for Rb, lb, yb, thb in dl_te_inv:
            Rb, lb, yb = Rb.to(device), lb.to(device), yb.to(device)
            th_pred = inv_model(Rb, lb, yb).cpu().numpy()
            theta_pred_n_list.append(th_pred)
            theta_true_n_list.append(thb.numpy())
            y_te_list.append(yb.cpu().numpy())

    theta_pred_n = np.concatenate(theta_pred_n_list, axis=0)
    theta_true_n = np.concatenate(theta_true_n_list, axis=0)
    y_te_int = np.concatenate(y_te_list, axis=0).argmax(1)

    mae_params = np.mean(np.abs(theta_pred_n - theta_true_n), axis=0)
    print("MAE per param in normalized space:", mae_params)

    theta_pred = np.empty_like(theta_true_n, dtype=np.float32)
    for k in range(K):
        m = (y_te_int == k)
        if not np.any(m):
            continue
        theta_pred[m] = theta_pred_n[m] * sd_th_c[k] + mu_th_c[k]

    theta_true = np.empty_like(theta_true_n, dtype=np.float32)
    for k in range(K):
        m = (y_te_int == k)
        if not np.any(m):
            continue
        theta_true[m] = theta_true_n[m] * sd_th_c[k] + mu_th_c[k]


    # ============================================================
    # Plots / outputs
    # ============================================================

    class_names = ["cone", "cuboid", "cylinder"]

    err_te = np.zeros_like(y_te_int, dtype=float)
    for c in np.unique(y_te_int):
        m = (y_te_int == c)
        dims = dims_by_class[int(c)]
        err_te[m] = np.mean(np.abs(theta_pred[m][:, dims] - theta_true[m][:, dims]), axis=1)

    wls_te = wls_full[te_idx]
    theta_true_nm = theta_true
    theta_pred_nm = theta_pred

    idx = select_n_per_class(y_te_int, n_per_class=300, seed=0, replace_if_needed=False)

    parity_spec = {
        0: [(0, "top radius"), (2, "height")],
        1: [(0, "length"), (2, "height")],
        2: [(0, "radius"), (2, "height")],
    }


    save_abcd_figdata_h5(
        f"results/rtoparam_results.h5",
        theta_true=theta_true_nm[idx],
        theta_pred=theta_pred_nm[idx],
        y_int=y_te_int[idx],
        class_names=class_names,
        parity_spec=parity_spec,
        dims_by_class=dims_by_class
    )

if __name__ == "__main__":
    main()