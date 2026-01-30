#!/usr/bin/env python3
"""
Train + evaluate inverse model

"""

import os
from glob import glob

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt

from helper import  (
    ensure_stackable_T, wl_to_1d_nm,
    geom_key_str, grouped_train_val_test_stratified, 
    set_seed, plot_curves
)

# ============================================================
# Reproducibility 
# ============================================================
SEED = 777
set_seed(SEED)


# -------------------------
# Models
# -------------------------

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


class T2ThetaResNetMT(nn.Module):
    """
    MLP: (T_flat(2*N*N), optional lambda) -> class logits + theta heads per class
    """
    def __init__(
        self,
        T_dim: int,              # 2*N*N
        d_per_class,             #  [3,3,3] or [3,3,2]
        use_lambda: bool = True,
        width: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_per_class = list(map(int, d_per_class))
        self.K = len(self.d_per_class)
        self.use_lambda = bool(use_lambda)
        in_dim = T_dim + (1 if self.use_lambda else 0)
        self.in_layer = nn.Linear(in_dim, width)
        # self.blocks = nn.Sequential(*[ResBlock(width, hidden=2*width, p=dropout) for _ in range(num_blocks)])
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.cls_head = nn.Linear(width, self.K)
        self.param_heads = nn.ModuleList([nn.Linear(width, d) for d in self.d_per_class])

    def forward(self, T_in, lam_n=None):
        if T_in.dim() == 4:
            B = T_in.size(0)
            T_flat = T_in.reshape(B, -1)   # [B, 2*N*N]
        elif T_in.dim() == 2:
            T_flat = T_in
        if self.use_lambda:
            if lam_n is None:
                raise ValueError("lam_n is required when use_lambda=True")
            if lam_n.dim() == 1:
                lam_n = lam_n[:, None]
            x = torch.cat([T_flat, lam_n], dim=1)
        else:
            x = T_flat
        h = F.gelu(self.in_layer(x))
        h = self.blocks(h)
        logits = self.cls_head(h)
        params_list = [head(h) for head in self.param_heads]
        return logits, params_list

# -------------------------
# Data utils
# -------------------------

def w_reg_at(epoch, warm=0, ramp=25, max_w=0.5):
    if epoch < warm:
        return 0.0
    t = min(1.0, (epoch - warm) / ramp)
    return max_w * 0.5 * (1 - np.cos(np.pi * t))



def build_theta_stats(params_tr, labels_tr, d_per_class, device):
    Kc = len(d_per_class)
    D_max = int(params_tr.size(1))

    theta_mean = torch.zeros(Kc, D_max, dtype=torch.float32, device=device)
    theta_std = torch.ones(Kc, D_max, dtype=torch.float32, device=device)

    for k in range(Kc):
        m = labels_tr == k
        if m.any():
            d_k = int(d_per_class[k])
            mu = params_tr[m, :d_k].mean(0)
            sd = params_tr[m, :d_k].std(0).clamp_min(1e-6)
            theta_mean[k, :d_k] = mu
            theta_std[k, :d_k] = sd

    theta_mean.requires_grad_(False)
    theta_std.requires_grad_(False)
    return theta_mean, theta_std


def norm_theta_batch(theta_packed, k_ids, theta_mean, theta_std):
    mu = theta_mean[k_ids]
    sd = theta_std[k_ids]
    return (theta_packed - mu) / sd


# -------------------------
# Evaluation dump + H5 
# -------------------------

@torch.no_grad()
def collect_eval_dump_balanced(inverse, loader, d_per_class, theta_mean, theta_std, device, n_per_class=300):
    inverse.eval()
    Kc = len(d_per_class)
    counts = np.zeros(Kc, dtype=int)

    K_true, K_pred = [], []
    Ys_raw, Yh_raw, Mask = [], [], []

    for x_b, k_b, th_b, lam_b in loader:
        keep = torch.zeros_like(k_b, dtype=torch.bool)
        for c in range(Kc):
            need = n_per_class - counts[c]
            if need <= 0:
                continue
            idx = (k_b == c).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            gen = torch.Generator(device=idx.device)
            gen.manual_seed(SEED)
            idx = idx[torch.randperm(idx.numel(), generator=gen, device=idx.device)]
            take = idx[:need]
            keep[take] = True
            counts[c] += int(take.numel())

        if not keep.any():
            if np.all(counts >= n_per_class):
                break
            continue

        x_b = x_b[keep].to(device, dtype=torch.float32)
        k_b = k_b[keep].to(device)
        th_b = th_b[keep].to(device, dtype=torch.float32)
        lam_b = lam_b[keep].to(device, dtype=torch.float32)

        logits, params_list = inverse(x_b, lam_b)
        k_hat = logits.argmax(-1)

        th_n = (th_b - theta_mean[k_b]) / theta_std[k_b]

        pred_n = torch.zeros_like(th_b, dtype=torch.float32)
        mask = torch.zeros_like(th_b, dtype=torch.bool)
        for k, d_k in enumerate(d_per_class):
            m = k_b == k
            if m.any():
                pred_n[m, :d_k] = params_list[k][m]
                mask[m, :d_k] = True

        th_raw = th_n * theta_std[k_b] + theta_mean[k_b]
        ph_raw = pred_n * theta_std[k_b] + theta_mean[k_b]

        K_true.append(k_b.cpu().numpy())
        K_pred.append(k_hat.cpu().numpy())
        Ys_raw.append(th_raw.cpu().numpy())
        Yh_raw.append(ph_raw.cpu().numpy())
        Mask.append(mask.cpu().numpy())

        if np.all(counts >= n_per_class):
            break

    dump = {
        "k_true": np.concatenate(K_true),
        "k_pred": np.concatenate(K_pred),
        "y_true": np.concatenate(Ys_raw),
        "y_pred": np.concatenate(Yh_raw),
        "mask": np.concatenate(Mask).astype(bool),
    }
    print("EVAL N:", len(dump["k_true"]), "counts:", np.bincount(dump["k_true"], minlength=Kc))
    return dump


def save_eval_h5(path, dump, class_names, parity_spec, *, delta=1.0):

    req = ["k_true", "k_pred", "y_true", "y_pred", "mask"]
    miss = [k for k in req if k not in dump]
    if miss:
        raise KeyError(f"dump missing keys: {miss}")

    str_dt = h5py.string_dtype(encoding="utf-8")
    path = str(path)

    with h5py.File(path, "w") as f:
        g = f.create_group("dump")
        for k in req:
            arr = np.asarray(dump[k])
            if k == "mask":
                arr = arr.astype(np.uint8)
            g.create_dataset(k, data=arr, compression="gzip", compression_opts=4, shuffle=True)

        m = f.create_group("meta")
        m.create_dataset("delta", data=float(delta))
        m.create_dataset("class_names", data=np.asarray(list(class_names), dtype=object), dtype=str_dt)

        gp = m.create_group("parity_spec")
        for cls, spec in parity_spec.items():
            cls = int(cls)
            dims = np.asarray([int(d) for d, _ in spec], dtype=np.int32)
            names = np.asarray([str(n) for _, n in spec], dtype=object)
            gc = gp.create_group(f"class_{cls}")
            gc.create_dataset("dims", data=dims)
            gc.create_dataset("names", data=names, dtype=str_dt)



# -------------------------
# Main
# -------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- config ----
    lmax = 5
    maxlen = 2 * lmax * (lmax + 2)
    class_total = 3000
    min_wls = 5

    WL_MIN_NM = 700.0
    WL_MAX_NM = 1000.0

    folders = {
        0: "cones/",
        1: "cuboids/",
        2: "cylinders/",
    } # MODIFY PATH

    out_h5 = "results/ttoparam_results.h5"

    d_per_class = [3, 3, 3]
    inv_epochs = 100
    lr = 3e-4
    w_ce = 1.0

    parity_spec = {
        0: [(0, "top radius"), (2, "height")],
        1: [(0, "length"), (2, "height")],
        2: [(0, "radius"), (2, "height")],
    }
    class_names = ("cone", "cuboid", "cylinder")

    # ---- load T-matrices into arrays ----
    t_matrices_full = []
    true_labels_full = []
    true_params_full = []
    wls_full = []
    emb_full = []
    circ_spheres = []
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

                mask = (wl >= WL_MIN_NM) & (wl <= WL_MAX_NM)
                sel = np.arange(T.shape[0])[mask]
                if sel.size == 0:
                    continue

                remaining = class_total - class_counts[cls]
                if remaining <= 0:
                    continue
                sel = sel[:remaining]
                if sel.size == 0:
                    continue

                g = f["scatterer/geometry"]
                if cls == 0:
                    rtop = float(g["radius_top"][()])
                    rbot = float(g["radius_bottom"][()])
                    h = float(g["height"][()])
                    base_p = [rtop, rbot, h]
                    rmax = max(rtop, rbot)
                    a = np.sqrt(rmax**2 + (0.5 * h) ** 2)
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
                    a = np.sqrt(r**2 + (0.5 * h) ** 2)

                emb = f["embedding/relative_permittivity"][...]

                T_sel = T[sel]
                wl_sel = wl[sel]
                k = int(sel.size)
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

    circ_spheres = np.asarray(circ_spheres)
    t_matrices_full = np.asarray(t_matrices_full)
    true_labels_full = np.asarray(true_labels_full, int)
    true_params_full = np.asarray(true_params_full, float)
    wls_full = np.asarray(wls_full, float)
    groups = np.asarray(groups)

    uniq, cnt = np.unique(groups, return_counts=True)
    count_map = dict(zip(uniq, cnt))
    keep = np.array([count_map[g] >= min_wls for g in groups], dtype=bool)
    t_matrices_full = t_matrices_full[keep]
    labels = torch.from_numpy(true_labels_full[keep]).long()
    params = torch.from_numpy(true_params_full[keep]).float()
    wls = torch.from_numpy(wls_full[keep]).float()
    groups = groups[keep]

    print("Data:", t_matrices_full.shape, "labels:", labels.shape, "params:", params.shape)

    # ----X_stack\; [B,2,N,N] ----
    B, N, _ = t_matrices_full.shape
    X_real = t_matrices_full.real
    X_imag = t_matrices_full.imag
    X_stack = torch.from_numpy(np.stack([X_real, X_imag], axis=1)).float()

    # ---- split by geometry ----
    tr_idx, va_idx, te_idx = grouped_train_val_test_stratified(labels.numpy(), groups, test_size=0.15, val_size=0.15)

    # ---- normalize X ---
    mean = X_stack[tr_idx].mean(dim=(0, 2, 3), keepdim=True)
    std = X_stack[tr_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    X_norm = (X_stack - mean) / std

    # ---- normalize lambda ----
    lam_mu = wls[tr_idx].mean()
    lam_std = wls[tr_idx].std().clamp_min(1e-6)
    lam_n = (wls - lam_mu) / lam_std

    ds_sup = TensorDataset(X_norm, labels, params, lam_n)
    ds_train = Subset(ds_sup, tr_idx.tolist())
    ds_val = Subset(ds_sup, va_idx.tolist())
    ds_test = Subset(ds_sup, te_idx.tolist())

    labels_tr = labels[tr_idx]
    # sampler = WeightedRandomSampler(
    #     weights=(1.0 / counts.clamp_min(1))[labels_tr],
    #     num_samples=int(labels_tr.numel()),
    #     replacement=True,
    # )

    g = torch.Generator().manual_seed(SEED)


    loader_tr = DataLoader(ds_train, batch_size=128, shuffle=True, generator=g)
    loader_va = DataLoader(ds_val, batch_size=128, shuffle=False, generator=g)
    loader_te = DataLoader(ds_test, batch_size=128, shuffle=False, generator=g)

    # ---- theta stats (classwise) ----
    theta_mean, theta_std = build_theta_stats(params[tr_idx].to(device), labels[tr_idx].to(device), d_per_class, device)

    B, N, _ = t_matrices_full.shape
    T_dim = 2 * N * N

    # ---- model ----
    inverse = T2ThetaResNetMT(
        T_dim=T_dim,
        d_per_class=d_per_class,
        use_lambda=True,
        width=1024,
        num_blocks=4,
        dropout=0.,
    ).to(device)
    opt_inv = torch.optim.Adam(inverse.parameters(), lr=lr)
    n_params = sum(p.numel() for p in inverse.parameters() if p.requires_grad)
    print(n_params)
    history = {
        "tr_loss": [], "va_loss": [],
        "tr_cls": [], "tr_reg": [],
        "va_cls": [], "va_reg": [],
        "va_acc": []
    }

    best_va = float("inf")
    best_state = None

    for epoch in range(inv_epochs):
        inverse.train()
        tr_loss_sum = tr_cls_sum = tr_reg_sum = 0.0
        tr_count = 0

        for x_b, k_b, th_b, lam_b in loader_tr:
            x_b = x_b.to(device, dtype=torch.float32)
            k_b = k_b.to(device)
            th_b = th_b.to(device, dtype=torch.float32)
            lam_b = lam_b.to(device, dtype=torch.float32)

            logits, params_list = inverse(x_b, lam_b)
            loss_cls = F.cross_entropy(logits, k_b)

            th_n = norm_theta_batch(th_b, k_b, theta_mean, theta_std)

            pred_n = torch.zeros_like(th_b, dtype=torch.float32)
            mask = torch.zeros_like(th_b, dtype=torch.bool)
            for k, d_k in enumerate(d_per_class):
                m = k_b == k
                if m.any():
                    pred_n[m, :d_k] = params_list[k][m]
                    mask[m, :d_k] = True

            diff = pred_n[mask] - th_n[mask]
            delta = 1.0
            absd = diff.abs()
            hub = torch.where(absd <= delta, 0.5 * diff**2, delta * (absd - 0.5 * delta))
            loss_reg = hub.mean()

            w_reg = float(w_reg_at(epoch))
            loss = w_ce * loss_cls + w_reg * loss_reg

            opt_inv.zero_grad()
            loss.backward()
            opt_inv.step()

            bs = int(k_b.size(0))
            tr_loss_sum += float(loss.item()) * bs
            tr_cls_sum += float(loss_cls.item()) * bs
            tr_reg_sum += float(loss_reg.item()) * bs
            tr_count += bs

        tr_loss = tr_loss_sum / max(1, tr_count)
        tr_cls = tr_cls_sum / max(1, tr_count)
        tr_reg = tr_reg_sum / max(1, tr_count)

        inverse.eval()
        va_loss_sum = va_cls_sum = va_reg_sum = 0.0
        va_count = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            for x_b, k_b, th_b, lam_b in loader_va:
                x_b = x_b.to(device, dtype=torch.float32)
                k_b = k_b.to(device)
                th_b = th_b.to(device, dtype=torch.float32)
                lam_b = lam_b.to(device, dtype=torch.float32)

                logits, params_list = inverse(x_b, lam_b)
                loss_cls = F.cross_entropy(logits, k_b)

                th_n = norm_theta_batch(th_b, k_b, theta_mean, theta_std)

                pred_n = torch.zeros_like(th_b, dtype=torch.float32)
                mask = torch.zeros_like(th_b, dtype=torch.bool)
                for k, d_k in enumerate(d_per_class):
                    m = k_b == k
                    if m.any():
                        pred_n[m, :d_k] = params_list[k][m]
                        mask[m, :d_k] = True

                diff = pred_n[mask] - th_n[mask]
                delta = 1.0
                absd = diff.abs()
                hub = torch.where(absd <= delta, 0.5 * diff**2, delta * (absd - 0.5 * delta))
                loss_reg = hub.mean()

                w_reg = float(w_reg_at(epoch))
                loss = w_ce * loss_cls + w_reg * loss_reg

                bs = int(k_b.size(0))
                va_loss_sum += float(loss.item()) * bs
                va_cls_sum += float(loss_cls.item()) * bs
                va_reg_sum += float(loss_reg.item()) * bs
                va_count += bs

                all_true.append(k_b.cpu())
                all_pred.append(logits.softmax(-1).argmax(-1).cpu())

        va_loss = va_loss_sum / max(1, va_count)
        va_cls = va_cls_sum / max(1, va_count)
        va_reg = va_reg_sum / max(1, va_count)
        y_true = torch.cat(all_true).numpy()
        y_pred = torch.cat(all_pred).numpy()
        va_acc = float((y_true == y_pred).mean())

        history["tr_loss"].append(tr_loss); history["tr_cls"].append(tr_cls); history["tr_reg"].append(tr_reg)
        history["va_loss"].append(va_loss); history["va_cls"].append(va_cls); history["va_reg"].append(va_reg)
        history["va_acc"].append(va_acc); 

        print(
            f"Epoch {epoch:03d} | tr {tr_loss:.4f} (cls {tr_cls:.2f}, reg {tr_reg:.4f})"
            f" | va {va_loss:.4f} (cls {va_cls:.2f}, reg {va_reg:.4f})"
            f" | acc {va_acc:.2f}"
        )

        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in inverse.state_dict().items()}

    if best_state is not None:
        inverse.load_state_dict(best_state)

    plot_curves(history, keys=["loss", "cls", "reg"],
                title="inverse training curves",
                save_path="figs/ttoparam_mlp_curves.png")

    # -- balanced dump on test set ----
    dump = collect_eval_dump_balanced(
        inverse, loader_te, d_per_class, theta_mean, theta_std, device, n_per_class=300
    )

    save_eval_h5(
        out_h5,
        dump,
        class_names=class_names,
        parity_spec=parity_spec,
        delta=1.0,

    )
    print("saved:", out_h5)


if __name__ == "__main__":
    main()
