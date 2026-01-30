
# ============================================================
# 0) Imports
# ============================================================
import os, random
from glob import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import treams
from helper import  (
    ensure_stackable_T, wl_to_1d_nm, geom_key_str,
    grouped_train_val_test_stratified, load_or_compute_Rs,
    Rvec_from_h5_tr, plot_curves, plot_per_class_mae
)

# ============================================================
# 1) Reproducibility
# ============================================================
def set_seed(seed=123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# ============================================================
# 2) Plotting utils
# ============================================================

def save_paper_fig_h5(
    path,
    err_sample,
    y_te,
    te_gid,
    wl_te,
    R_true_te,
    R_pred_te,
    R_ref,
    wl_ref,
    angles,
    class_names,
    gid_spec,
    ang_ids=(0,1,2),
    pol_ids=(0,1),
):
    str_dt = h5py.string_dtype("utf-8")

    err_sample = np.asarray(err_sample, dtype=np.float32)
    y_te       = np.asarray(y_te, dtype=np.int64)
    wl_te      = np.asarray(wl_te, dtype=np.float32)
    R_true_te  = np.asarray(R_true_te, dtype=np.float32)
    R_pred_te  = np.asarray(R_pred_te, dtype=np.float32)
    angles     = np.asarray(angles, dtype=np.float32)
    ang_ids    = np.asarray(ang_ids, dtype=np.int32)
    pol_ids    = np.asarray(pol_ids, dtype=np.int32)

    te_gid = np.asarray(te_gid)
    te_gid_str = np.array([str(x) for x in te_gid], dtype=object)

    if isinstance(class_names, dict):
        cn = [str(v) for v in class_names.values()]
    else:
        cn = [str(v) for v in class_names]
    class_names_arr = np.array(cn, dtype=object)

    with h5py.File(path, "w") as f:
        f.create_dataset("err_sample", data=err_sample, compression="gzip", compression_opts=4)
        f.create_dataset("y_te",       data=y_te,       compression="gzip", compression_opts=4)
        f.create_dataset("te_gid",     data=te_gid_str, dtype=str_dt)

        f.create_dataset("wl_te",      data=wl_te,      compression="gzip", compression_opts=4)
        f.create_dataset("R_true_te",  data=R_true_te,  compression="gzip", compression_opts=4)
        f.create_dataset("R_pred_te",  data=R_pred_te,  compression="gzip", compression_opts=4)
        f.create_dataset("R_ref",      data=R_ref,      compression="gzip", compression_opts=4)
        f.create_dataset("wl_ref",     data=wl_ref,     compression="gzip", compression_opts=4)

        f.create_dataset("angles",     data=angles)
        f.create_dataset("ang_ids",    data=ang_ids)
        f.create_dataset("pol_ids",    data=pol_ids)

        f.create_dataset("class_names", data=class_names_arr, dtype=str_dt)
        f.create_dataset("gid_spec",    data=np.array([str(gid_spec)], dtype=object), dtype=str_dt)



# ============================================================
# 5) Model + evaluation helpers
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
            s[0] += float(e); s[1] += 1

    out = {}
    for c, (s, n) in sums.items():
        out[c] = (s / max(1, n), n)
    return out

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        return x + h

class Theta2RResNet(nn.Module):
    def __init__(self, theta_dim: int = 3, y_dim: int = 3,
                 out_dim: int = 3,
                 width: int = 512, num_blocks: int = 4):
        super().__init__()
        in_dim = theta_dim + 1 + y_dim
        self.in_layer = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(num_blocks)])
        self.out_layer = nn.Linear(width, out_dim)

    def forward(self, theta_n: torch.Tensor, lambda_n: torch.Tensor, y_oh: torch.Tensor) -> torch.Tensor:
        x = torch.cat([theta_n, lambda_n, y_oh], dim=1)
        h = F.gelu(self.in_layer(x))
        h = self.blocks(h)
        logits = self.out_layer(h)
        return torch.sigmoid(logits)



# ============================================================
# 7) Main 
# ============================================================
def main():
    # ---------------- seed ----------------
    SEED = 777
    set_seed(SEED)

    # ---------------- config ----------------
    lmax = 5
    maxlen = 2*lmax*(lmax+2)
    class_total = 3000
    library = {0:"cone", 1:"cuboid", 2:"cylinder"}
    GAP_NM   = 200.0

    folders = {
        0: "cones/",
        1: "cuboids/",
        2: "cylinders/",
    } # MODIFY PATH

    os.makedirs("figs/", exist_ok=True)

    # ---------------- load dataset ----------------
    t_matrices_full  = []
    true_labels_full = []
    true_params_full = []
    wls_full         = []
    counts = {0:0, 1:0, 2:0}
    circ_spheres = []
    emb_full = []
    groups = []
    WL_MIN_NM = 700.0
    WL_MAX_NM = 1000.0
    for cls, folder in folders.items():
        files = sorted(glob(os.path.join(folder, "*.h5")))
        for fp in files:
            if counts[cls] >= class_total:
                break
            with h5py.File(fp, "r") as f:
                T  = ensure_stackable_T(f["tmatrix"][...], maxlen)
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
                    rtop  = float(g["radius_top"][()])
                    rbot  = float(g["radius_bottom"][()])
                    h     = float(g["height"][()])
                    base_p = [rtop, rbot, h]
                    rmax   = max(rtop, rbot)
                    a      = np.sqrt(rmax**2 + (0.5*h)**2)
                elif cls == 1:
                    Lx = float(g["lengthx"][()])
                    Ly = float(g["lengthy"][()])
                    Lz = float(g["lengthz"][()])
                    base_p = [Lx, Ly, Lz]
                    a      = 0.5 * np.sqrt(Lx**2 + Ly**2 + Lz**2)
                else:
                    r = float(g["radius"][()])
                    h = float(g["height"][()])
                    base_p = [r, 0., h]
                    a      = np.sqrt(r**2 + (0.5*h)**2)

                emb = f["embedding/relative_permittivity"][...]
                T_sel  = T[sel]
                wl_sel = wl[sel]
                k = sel.size
                P_sel  = np.repeat(np.array(base_p, float)[None, :], k, axis=0)

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
    emb_full = np.array(emb_full)
    wls_full = np.array(wls_full)
    groups = np.array(groups)
    t_matrices_full = np.array(t_matrices_full)
    true_labels_full = np.array(true_labels_full)
    true_params_full = np.array(true_params_full)

    # ---------------- physics setup ----------------
    rmax_coef = 3.
    pols = [0, 1]
    poltype = "parity"
    treams.config.POLTYPE = poltype

    # ---------------- filter geometries with >= min_wl samples ----------------
    uniq, counts_u = np.unique(groups, return_counts=True)
    count_map = dict(zip(uniq, counts_u))
    min_wls = 5
    keep = np.array([count_map[g] >= min_wls for g in groups], dtype=bool)

    t_matrices_full  = t_matrices_full[keep]
    true_labels_full = true_labels_full[keep]
    true_params_full = true_params_full[keep]
    wls_full         = wls_full[keep]
    emb_full         = emb_full[keep]
    circ_spheres     = circ_spheres[keep]
    groups           = groups[keep]

    ps = 2.0*circ_spheres + GAP_NM
    k0s = 2*np.pi/wls_full

    y       = np.asarray(true_labels_full, dtype=np.int64)
    theta   = np.asarray(true_params_full, dtype=np.float32)

    # ---------------- compute Rs in parallel ----------------
    angles = np.array([0., np.pi/6, np.pi/4])
    n_pols   = len(pols)
    n_struct = len(t_matrices_full)
    n_angles = len(angles)
    n_steps = 150

    title = (
        f"angles_pol_{pols[0]}_{pols[1]}_{angles[0]}_{angles[-1]}_{len(angles)}"
        f"_gap_{GAP_NM}_rmax_{rmax_coef}_lmax_{lmax}_min_wls_{min_wls}"
    )

    Rs = load_or_compute_Rs(title, t_matrices_full, k0s, emb_full, ps, angles, pols, rmax_coef, poltype, lmax)

    # ---------------- reference reflectance ----------------
    reffile = "cylinder_si_r_110.0_h_190.0_l_5_wls_7.000000000000001e-07_1.0000000000000002e-06_61_msl_2_3_domain_500_500.tmat.h5" #MODIFY PATH
    R_ref, wl_ref = Rvec_from_h5_tr(reffile, angles=angles, GAP_NM=GAP_NM, rmax_coef=rmax_coef)

    # ---------------- split ----------------
    tr_idx, va_idx, te_idx = grouped_train_val_test_stratified(
        y, groups, test_size=0.15, val_size=0.15
    )

    te_gid = groups[te_idx]

    # ---------------- wavelength normalization ----------------
    mu_lam = wls_full[tr_idx].mean()
    sd_lam = wls_full[tr_idx].std() + 1e-6
    lam_n  = (wls_full - mu_lam) / sd_lam

    # ---------------- split arrays ----------------
    theta_tr, theta_va, theta_te = theta[tr_idx], theta[va_idx], theta[te_idx]
    lam_tr_n, lam_va_n, lam_te_n = lam_n[tr_idx,None], lam_n[va_idx,None], lam_n[te_idx,None]
    R_tr, R_va, R_te = Rs[tr_idx], Rs[va_idx], Rs[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # ---------------- classwise theta normalization ----------------
    K = int(y.max()) + 1
    mu_th_c, sd_th_c = {}, {}
    for k in range(K):
        m = (y_tr == k)
        mu_th_c[k] = theta_tr[m].mean(axis=0, keepdims=True)
        sd_th_c[k] = theta_tr[m].std(axis=0, keepdims=True) + 1e-6

    def norm_theta_classwise(theta_arr, y_int, mu_dict, sd_dict):
        out = np.empty_like(theta_arr, dtype=np.float32)
        for kk in range(len(mu_dict)):
            m = (y_int == kk)
            out[m] = (theta_arr[m] - mu_dict[kk]) / sd_dict[kk]
        return out

    th_tr_n = norm_theta_classwise(theta_tr, y_tr, mu_th_c, sd_th_c)
    th_va_n = norm_theta_classwise(theta_va, y_va, mu_th_c, sd_th_c)
    th_te_n = norm_theta_classwise(theta_te, y_te, mu_th_c, sd_th_c)

    # ---------------- torch datasets/loaders ----------------
    ytr_oh = F.one_hot(torch.from_numpy(y_tr).long(), num_classes=K).float()
    yva_oh = F.one_hot(torch.from_numpy(y_va).long(), num_classes=K).float()
    yte_oh = F.one_hot(torch.from_numpy(y_te).long(), num_classes=K).float()

    th_tr_t  = torch.from_numpy(th_tr_n).float()
    th_va_t  = torch.from_numpy(th_va_n).float()
    th_te_t  = torch.from_numpy(th_te_n).float()
    lam_tr_t = torch.from_numpy(lam_tr_n).float()
    lam_va_t = torch.from_numpy(lam_va_n).float()
    lam_te_t = torch.from_numpy(lam_te_n).float()
    R_tr_t   = torch.from_numpy(R_tr).float()
    R_va_t   = torch.from_numpy(R_va).float()
    R_te_t   = torch.from_numpy(R_te).float()

    ds_tr = TensorDataset(th_tr_t, lam_tr_t, ytr_oh, R_tr_t)
    ds_va = TensorDataset(th_va_t, lam_va_t, yva_oh, R_va_t)
    ds_te = TensorDataset(th_te_t, lam_te_t, yte_oh, R_te_t)

    g = torch.Generator().manual_seed(SEED)
    dl_tr = DataLoader(ds_tr, batch_size=512, shuffle=True, generator=g)
    dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, generator=g)
    dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False, generator=g)

    # ---------------- model ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width = 1024
    model = Theta2RResNet(
        theta_dim=theta.shape[1],
        y_dim=K,
        out_dim=R_tr.shape[1],
        width=width,
        num_blocks=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters (DoF):", n_params)

    # ---------------- training ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    best_state = None

    hist = {
        "ep": [], "train_loss": [], "val_loss": [],
        "mae_tr_cone": [], "mae_tr_cuboid": [], "mae_tr_cylinder": [],
        "mae_va_cone": [], "mae_va_cuboid": [], "mae_va_cylinder": [],
    }

    for ep in range(n_steps):
        model.train()
        tr_loss = n = 0
        for thb, lb, yb, Rb in dl_tr:
            thb, lb, yb, Rb = thb.to(device), lb.to(device), yb.to(device), Rb.to(device)
            pred = model(thb, lb, yb)
            loss = F.mse_loss(pred, Rb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = Rb.size(0)
            tr_loss += loss.item() * bs
            n += bs
        tr_loss /= max(n, 1)

        model.eval()
        va_loss = vn = 0
        with torch.no_grad():
            for thb, lb, yb, Rb in dl_va:
                thb, lb, yb, Rb = thb.to(device), lb.to(device), yb.to(device), Rb.to(device)
                pred = model(thb, lb, yb)
                loss = F.mse_loss(pred, Rb)
                va_loss += loss.item() * Rb.size(0)
                vn += Rb.size(0)
        va_loss /= max(vn, 1)
        sched.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        mae_tr = per_class_mae(model, dl_tr, device=device)
        mae_va = per_class_mae(model, dl_va, device=device)

        hist["ep"].append(ep)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["mae_tr_cone"].append(mae_tr.get(0, (float("nan"),0))[0])
        hist["mae_tr_cuboid"].append(mae_tr.get(1, (float("nan"),0))[0])
        hist["mae_tr_cylinder"].append(mae_tr.get(2, (float("nan"),0))[0])
        hist["mae_va_cone"].append(mae_va.get(0, (float("nan"),0))[0])
        hist["mae_va_cuboid"].append(mae_va.get(1, (float("nan"),0))[0])
        hist["mae_va_cylinder"].append(mae_va.get(2, (float("nan"),0))[0])

    # ---------------- restore best ----------------
    model.load_state_dict(best_state)
    model.to(device).eval()  # runtime fix: actually call eval()

    ckpt = {
        "model_state": model.state_dict(),
        "cfg": dict(
            arch="Theta2RResNet",
            theta_dim=theta.shape[1],
            y_dim=K,
            n_angles=n_angles,
            n_pols=n_pols,
            width=width,
            num_blocks=4,
        ),
        "norm": dict(
            mu_lam=float(mu_lam),
            sd_lam=float(sd_lam),
            mu_th_c={int(k): mu_th_c[k] for k in mu_th_c},
            sd_th_c={int(k): sd_th_c[k] for k in sd_th_c},
            theta_scaled_by_wavelength=False,
        ),
        "meta": dict(
            title=title,
            angles=np.asarray(angles),
            pols=np.asarray(pols),
        ),
        "data": {
            "Rs": torch.from_numpy(Rs).float(),
            "theta": torch.from_numpy(true_params_full).float(),
            "wl": torch.from_numpy(wls_full).float(),
            "y": torch.from_numpy(y).long(),
            "groups": groups.tolist(),
            "emb": torch.from_numpy(emb_full).float(),
            "tr_idx": torch.from_numpy(tr_idx).long(),
            "va_idx": torch.from_numpy(va_idx).long(),
            "te_idx": torch.from_numpy(te_idx).long(),
        }
    }
    # ---------------- test prediction ----------------
    preds, trues = [], []
    with torch.no_grad():
        for thb, lb, yb, Rb in dl_te:
            thb, lb, yb = thb.to(device), lb.to(device), yb.to(device)
            pn = model(thb, lb, yb).cpu().numpy()
            preds.append(pn)
            trues.append(Rb.numpy())

    R_hat = np.concatenate(preds, axis=0)
    R_true = np.concatenate(trues, axis=0)

    plot_curves(hist, title=title, keys=["loss"], save_path="figs/paramtor_curves")


    # ----------------  figure + H5 ----------------
    err_sample = np.mean(np.abs(R_hat - R_true), axis=1)
    wl_te  = wls_full[te_idx]

    uniq = np.unique(te_gid)
    var_geom = np.array([R_te[te_gid == g].std() for g in uniq])

    gid_spec = uniq[np.argsort(var_geom)[-1]]
    
    print("gid spec", gid_spec)

    plot_per_class_mae(hist, split="tr", save_prefix="paramtor_tr")
    plot_per_class_mae(hist, split="va", save_prefix="paramtor_va")



    h5path = f"results/paramtor_results.h5"
    save_paper_fig_h5(
        h5path,
        err_sample, y_te, te_gid, wl_te,
        R_true, R_hat, R_ref, wl_ref,
        angles, list(library.values()),
        gid_spec,
        ang_ids=(0, 1, 2),
        pol_ids=(0, 1)
    )


if __name__ == "__main__":
    main()
