import os
import h5py
import torch
import random
import numpy as np
import treams
from itertools import product
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def set_seed(seed=123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def ensure_stackable_T(T, maxlen):
    """Accept [N,N] or [nwl,N,N]; return [nwl,N,N] cropped to maxlen."""
    T = np.asarray(T)
    if T.ndim == 2:
        T = T[None, ...]
    return T[:, :maxlen, :maxlen]

def wl_to_1d_nm(wl_ds, n_target):
    wl = np.asarray(wl_ds, float) * 1e9
    if wl.ndim == 0:
        wl = np.full((n_target,), float(wl))
    else:
        wl = wl[:n_target]
    return wl

def geom_key_str(cls, base_p, emb, nd=4):
    p = ",".join([f"{v:.{nd}f}" for v in np.asarray(base_p, float)])
    e = np.asarray(emb).ravel()
    e = ",".join([f"{v:.{nd}f}" for v in e.astype(float)])
    return f"{int(cls)}|{p}|{e}"


def grouped_train_val_test_stratified(y, groups, test_size=0.2, val_size=0.25,
                                      seed=42):
    """
    Stratified over class, and group-aware over geometries.
    y      : [B] class labels
    groups : [B] geometry ids (strings)
    """
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)

    cls2gids = {}
    for gid, cls in zip(groups, y):
        cls2gids.setdefault(int(cls), set()).add(gid)

    train_gids, val_gids, test_gids = set(), set(), set()
    for cls in sorted(cls2gids.keys()):
        gids = np.array(sorted(cls2gids[cls]))
        rng.shuffle(gids)

        n = len(gids)
        n_test = max(1, int(round(test_size * n)))
        n_val  = max(1, int(round(val_size * (n - n_test))))

        test_gids.update(gids[:n_test])
        val_gids.update(gids[n_test:n_test + n_val])
        train_gids.update(gids[n_test + n_val:])

    tr_idx = np.where(np.isin(groups, list(train_gids)))[0]
    va_idx = np.where(np.isin(groups, list(val_gids)))[0]
    te_idx = np.where(np.isin(groups, list(test_gids)))[0]
    return tr_idx, va_idx, te_idx

def create_r(i, j, p_idx, T_array, k0s, emb_full, ps, angles, pols, rmax_coef, poltype, lmax=5):
    k0 = k0s[i]
    kpar = np.array([k0*np.sin(angles[j]), 0.])
    eps = treams.Material(emb_full[i])
    tm = treams.TMatrix(
        T_array[i],
        basis=treams.SphericalWaveBasis.default(lmax),
        k0=k0,
        material=eps,
        poltype=poltype,
    )

    lattice = treams.Lattice.square(ps[i])
    metasurf_t = tm.latticeinteraction.solve(lattice, kpar)

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(
        kpar, lattice, rmax_coef * k0
    )
    pol = pols[p_idx]
    illu = treams.plane_wave(
        kpar, pol, k0=k0, basis=pwb, material=eps, poltype=poltype
    )

    metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)
    r = metasurf_s.tr(illu)[1]
    return r

def rs_cache_name(title):
    return f"rs_{title}.npz"

def load_or_compute_Rs(title, t_matrices_full, k0s, emb_full, ps, angles, pols, rmax_coef, poltype, lmax):
    path = rs_cache_name(title)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["rs"]  

    n_struct = len(t_matrices_full)
    n_angles = len(angles)
    n_pols = len(pols)
    ijk_list =  list(product(range(n_struct), range(n_angles), range(n_pols)))
    rs_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(create_r)(i, j, p, t_matrices_full, k0s, emb_full, ps, angles, pols, rmax_coef, poltype, lmax=lmax)
        for (i, j, p) in ijk_list
    )
    rs_flat = np.array(rs_list)
    Rs = rs_flat.reshape(n_struct, n_angles * n_pols)
    np.savez(path, rs=Rs)
    return Rs


def Rvec_from_h5_tr(
    h5path,
    angles,
    *,
    GAP_NM=200.0,
    pols=(0, 1),
    poltype="parity",
    rmax_coef=1.0,
    lmax=5,
):
    with h5py.File(h5path, "r") as f:
        T = np.asarray(f["tmatrix"][...])
        wl = np.asarray(f["vacuum_wavelength"][...]).squeeze() * 1e9

        g = f["scatterer/geometry"]
        if ("radius_top" in g) and ("radius_bottom" in g) and ("height" in g):
            rtop = float(g["radius_top"][()])
            rbot = float(g["radius_bottom"][()])
            h = float(g["height"][()])
            rmax = max(rtop, rbot)
            a = np.sqrt(rmax**2 + (0.5 * h)**2)
        elif ("lengthx" in g) and ("lengthy" in g) and ("lengthz" in g):
            Lx = float(g["lengthx"][()])
            Ly = float(g["lengthy"][()])
            Lz = float(g["lengthz"][()])
            a = 0.5 * np.sqrt(Lx**2 + Ly**2 + Lz**2)
        elif ("radius" in g) and ("height" in g):
            r = float(g["radius"][()])
            h = float(g["height"][()])
            a = np.sqrt(r**2 + (0.5 * h)**2)
        else:
            raise KeyError(f"Unknown geometry keys in {h5path}: {list(g.keys())}")

        emb = float(np.asarray(f["embedding/relative_permittivity"][...]).squeeze())

    if T.ndim == 2:
        T = T[None, ...]
    nwl = T.shape[0]

    treams.config.POLTYPE = poltype

    p_nm = 2.0 * a + float(GAP_NM)
    k0 = 2 * np.pi / wl

    eps = treams.Material(emb)
    tms = [
        treams.TMatrix(
            T[i],
            basis=treams.SphericalWaveBasis.default(lmax),
            k0=k0[i],
            material=eps,
            poltype=poltype,
        )
        for i in range(len(k0))
    ]

    lattice = treams.Lattice.square(p_nm)
    Rs_ref = np.zeros((len(tms), len(angles) * len(pols)))
    for i, tm in enumerate(tms):
        for j, ang in enumerate(angles):
            kpar = np.array([k0[i] * np.sin(ang), 0.0])
            metasurf_t = tm.latticeinteraction.solve(lattice, kpar)

            pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, rmax_coef * k0[i])
            metasurf_s = treams.SMatrices.from_array(metasurf_t, pwb)

            for k, pol in enumerate(pols):
                illu = treams.plane_wave(kpar, pol, k0=k0[i], basis=pwb, material=eps, poltype=poltype)
                Rs_ref[i, j * len(pols) + k] = metasurf_s.tr(illu)[1]

    return Rs_ref, wl

def plot_curves(hist, keys, title="loss curves", save_path=None, show=False):
    """
    Supports either:
       hist["tr_{k}"], hist["va_{k}"]       
       hist["train_{k}"], hist["val_{k}"] + hist["ep"] 

    keys: list like ["loss","cls","reg"]  OR ["data","total"]
    """

    # epoch axis
    if "ep" in hist:
        ep = np.asarray(hist["ep"])
    else:
        k0 = keys[0]
        for pref in ("tr_", "train_"):
            name = f"{pref}{k0}"
            if name in hist:
                ep = np.arange(len(hist[name]))
                break
        else:
            raise KeyError("Could not infer epochs: provide hist['ep'] or matching series keys.")

    plt.figure(figsize=(8, 5))

    for k in keys:
        # try (tr_/va_) convention
        if f"tr_{k}" in hist:
            plt.plot(ep, hist[f"tr_{k}"], label=f"tr_{k}")
        if f"va_{k}" in hist:
            plt.plot(ep, hist[f"va_{k}"], label=f"va_{k}")

        # try (train_/val_) convention
        if f"train_{k}" in hist:
            plt.plot(ep, hist[f"train_{k}"], label=f"train_{k}")
        if f"val_{k}" in hist:
            plt.plot(ep, hist[f"val_{k}"], label=f"val_{k}")

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.yscale("log")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

def plot_per_class_mae(hist, split="va", classes=("cone", "cuboid", "cylinder"),
                       title=None, save_prefix=None):
    ep = np.asarray(hist.get("ep", []))
    if ep.size == 0:
        raise ValueError("hist['ep'] is missing or empty.")

    plt.figure(figsize=(7, 5))
    any_plotted = False
    for cls in classes:
        key = f"mae_{split}_{cls}"
        if key in hist and len(hist[key]) == len(ep):
            plt.plot(ep, hist[key], label=cls)
            any_plotted = True
    if not any_plotted:
        return

    plt.title(title or f"Per-class MAE ({split})")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        plt.savefig(f"figs/{save_prefix}_mae_{split}.png", dpi=160, bbox_inches="tight")
    plt.show()
