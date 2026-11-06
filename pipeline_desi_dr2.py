# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KMS = 299792.458
NEFF = 3.043
WR_H2 = 2.4728e-5 * (1 + 0.2271 * NEFF)
WGAMMA_H2 = 2.4728e-5
ZSTAR = 1089.80
QUICK = os.environ.get("QUICK") == "1"
GHOST_FULL = True
KAPPA_MAX = 0.101
ZB = np.linspace(0.0, 2.35, 601)

MEAN_FILE = "desi_bao_all_gaussian_bao_ALL_GCcomb_mean.txt"
COV_FILE  = "desi_bao_all_gaussian_bao_ALL_GCcomb_cov.txt"

_z, _v, _q = [], [], []
for line in open(MEAN_FILE):
    p = line.split()
    if len(p) == 3 and not p[0].startswith("#"):
        _z.append(float(p[0])); _v.append(float(p[1])); _q.append(p[2])

OBS_Z = np.array(_z)
OBS_VEC = np.array(_v)
OBS_Q = list(_q)
COV13 = np.loadtxt(COV_FILE)
COV13_INV = np.linalg.inv(COV13)
SIG_VEC = np.sqrt(np.diag(COV13))

LABELS = []
for zz, qq in zip(OBS_Z, OBS_Q):
    name = {0.295: "BGS", 0.510: "LRG1", 0.706: "LRG2", 0.934: "LRG3",
            1.321: "ELG2", 1.484: "QSO", 2.330: "LYA"}.get(round(zz, 3), "z=%.2f" % zz)
    tag = "DM" if "DM" in qq else ("DH" if "DH" in qq else "DV")
    LABELS.append(name + ":" + tag)

CMB_CENTERS = np.array([0.0104110, 0.02223, 0.14208])
COV_CMB = 1e-9 * np.array([
    [0.006621,  0.12444,  -1.1929],
    [0.12444,  21.344,   -94.001],
    [-1.1929,  -94.001,  1488.4]
])
COV_CMB_INV = np.linalg.inv(COV_CMB)

# ================= ФОН =================
def E2_lcdm(z, Om, Or):
    return Or*(1+z)**4 + Om*(1+z)**3 + (1.0 - Om - Or)

def E2_cpl(z, Om, Or, w0, wa):
    a = 1.0/(1.0+z)
    return (Or*a**-4 + Om*a**-3
            + (1.0-Om-Or)*a**(-3*(1+w0+wa))*np.exp(3*wa*(a-1.0)))

def rho_x_cand(zgrid, Om, Or, kappa, beta, lam0, alpha_decay):
    Ode = 1.0 - Om - Or
    if Ode <= 1e-6:
        return None
    if abs(kappa) < 1e-12 and abs(lam0) < 1e-12:
        return np.full_like(zgrid, Ode)
    s_min = -np.log(1.0 + zgrid.max()) - 1e-6
    def rhs(s, y):
        a = np.exp(s)
        rt = Om*a**-3 + Or*a**-4 + y[0]
        x = y[0] / rt
        keff = kappa * (1.0 + beta * a**-3)
        lamN = lam0 * np.exp(-alpha_decay * s)
        return [y[0] * (lamN - keff * (1.0 - x))]
    sol = solve_ivp(rhs, (0.0, s_min), [Ode], method="LSODA",
                    dense_output=True, rtol=1e-8, atol=1e-12)
    if not sol.success:
        return None
    s_pts = -np.log(1.0 + zgrid)
    rho = sol.sol(s_pts)[0]
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        return None
    return rho

def E2_cand(zgrid, Om, Or, kappa, beta, lam0, alpha_decay, rho=None):
    if rho is None:
        rho = rho_x_cand(zgrid, Om, Or, kappa, beta, lam0, alpha_decay)
        if rho is None:
            return None
    return Or*(1+zgrid)**4 + Om*(1+zgrid)**3 + rho

# ================= BAO =================
def rd_of(wb, wbc):
    return 147.05 * (wb/0.02236)**-0.13 * (wbc/0.1432)**-0.23 * (NEFF/3.04)**-0.1

def bao_pred(model, p, use_cmb):
    if use_cmb:
        h, wb, wbc = p[0], p[1], p[2]
        if not (0.4 < h < 1.0) or not (0.015 < wb < 0.032) or not (0.08 < wbc < 0.20):
            return None
        Om = wbc/h**2
        Or = WR_H2/h**2
        rd = rd_of(wb, wbc)
        alpha = C_KMS/(100.0*h*rd)
        rest = p[3:]
    else:
        Om, alpha = p[0], p[1]
        if not (0.02 < Om < 0.7) or not (5.0 < alpha < 60.0):
            return None
        Or = WR_H2/0.706**2
        rest = p[2:]

    if Om + Or >= 1.0:
        return None

    if model == "lcdm":
        E2 = E2_lcdm(ZB, Om, Or)
    elif model == "cpl":
        w0, wa = rest
        if not (-2.5 < w0 < 0.5) or not (-8.0 < wa < 4.0):
            return None
        E2 = E2_cpl(ZB, Om, Or, w0, wa)
    elif model == "cand":
        kappa, beta, lam0, alpha_decay = rest
        if not (-0.001 <= kappa <= KAPPA_MAX): return None
        if not (0.0 <= beta <= 10.0): return None
        if not (-0.5 <= lam0 <= 0.8): return None
        if not (0.0 <= alpha_decay < 3.0): return None
        rho = rho_x_cand(ZB, Om, Or, kappa, beta, lam0, alpha_decay)
        if rho is None: return None
        E2 = E2_cand(ZB, Om, Or, kappa, beta, lam0, alpha_decay, rho=rho)
        rt = Or*(1+ZB)**4 + Om*(1+ZB)**3 + rho
        x = rho / rt
        a = 1.0/(1.0+ZB)
        keff = kappa * (1.0 + beta * a**-3)
        lamN = lam0 * np.exp(-alpha_decay * np.log(a))
        G = lamN - keff*(1.0 - x)
        if GHOST_FULL:
            if np.min(G) < -1e-9: return None
        else:
            if G[0] < -1e-9: return None
    else:
        raise ValueError(model)

    if np.any(~np.isfinite(E2)) or np.any(E2 <= 0):
        return None

    I = cumulative_trapezoid(1.0/np.sqrt(E2), ZB, initial=0)
    DM_r = I*alpha
    DH_r = alpha/np.sqrt(E2)

    preds = []
    for zz, qq in zip(OBS_Z, OBS_Q):
        dm = np.interp(zz, ZB, DM_r)
        dh = np.interp(zz, ZB, DH_r)
        if "DV" in qq:
            preds.append((zz*dm*dm*dh)**(1.0/3.0))
        elif "DM" in qq:
            preds.append(dm)
        elif "DH" in qq:
            preds.append(dh)
        else:
            return None
    return np.array(preds)

def chi2_desi(model, p, use_cmb, full=False):
    preds = bao_pred(model, p, use_cmb)
    if preds is None or len(preds) != len(OBS_VEC):
        return 1e6
    d = preds - OBS_VEC
    c2 = float(d @ COV13_INV @ d)
    if full:
        return c2, d, SIG_VEC
    return c2

# ================= ПРИОР CMB (theta*, wb, wbc) =================
def cmb_pred(h, wb, wbc):
    Om = wbc / h**2
    Or = WR_H2 / h**2
    if Om + Or >= 1.0:
        return None

    z = np.linspace(0.0, ZSTAR, 5000)
    E = np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + (1.0 - Om - Or))
    DM_cH0 = np.trapezoid(1.0/E, z)

    lnz = np.linspace(np.log(1.0 + ZSTAR), np.log(1.0 + 1e6), 5000)
    zz = np.exp(lnz) - 1.0
    Ez = np.sqrt(Om*(1+zz)**3 + Or*(1+zz)**4)
    Rb = 3.0*wb/(4.0*WGAMMA_H2)/(1.0+zz)
    cs = 1.0/np.sqrt(3.0*(1.0+Rb))
    rs_cH0 = np.trapezoid(cs/Ez*(1.0+zz), lnz)

    theta_star = rs_cH0 / DM_cH0
    return np.array([theta_star, wb, wbc])

# --- калибровка theta* по фидуциалу (гасит упрощённость ранней физики) ---
_th_fid = cmb_pred(0.6766, 0.02223, 0.14208)[0]
TH_CAL = 0.0104110 / _th_fid

def chi2_cmb(h, wb, wbc):
    pred = cmb_pred(h, wb, wbc)
    if pred is None:
        return 1e6
    pred[0] *= TH_CAL
    d = pred - CMB_CENTERS
    return float(d @ COV_CMB_INV @ d)

def chi2_total(x, model, use_cmb):
    if not use_cmb:
        return chi2_desi(model, x, use_cmb)
    ch = chi2_cmb(x[0], x[1], x[2])
    if ch >= 1e6:
        return 1e6
    return ch + chi2_desi(model, x, use_cmb)

# ================= ФИТЫ =================
def fit_model(model, use_cmb, n_starts=12, maxiter=4000, verbose=False):
    rng = np.random.default_rng(42)
    if use_cmb:
        sp = [[0.68, 0.0223, 0.1432], [0.72, 0.0223, 0.1432], [0.62, 0.0223, 0.1432]]
        if model == "cpl":
            sp = [s + [-0.9, -0.3] for s in sp] + [
                [0.68, 0.0223, 0.1432, -0.4, -1.7],
                [0.68, 0.0223, 0.1432, -1.2, 0.0]]
        elif model == "cand":
            sp = [s + [0.01, 0.0, 0.05, 0.0] for s in sp] + [
                [0.68, 0.0223, 0.1432, 0.10, 0.0, 0.10, 0.0],
                [0.68, 0.0223, 0.1432, 0.05, 1.0, 0.10, 0.5],
                [0.68, 0.0223, 0.1432, 0.00, 0.0, 0.00, 0.0]]
    else:
        sp = [[0.28, 29.0], [0.32, 31.0], [0.25, 28.0]]
        if model == "cpl":
            sp = [s + [-0.9, -0.3] for s in sp] + [
                [0.30, 30.0, -0.4, -1.8],
                [0.30, 30.0, -1.3, 0.0]]
        elif model == "cand":
            sp = [s + [0.01, 0.0, 0.05, 0.0] for s in sp] + [
                [0.30, 30.0, 0.10, 0.0, 0.10, 0.0],
                [0.30, 30.0, 0.05, 1.0, 0.10, 0.5],
                [0.30, 30.0, 0.00, 0.0, 0.00, 0.0]]

    while len(sp) < n_starts:
        sp.append(list(np.array(sp[rng.integers(len(sp))]) *
                       (1 + 0.05*rng.normal(size=len(sp[0])))))

    best, best_x = np.inf, None
    for x0 in sp:
        try:
            r = minimize(chi2_total, np.array(x0), args=(model, use_cmb),
                         method="Nelder-Mead",
                         options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-7,
                                      adaptive=True))
        except Exception:
            continue
        if r.fun < best and np.all(np.isfinite(r.x)):
            best, best_x = float(r.fun), r.x.copy()
    return best, best_x

def profile_kappa_desi(npts=14):
    ks = np.linspace(0.005, KAPPA_MAX, npts)
    prof = []
    for k in ks:
        def f(q, k=k):
            return chi2_desi("cand", [q[0], q[1], k, 0.0, q[2], 0.0], False)
        best = np.inf
        for x0 in ([0.30, 30.0, 0.10], [0.28, 29.5, 0.15], [0.32, 30.5, 0.05]):
            r = minimize(f, np.array(x0), method="Nelder-Mead",
                         options=dict(maxiter=2000, xatol=1e-4, fatol=1e-7))
            best = min(best, float(r.fun))
        prof.append((k, best))
        print("  kappa=%.4f  chi2=%.3f" % (k, best), flush=True)
    return prof

def main():
    lines = ["=== DESI DR2 BAO + приор DESI (theta*, wb, wbc), TH_CAL=%.6f ===" % TH_CAL,
             "центр: theta*=0.0104110, wb=0.02223, wbc=0.14208",
             "GHOST_FULL=%s, ZSTAR=%.2f" % (GHOST_FULL, ZSTAR),
             "данные BAO: %d величин" % len(OBS_VEC), ""]
    fits = {}
    for use_cmb in (False, True):
        tag = "DESI only" if not use_cmb else "DESI + CMB (DESI prior)"
        lines.append("--- %s ---" % tag)
        ns, mi = (4, 1500) if QUICK else (12, 4000)
        for model in ("lcdm", "cpl", "cand"):
            c2, x = fit_model(model, use_cmb, n_starts=ns, maxiter=mi)
            fits[(model, use_cmb)] = x
            if x is None:
                lines.append("%5s  chi2 = %9.3f   params = FAILED" % (model, c2))
            else:
                lines.append("%5s  chi2 = %9.3f   params = %s"
                             % (model, c2, np.round(x, 4).tolist()))
        lines.append("")
    print("\n".join(lines), flush=True)

    print("профиль по kappa...", flush=True)
    prof = profile_kappa_desi(npts=8 if QUICK else 14)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([p[0] for p in prof], [p[1] for p in prof], "o-")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\chi^2$ (DESI only)")
    ax.set_title(r"Профиль $\chi^2(\kappa)$")
    fig.tight_layout(); fig.savefig("profile_kappa.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    zz = np.linspace(0.0, 2.33, 200)
    ax2.plot(zz, -1.0*np.ones_like(zz), "k--", lw=1.5, label="LCDM")
    xcpl = fits.get(("cpl", True))
    if xcpl is not None:
        ax2.plot(zz, xcpl[3] + xcpl[4]*zz/(1+zz), label="CPL (DESI+CMB)")
    xc = fits.get(("cand", True))
    if xc is not None:
        h, wb, wbc, kap, bet, lam0, adec = xc
        Om, Or = wbc/h**2, WR_H2/h**2
        rho = rho_x_cand(zz, Om, Or, kap, bet, lam0, adec)
        if rho is not None:
            a = 1.0/(1.0+zz)
            rt = Or*(1+zz)**4 + Om*(1+zz)**3 + rho
            x = rho/rt
            keff = kap*(1.0 + bet*a**-3)
            lamN = lam0*np.exp(-adec*np.log(a))
            ax2.plot(zz, -1.0 + (lamN - keff*(1.0-x))/3.0,
                     label="кандидат (DESI+CMB)")
    ax2.set_xlabel("z"); ax2.set_ylabel("w(z)"); ax2.legend()
    fig2.tight_layout(); fig2.savefig("wz_curves.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(9, 4.6))
    for model, lbl, col in (("lcdm", "LCDM", "C0"), ("cpl", "CPL", "C1"), ("cand", "кандидат", "C2")):
        r = chi2_desi(model, fits[(model, False)], False, full=True)
        if isinstance(r, float):
            continue
        d, info = r[1], r[2]
        ax3.plot(range(len(LABELS)), np.array(d)/info, "o", color=col, label=lbl)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xticks(range(len(LABELS)))
    ax3.set_xticklabels(LABELS, rotation=60, fontsize=7)
    ax3.set_ylabel("(набл - теор)/ошибка"); ax3.legend(fontsize=8)
    fig3.tight_layout(); fig3.savefig("residuals.png", dpi=150)

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("ГОТОВО: results.txt, profile_kappa.png, wz_curves.png, residuals.png")

if __name__ == "__main__":
    main()