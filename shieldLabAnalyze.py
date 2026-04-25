#!/usr/bin/env python3
"""
analyze_interactive.py  —  Oumano et al. 2025, JACMP 26:e70084
===============================================================================
Post-processing for GATE 10 DoseActor .mhd output.

Fitting methodology  (Section 2.4 of the paper)
------------------------------------------------
ODR:  Orthogonal Distance Regression is used to fit the three Archer parameters
      (alpha, beta, gamma) simultaneously, accounting for uncertainty in both
      the thickness measurements (x) and the transmission values (y=T).

FVL:  For each layer thickness (HVL/TVL/CVL/MVL), three transmission-factor
      data points are selected around the target value, an exponential AND a
      polynomial are fitted, and whichever gives the higher R^2 is used to
      solve for the thickness analytically.  A fit is accepted when all four
      estimated thicknesses differ by <= 10% from the local-bracketing values.

Alpha determination (data-driven)
----------------------------------
alpha is bounded near the asymptotic slope of ln(T) vs x measured from the
last few (thickest) data points.  This is purely empirical — no NIST table
is used to constrain any parameter.

    ln T(x) -> -alpha * x  at large x   =>  alpha = -d(lnT)/dx

beta and gamma are then free within physically generous bounds.
beta is NOT tied to any NIST narrow-beam attenuation coefficient.
"""

import argparse, csv, math, threading, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

# ─── output / geometry constants ─────────────────────────────────────────────
OUTPUT_DIR           = Path("output")
N_PRIMARIES_FALLBACK = 2_000_000_000
VOXEL_XY_MM          = 2.5
VOXEL_Z_MM           = 5.0
ARRAY_NXY            = 800
ROI_DIAM_MM          = 150.0
ROI_RADIUS_V         = (ROI_DIAM_MM / 2.0) / VOXEL_XY_MM   # 30 voxels
DEFAULT_TARGET_UNC   = 0.01   # 1%
DEFAULT_ALPHA_TAIL_N = 3
DEFAULT_ALPHA_TOL    = 0.15   # ±15% band around alpha_tail
GAMMA_MAX_FIT        = 50.0   # prevents degenerate large-gamma basin
FVL_ACCEPT_THRESH    = 0.10   # 10% acceptance criterion (paper Section 2.4)

# ─── FVL target transmission values ──────────────────────────────────────────
FVL_TARGETS = {"HVL": 0.5, "QVL": 0.25, "TVL": 0.10,
               "CVL": 0.01, "MVL": 0.001}

# ─── published Table 2 parameters ────────────────────────────────────────────
TABLE2 = {
    ("Tc99m","Lead"):       (2.558,    1.010,    4.344),
    ("Tc99m","Gypsum"):     (0.009549,-0.005312, 1.430),
    ("Tc99m","LWConcrete"): (0.02047, -0.01122,  0.4389),
    ("Tc99m","NWConcrete"): (0.03102, -0.01729,  0.3622),
    ("Tc99m","Steel"):      (0.1581,  -0.04346,  0.2602),
    ("Tc99m","Glass"):      (0.03419, -0.02009,  0.3076),
    ("Lu177","Lead"):       (0.3855,   1.071,    0.2822),
    ("Lu177","Gypsum"):     (0.009594,-0.003783, 0.3739),
    ("Lu177","LWConcrete"): (0.01615, -0.007056, 0.5194),
    ("Lu177","NWConcrete"): (0.02477, -0.01173,  0.4404),
    ("Lu177","Steel"):      (0.0797,   2.243,   28.74),
    ("Lu177","Glass"):      (0.02456, -0.01197,  0.6480),
    ("I131", "Lead"):       (0.1082,   0.2072,   0.5385),
    ("I131", "LWConcrete"): (0.01363, -0.007896, 0.4847),
    ("I131", "NWConcrete"): (0.02062, -0.01220,  0.4179),
    ("I131", "Steel"):      (0.05786, -0.02574,  0.8742),
    ("I131", "Glass"):      (0.02191, -0.01319,  0.4497),
    ("F18",  "Lead"):       (0.166,   -0.02184,  0.2436),
    ("F18",  "LWConcrete"): (0.01126, -0.006463, 0.7475),
    ("F18",  "NWConcrete"): (0.01558, -0.008775, 0.8600),
    ("F18",  "Steel"):      (0.05032, -0.02632,  1.223),
}

# NIST narrow-beam mu — informational display only, NOT used in fitting
MU_NARROW_NIST = {
    ("F18",  "Lead"):       0.1767, ("F18",  "Steel"):      0.0674,
    ("F18",  "NWConcrete"): 0.0205, ("F18",  "LWConcrete"): 0.0157,
    ("F18",  "Glass"):      0.0184, ("F18",  "Gypsum"):     0.0191,
    ("Tc99m","Lead"):       2.284,  ("Tc99m","Steel"):      0.2730,
    ("Tc99m","NWConcrete"): 0.0373, ("Tc99m","LWConcrete"): 0.0285,
    ("Tc99m","Glass"):      0.0335, ("Tc99m","Gypsum"):     0.0350,
    ("I131", "Lead"):       0.3160, ("I131", "Steel"):      0.0865,
    ("I131", "NWConcrete"): 0.0224, ("I131", "LWConcrete"): 0.0171,
    ("I131", "Glass"):      0.0202, ("I131", "Gypsum"):     0.0210,
    ("Lu177","Lead"):       1.090,  ("Lu177","Steel"):      0.1420,
    ("Lu177","NWConcrete"): 0.0290, ("Lu177","LWConcrete"): 0.0222,
    ("Lu177","Glass"):      0.0261, ("Lu177","Gypsum"):     0.0272,
    ("Zr89", "Lead"):       0.0795, ("Zr89", "Steel"):      0.0487,
    ("Zr89", "NWConcrete"): 0.0151, ("Zr89", "LWConcrete"): 0.0116,
    ("Zr89", "Glass"):      0.0136, ("Zr89", "Gypsum"):     0.0141,
}
NUCLIDE_ENERGY_KEV = {"F18":511.0,"Tc99m":140.5,"I131":364.0,
                       "Lu177":208.0,"Zr89":909.0}
BARRIER_COLORS = {
    "Lead":"#1f77b4","LWConcrete":"#ff7f0e","NWConcrete":"#2ca02c",
    "Steel":"#d62728","Glass":"#9467bd","Gypsum":"#8c564b",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHER EQUATION
# ═══════════════════════════════════════════════════════════════════════════════

def archer_transmission(x, alpha, beta, gamma):
    """
    T(x) = [(1 + beta/alpha)*exp(alpha*gamma*x) - beta/alpha]^(-1/gamma)

    Physical roles
    --------------
    alpha  : asymptotic slope  -d(lnT)/dx at large x  (mm^-1)
             Determined empirically from the data tail.
    beta   : shape / build-up parameter.  Free — NOT tied to NIST mu.
    gamma  : beam-hardening rate.  > 0, capped at 50.
    """
    return ((1.0 + beta/alpha)*np.exp(alpha*gamma*x) - beta/alpha)**(-1.0/gamma)


def archer_odr_func(params, x):
    """ODR-compatible wrapper: params = [alpha, beta, gamma]."""
    a, b, g = params
    return archer_transmission(x, a, b, g)


def archer_thickness(T_target, alpha, beta, gamma):
    """Inverse Archer: thickness (mm) giving transmission T_target."""
    return (1.0/(alpha*gamma))*math.log(
        (T_target**(-gamma) + beta/alpha) / (1.0 + beta/alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ALPHA FROM DATA TAIL  (OLS on ln T vs x)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_alpha_from_tail(thicknesses, transmissions, n_tail=DEFAULT_ALPHA_TAIL_N):
    """
    Fit  ln T = -alpha*x + c  to the last n_tail valid points (OLS).

    At large x the Archer equation asymptotes exactly to this line,
    so the slope gives alpha directly from the data.

    Returns
    -------
    alpha_tail, r2, tail_x, tail_lnT, slope_se
    """
    valid   = (np.asarray(transmissions) > 0) & np.isfinite(transmissions)
    xv      = np.asarray(thicknesses)[valid]
    Tv      = np.asarray(transmissions)[valid]
    n_tail  = min(n_tail, int(valid.sum()))
    if n_tail < 2:
        raise ValueError("Need >= 2 valid points for tail fit.")

    idx      = np.argsort(xv)[-n_tail:]
    tail_x   = xv[idx]
    tail_lnT = np.log(Tv[idx])

    X      = np.column_stack([np.ones(n_tail), tail_x])
    coeffs,_,_,_ = np.linalg.lstsq(X, tail_lnT, rcond=None)
    c, neg_alpha = coeffs
    alpha_tail   = max(-neg_alpha, 1e-9)

    lnT_pred = c - alpha_tail*tail_x
    ss_res   = float(np.sum((tail_lnT - lnT_pred)**2))
    ss_tot   = float(np.sum((tail_lnT - tail_lnT.mean())**2))
    r2       = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")

    slope_se = float("nan")
    if n_tail > 2 and ss_res > 0:
        sxx = float(np.sum((tail_x - tail_x.mean())**2))
        if sxx > 0:
            slope_se = math.sqrt((ss_res/(n_tail-2)) / sxx)

    return alpha_tail, r2, tail_x, tail_lnT, slope_se


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ODR FIT  (orthogonal distance regression, paper Section 2.4)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_archer_odr(thicknesses, transmissions, sigma_T,
                   alpha_tail, alpha_tol=DEFAULT_ALPHA_TOL,
                   thickness_unc_mm=0.5,
                   nuclide=None, barrier=None):
    """
    Fit Archer parameters using Orthogonal Distance Regression (ODR),
    matching the OriginPro ODR algorithm described in Section 2.4.

    ODR weights
    -----------
    w_x = 1 / sigma_x^2   where sigma_x = thickness_unc_mm (default 0.5 mm)
    w_y = 1 / sigma_T^2   where sigma_T is from DoseActor quadrature
                           (falls back to 1% if unavailable)

    Alpha bounds
    ------------
    alpha is constrained to [alpha_tail*(1-tol), alpha_tail*(1+tol)].
    beta and gamma are free (beta has no NIST anchor).

    Returns
    -------
    alpha, beta, gamma, sd_alpha, sd_beta, sd_gamma, odr_result
    """
    x   = np.asarray(thicknesses, float)
    y   = np.asarray(transmissions, float)
    sig = np.asarray(sigma_T, float)

    # Fallback uncertainty: 1% where missing/nan
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.01 * y)
    sig = np.where(sig > 0, sig, 1e-6)

    sigma_x = np.full_like(x, thickness_unc_mm)

    a_lo = alpha_tail * (1.0 - alpha_tol)
    a_hi = alpha_tail * (1.0 + alpha_tol)
    beta_max = 5.0 * alpha_tail
    # Physical constraint: (1 + beta/alpha) > 0  →  beta > -alpha
    # This ensures T(x) is strictly monotonically decreasing.
    # Use -0.90*alpha_tail as the lower bound (small safety margin from -alpha).
    beta_min = -0.90 * alpha_tail

    # Build multi-start grid — exclude starting points below physical bound
    beta_vals  = np.array([-5,-3,-1.5,-0.5,-0.1,0,0.1,0.5,1.5,3,5]) * alpha_tail
    beta_vals  = beta_vals[beta_vals >= beta_min]   # drop unphysical starts
    gamma_vals = np.unique(np.round(np.concatenate([
        np.linspace(0.05, 0.5,  5),
        np.linspace(0.5,  5.0,  8),
        np.linspace(5.0, 30.0,  5)]), 4))
    candidates = [(alpha_tail, b, g)
                  for b in beta_vals for g in gamma_vals]
    pub = TABLE2.get((nuclide, barrier)) if nuclide else None
    if pub:
        candidates.insert(0, pub)

    print(f"\n  ODR multi-start: {len(candidates)} candidates  "
          f"alpha in [{a_lo:.5f}, {a_hi:.5f}]  "
          f"beta in [{beta_min:.5f}, {beta_max:.5f}]  "
          f"gamma cap {GAMMA_MAX_FIT}")

    def _clip(p0):
        a = float(np.clip(p0[0], a_lo + 1e-10, a_hi - 1e-10))
        b = float(np.clip(p0[1], beta_min + 1e-10, beta_max - 1e-10))
        g = float(np.clip(p0[2], 1e-4, GAMMA_MAX_FIT - 1e-4))
        return [a, b, g]

    best_popt = None; best_res = None; best_rmse = np.inf; n_conv = 0

    for p0 in candidates:
        p0c = _clip(p0)
        # --- scipy.odr path ---
        try:
            model = Model(archer_odr_func)
            data  = RealData(x, y, sx=sigma_x, sy=sig)
            odr   = ODR(data, model, beta0=p0c,
                        ifixb=[0, 0, 0],   # all free
                        maxit=1000)
            # Enforce bounds via penalty restart using curve_fit fallback
            res   = odr.run()
            popt  = res.beta
            if not (a_lo <= popt[0] <= a_hi and
                    beta_min <= popt[1] <= beta_max and
                    0 < popt[2] <= GAMMA_MAX_FIT):
                raise ValueError("ODR solution outside bounds")
            y_fit = archer_transmission(x, *popt)
            ok    = (y > 0) & (y_fit > 0) & np.isfinite(y_fit)
            if not ok.any():
                raise ValueError("no valid points")
            rmse = float(np.sqrt(np.mean(
                (np.log10(y_fit[ok]) - np.log10(y[ok]))**2)))
            n_conv += 1
            if rmse < best_rmse:
                best_rmse = rmse; best_popt = popt; best_res = res
        except Exception:
            # Fallback: bounded curve_fit with ODR-equivalent weights
            try:
                w = 1.0 / sig**2
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        archer_transmission, x, y,
                        p0=p0c,
                        bounds=([a_lo, beta_min, 1e-4],
                                [a_hi, +beta_max, GAMMA_MAX_FIT]),
                        sigma=sig, absolute_sigma=True,
                        maxfev=200_000)
                y_fit = archer_transmission(x, *popt)
                ok    = (y > 0) & (y_fit > 0) & np.isfinite(y_fit)
                if not ok.any(): continue
                rmse = float(np.sqrt(np.mean(
                    (np.log10(y_fit[ok]) - np.log10(y[ok]))**2)))
                n_conv += 1
                if rmse < best_rmse:
                    best_rmse = rmse; best_popt = popt; best_res = None
            except Exception:
                continue

    if best_popt is None:
        raise ValueError(
            f"All {len(candidates)} ODR starting points failed.  "
            "Try --fit-min-T or --alpha-tol to widen bounds.")

    a, b, g = best_popt
    # Extract parameter standard deviations from ODR result if available
    if best_res is not None and hasattr(best_res, "sd_beta"):
        sd_a, sd_b, sd_g = best_res.sd_beta
    else:
        sd_a = sd_b = sd_g = float("nan")

    print(f"  ODR converged: {n_conv}/{len(candidates)}  "
          f"best log-RMSE = {best_rmse:.5f}")
    print(f"  alpha={a:.6f} (alpha/alpha_tail={a/alpha_tail:.4f})  "
          f"beta={b:.6f}  gamma={g:.6f}")

    return a, b, g, sd_a, sd_b, sd_g, best_res


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LOCAL BRACKETING FVL  (paper Section 2.4 spreadsheet method)
# ═══════════════════════════════════════════════════════════════════════════════

def local_fvl(thicknesses, transmissions, T_target, n_bracket=3):
    """
    Compute a single FVL thickness by the paper's local-bracketing method:

    1. Select the n_bracket points whose T is closest to T_target
       (measured in |log(T) - log(T_target)|).
    2. Fit an exponential:  T = a * exp(b * x)   (OLS on ln T)
    3. Fit a polynomial  :  T = a + b*x + c*x^2  (polyfit degree 2)
    4. Choose whichever gives higher R^2.
    5. Solve analytically for the thickness x where T = T_target.

    Returns
    -------
    dict with keys: thickness, method ("exp"/"poly"), r2_exp, r2_poly, ok
    """
    t = np.asarray(thicknesses, float)
    T = np.asarray(transmissions, float)
    valid = (T > 0) & np.isfinite(T)
    tv, Tv = t[valid], T[valid]

    if len(tv) < 2:
        return {"thickness": float("nan"), "method": "n/a",
                "r2_exp": float("nan"), "r2_poly": float("nan"), "ok": False}

    log_tgt = math.log(T_target)
    dist    = np.abs(np.log(Tv) - log_tgt)

    # Split into above / below T_target and select straddling points.
    # Prefer at least one point on each side so the interpolation crosses
    # T_target rather than extrapolating a same-side parabola.
    above = np.where(Tv >= T_target)[0]   # T >= T_target  (thinner barrier)
    below = np.where(Tv  < T_target)[0]   # T <  T_target  (thicker barrier)
    n_each = max(1, n_bracket // 2)

    sel = set()
    if len(above) > 0:
        for i in above[np.argsort(dist[above])[:n_each]]:
            sel.add(i)
    if len(below) > 0:
        for i in below[np.argsort(dist[below])[:n_each]]:
            sel.add(i)
    for i in np.argsort(dist):        # pad to n_bracket with overall nearest
        if len(sel) >= n_bracket:
            break
        sel.add(i)

    idx = np.array(sorted(sel))
    xb  = tv[idx]; Tb = Tv[idx]

    if len(xb) < 2:
        return {"thickness": float("nan"), "method": "n/a",
                "r2_exp": float("nan"), "r2_poly": float("nan"), "ok": False}

    x_lo, x_hi = float(xb.min()), float(xb.max())

    # ── Exponential fit: ln T = ln(a) + b*x ────────────────────────────
    lnTb = np.log(Tb)
    X    = np.column_stack([np.ones(len(xb)), xb])
    try:
        ce,_,_,_ = np.linalg.lstsq(X, lnTb, rcond=None)
        lnA, bE  = ce
        lnTb_pred_e = lnA + bE * xb
        ss_res_e = float(np.sum((lnTb - lnTb_pred_e)**2))
        ss_tot   = float(np.sum((lnTb - lnTb.mean())**2))
        r2_exp   = 1.0 - ss_res_e/ss_tot if ss_tot > 0 else float("nan")
        x_exp    = (math.log(T_target) - lnA) / bE if abs(bE) > 1e-15 else float("nan")
        # Reject wildly extrapolated solutions
        if math.isfinite(x_exp) and not (x_lo * 0.5 <= x_exp <= x_hi * 2.0):
            x_exp = float("nan")
    except Exception:
        r2_exp = float("nan"); x_exp = float("nan")

    # ── Polynomial fit: T = a + b*x + c*x^2 (or linear if only 2 pts) ──
    try:
        deg   = min(2, len(xb) - 1)
        cp    = np.polyfit(xb, Tb, deg)
        Tb_pred_p = np.polyval(cp, xb)
        ss_res_p  = float(np.sum((Tb - Tb_pred_p)**2))
        ss_tot_p  = float(np.sum((Tb - Tb.mean())**2))
        r2_poly   = 1.0 - ss_res_p/ss_tot_p if ss_tot_p > 0 else float("nan")
        x_poly    = float("nan")
        if deg == 1:
            c1, c0 = cp
            x_poly = (T_target - c0) / c1 if abs(c1) > 1e-15 else float("nan")
        elif deg == 2:
            c2, c1, c0 = cp
            disc = c1**2 - 4*c2*(c0 - T_target)
            if abs(c2) < 1e-15 and abs(c1) > 1e-15:
                x_poly = (T_target - c0) / c1
            elif disc >= 0 and abs(c2) > 1e-15:
                r1 = (-c1 + math.sqrt(disc)) / (2*c2)
                r2 = (-c1 - math.sqrt(disc)) / (2*c2)
                xmid = float(xb.mean())
                cands = [r for r in (r1, r2) if math.isfinite(r) and r > 0]
                x_poly = (min(cands, key=lambda r: abs(r - xmid))
                          if cands else float("nan"))
        # Reject wildly extrapolated solutions
        if math.isfinite(x_poly) and not (x_lo * 0.5 <= x_poly <= x_hi * 2.0):
            x_poly = float("nan")
    except Exception:
        r2_poly = float("nan"); x_poly = float("nan")

    # ── Choose best: prefer whichever gives a valid result with higher R² ──
    exp_ok  = math.isfinite(r2_exp)  and math.isfinite(x_exp)
    poly_ok = math.isfinite(r2_poly) and math.isfinite(x_poly)
    if exp_ok and poly_ok:
        use_exp = r2_exp >= r2_poly
    else:
        use_exp = exp_ok
    method   = "exp" if use_exp else "poly"
    x_chosen = x_exp if use_exp else x_poly

    return {"thickness": x_chosen, "method": method,
            "r2_exp": r2_exp, "r2_poly": r2_poly, "ok": math.isfinite(x_chosen)}


def compute_fvl_with_local(thicknesses, transmissions,
                            alpha, beta, gamma, n_bracket=3):
    """
    Compute FVL thicknesses via:
      (a) Archer equation analytical inversion
      (b) Local bracketing (paper Section 2.4)

    Acceptance: |archer - local| / local <= FVL_ACCEPT_THRESH (10%)

    Returns list of dicts, one per FVL layer.
    """
    rows = []
    for label, T_target in FVL_TARGETS.items():
        try:
            x_archer = archer_thickness(T_target, alpha, beta, gamma)
        except Exception:
            x_archer = float("nan")

        loc = local_fvl(thicknesses, transmissions, T_target, n_bracket)
        x_local = loc["thickness"]

        if math.isfinite(x_archer) and math.isfinite(x_local) and x_local > 0:
            delta_pct = abs(x_archer - x_local) / x_local * 100.0
            accepted  = delta_pct <= FVL_ACCEPT_THRESH * 100
        else:
            delta_pct = float("nan")
            accepted  = False

        rows.append({
            "label":    label,
            "T_target": T_target,
            "x_archer": x_archer,
            "x_local":  x_local,
            "method":   loc["method"],
            "r2_exp":   loc["r2_exp"],
            "r2_poly":  loc["r2_poly"],
            "delta_pct":delta_pct,
            "accepted": accepted,
        })
    return rows


def all_fvl_accepted(fvl_rows):
    return all(r["accepted"] for r in fvl_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

def check_constraints(alpha, beta, gamma, alpha_tail, alpha_tol):
    """
    Returns list of (label, passed, detail_str).
    All constraints are advisory (displayed live in GUI).
    """
    c = []
    c.append(("alpha > 0",
               alpha > 0,
               f"alpha = {alpha:.6f} mm^-1"))

    if alpha_tail is not None:
        lo = alpha_tail*(1-alpha_tol); hi = alpha_tail*(1+alpha_tol)
        c.append((f"alpha within +/-{alpha_tol*100:.0f}% of alpha_tail",
                   lo <= alpha <= hi,
                   f"alpha={alpha:.6f}  alpha_tail={alpha_tail:.6f}  "
                   f"ratio={alpha/alpha_tail:.3f}  [{lo:.6f},{hi:.6f}]"))

    c.append(("gamma > 0",
               gamma > 0,
               f"gamma = {gamma:.6f}"))
    c.append((f"gamma <= {GAMMA_MAX_FIT:.0f}",
               gamma <= GAMMA_MAX_FIT,
               f"gamma = {gamma:.4f}"))
    return c


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O  (unchanged from analyzeFinal.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_dose(path):
    return sitk.GetArrayFromImage(
        sitk.ReadImage(str(path))).astype(np.float64)

def load_uncertainty(dose_path):
    name = dose_path.name
    unc_name = (name
                .replace("_dose_dose.mhd","_dose_uncertainty.mhd")
                .replace("_dose.mhd","_dose_uncertainty.mhd")
                .replace("_edep.mhd","_edep_uncertainty.mhd"))
    p = dose_path.parent / unc_name
    if not p.exists(): return None
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float64)

def read_n_primaries(dose_path):
    stem = (dose_path.stem
            .replace("_dose_dose","_dose")
            .replace("_dose","")
            .replace("_edep",""))
    cands = [dose_path.parent / f"{stem}_stats.txt",
             dose_path.parent/"output"/f"{stem}_stats.txt",
             dose_path.parent.parent/f"{stem}_stats.txt"]
    sp = next((p for p in cands if p.exists()), None)
    if sp is None:
        print(f"  !! Stats not found for {dose_path.name},"
              f" using fallback N={N_PRIMARIES_FALLBACK:,}")
        return N_PRIMARIES_FALLBACK
    import json as _j; txt = sp.read_text()
    try:
        d = _j.loads(txt)
        for k in ("events","nb_events","NumberOfEvents"):
            if k in d:
                v = d[k]
                if isinstance(v,dict): v = v.get("value",v)
                return int(v)
    except Exception: pass
    for line in txt.splitlines():
        for k in ("NumberOfEvents","Number of events","Events","nb_events"):
            if k.lower() in line.lower() and "=" in line:
                try:
                    return int(line.split("=")[-1].strip()
                               .split()[0].replace(",",""))
                except Exception: pass
    print(f"  !! Could not parse N from {sp.name},"
          f" using fallback N={N_PRIMARIES_FALLBACK:,}")
    return N_PRIMARIES_FALLBACK

def parse_thickness(path):
    stem = path.stem.replace("_dose","").replace("_edep","")
    for part in stem.split("_"):
        if "mm" in part:
            try: return float(part.replace("mm",""))
            except ValueError: pass
    raise ValueError(f"Cannot parse thickness from {path.name}")

def _build_roi_mask(nY, nX):
    cx = (nX-1)/2.0; cy = (nY-1)/2.0
    yy,xx = np.ogrid[:nY,:nX]
    return (xx-cx)**2+(yy-cy)**2 <= ROI_RADIUS_V**2

def _is_original_actor(arr):
    """Detect whether array is the original large DoseActor (800x800x4-ish)
    vs the reduced footprint from gateTurbo/gateFast."""
    if arr.ndim < 3:
        return False
    nZ, nY, nX = arr.shape
    # Original actor: large XY extent (>100) and multiple Z slices (>=4)
    return nY >= 100 and nX >= 100 and nZ >= 4

def roi_mean_dose(arr, z_flipped=False):
    if _is_original_actor(arr):
        nZ,nY,nX = arr.shape
        p0,p1 = (nZ-3,nZ-1) if z_flipped else (1,3)
        return float(arr[p0:p1,:,:][:,_build_roi_mask(nY,nX)].mean())
    else:
        # Reduced DoseActor: mean of all voxels
        flat = arr.ravel()
        if flat.size == 0:
            return 0.0
        return float(flat.mean())

def roi_relative_uncertainty(dose_arr, unc_arr, z_flipped=False):
    if _is_original_actor(dose_arr):
        nZ,nY,nX = dose_arr.shape
        p0,p1 = (nZ-3,nZ-1) if z_flipped else (1,3)
        mask  = _build_roi_mask(nY,nX)
        D_roi = dose_arr[p0:p1,:,:][:,mask]
        u_roi =  unc_arr[p0:p1,:,:][:,mask]
    else:
        # Reduced actor: use all voxels
        D_roi = dose_arr.ravel()
        u_roi = unc_arr.ravel()
    D_mean= float(D_roi.mean())
    if D_mean==0: return float("nan")
    sigma_abs = math.sqrt(float(np.sum((u_roi*D_roi)**2)))/D_roi.size
    return sigma_abs/D_mean

def transmission_uncertainty(dose_bar, unc_bar, dose_air, unc_air,
                              z_flipped=False):
    D_bar = roi_mean_dose(dose_bar,z_flipped)
    D_air = roi_mean_dose(dose_air,z_flipped)
    s_bar = roi_relative_uncertainty(dose_bar,unc_bar,z_flipped)
    s_air = roi_relative_uncertainty(dose_air,unc_air,z_flipped)
    if math.isnan(s_bar) or math.isnan(s_air) or D_air==0:
        return {"D_bar_roi":D_bar,"D_air_roi":D_air,
                "sigma_rel_bar":s_bar,"sigma_rel_air":s_air,
                "sigma_rel_T":float("nan"),"sigma_abs_T":float("nan")}
    T = D_bar/D_air
    s_T = math.sqrt(s_bar**2+s_air**2)
    return {"D_bar_roi":D_bar,"D_air_roi":D_air,
            "sigma_rel_bar":s_bar,"sigma_rel_air":s_air,
            "sigma_rel_T":s_T,"sigma_abs_T":T*s_T}

def estimate_n_primaries(sigma_rel_T_test, n_test,
                          target_rel_T=DEFAULT_TARGET_UNC):
    if math.isnan(sigma_rel_T_test) or sigma_rel_T_test<=0:
        return {"N_needed":None,"note":"Cannot estimate: sigma nan or zero"}
    scale = (sigma_rel_T_test/target_rel_T)**2
    return {"N_test":n_test,"sigma_test_%":sigma_rel_T_test*100,
            "target_%":target_rel_T*100,"scale_factor":scale,
            "N_needed":int(math.ceil(n_test*scale))}


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECT TRANSMISSION DATA
# ═══════════════════════════════════════════════════════════════════════════════

def collect_transmission(nuclide, barrier, output_dir,
                          verbose=False, target_unc=DEFAULT_TARGET_UNC):
    air_globs = (list(output_dir.glob(f"{nuclide}_Air_*mm_dose.mhd")) or
                 list(output_dir.glob(f"{nuclide}_Air_*mm_dose_dose.mhd")) or
                 list(output_dir.glob(f"{nuclide}_Air_*mm_edep.mhd")))
    if not air_globs:
        raise FileNotFoundError(
            f"No air reference for {nuclide} in {output_dir}")
    air_path = air_globs[0]
    arr_air  = load_dose(air_path)
    unc_air  = load_uncertainty(air_path)
    n_air    = read_n_primaries(air_path)
    nZ       = arr_air.shape[0]
    if _is_original_actor(arr_air):
        prof_air = np.array([arr_air[z].mean() for z in range(nZ)])
        z_flipped= bool(prof_air[0] < prof_air[nZ-1])
    else:
        z_flipped = False   # irrelevant for reduced actor

    d_air = roi_mean_dose(arr_air, z_flipped) / n_air

    actor_mode = "original (ROI)" if _is_original_actor(arr_air) else "reduced (full mean)"
    print(f"\n  Air reference: {air_path.name}")
    print(f"  Array shape: {arr_air.shape}  actor mode: {actor_mode}")
    print(f"  N_primaries: {n_air:,}  d_air/N: {d_air:.6e}")

    files = sorted(
        list(output_dir.glob(f"{nuclide}_{barrier}_*mm_dose.mhd"))+
        list(output_dir.glob(f"{nuclide}_{barrier}_*mm_dose_dose.mhd"))+
        list(output_dir.glob(f"{nuclide}_{barrier}_*mm_edep.mhd")),
        key=parse_thickness)
    seen = {}
    for f in files:
        t = parse_thickness(f)
        if t not in seen or "_dose_dose" not in f.name: seen[t]=f
    files = [seen[t] for t in sorted(seen)]
    if not files:
        raise FileNotFoundError(
            f"No dose files for {nuclide}/{barrier} in {output_dir}")

    thicknesses=[]; transmissions=[]; doses_barrier=[]
    unc_results=[]; n_estimates=[]; sigma_T=[]

    print(f"\n  {'t(mm)':>8}  {'T':>10}  {'sbar%':>6}  {'sair%':>6}  "
          f"{'sT%':>6}  {'T+-s':>14}")
    print(f"  {'--':>8}  {'--':>10}  {'--':>6}  {'--':>6}  {'--':>6}  {'--':>14}")

    for f in files:
        t       = parse_thickness(f)
        arr_bar = load_dose(f)
        unc_bar = load_uncertainty(f)
        n_bar   = read_n_primaries(f)
        d_bar   = roi_mean_dose(arr_bar,z_flipped)/n_bar
        T       = d_bar/d_air

        if unc_bar is not None and unc_air is not None:
            unc   = transmission_uncertainty(arr_bar,unc_bar,arr_air,unc_air,z_flipped)
            n_est = estimate_n_primaries(unc["sigma_rel_T"],n_bar,target_unc)
        else:
            raw_b = roi_mean_dose(arr_bar,z_flipped)
            raw_a = roi_mean_dose(arr_air,z_flipped)
            unc   = {"D_bar_roi":raw_b,"D_air_roi":raw_a,
                     "sigma_rel_bar":float("nan"),"sigma_rel_air":float("nan"),
                     "sigma_rel_T":float("nan"),"sigma_abs_T":float("nan")}
            n_est = {"N_needed":None,"note":"unc file missing"}

        s_T_val = unc["sigma_rel_T"]
        a_T_val = unc["sigma_abs_T"]
        def _f(v): return f"{v*100:6.2f}" if math.isfinite(v) else "   n/a"
        T_pm = (f"{T:.4f}+-{a_T_val:.4f}" if math.isfinite(a_T_val)
                else f"{T:.6f}      ")
        print(f"  {t:>8.3f}  {T:>10.5f}  {_f(unc['sigma_rel_bar'])}  "
              f"{_f(unc['sigma_rel_air'])}  {_f(s_T_val)}  {T_pm:>14}")

        thicknesses.append(t); transmissions.append(T)
        doses_barrier.append(d_bar); unc_results.append(unc)
        n_estimates.append(n_est)
        sigma_T.append(a_T_val if math.isfinite(a_T_val) else float("nan"))

    return (np.array(thicknesses), np.array(transmissions),
            np.array(doses_barrier), float(d_air),
            unc_results, n_estimates, np.array(sigma_T))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def select_fit_points(thicknesses, transmissions, n_points=None, min_T=None):
    mask = transmissions > 0
    if min_T is not None: mask &= transmissions >= min_T
    if n_points is not None and n_points > 0:
        idx = np.where(mask)[0]
        if len(idx) > n_points: mask[idx[n_points:]] = False
    return mask

def fit_archer_full(thicknesses, transmissions, sigma_T,
                    n_points=None, min_T=None,
                    alpha_tail_n=DEFAULT_ALPHA_TAIL_N,
                    alpha_tol=DEFAULT_ALPHA_TOL,
                    thickness_unc_mm=0.5,
                    nuclide=None, barrier=None):
    """
    Full two-step Archer fit per Section 2.4:
      Step 1: alpha from OLS on ln T vs x (last alpha_tail_n valid points)
      Step 2: ODR with alpha pinned near alpha_tail, beta+gamma free
    """
    fit_mask = select_fit_points(thicknesses, transmissions, n_points, min_T)
    x  = thicknesses[fit_mask]
    y  = transmissions[fit_mask]
    sy = sigma_T[fit_mask]

    if len(x) < 3:
        raise ValueError(f"Only {len(x)} point(s) selected — need >= 3.")

    # Step 1
    alpha_tail, r2_tail, tail_x, tail_lnT, slope_se = \
        fit_alpha_from_tail(x, y, n_tail=alpha_tail_n)

    se_s = f"  SE={slope_se:.5f}" if math.isfinite(slope_se) else ""
    print(f"\n  Step 1 — alpha from tail ({alpha_tail_n} pts: "
          f"x={','.join(f'{v:.1f}' for v in tail_x)} mm)")
    print(f"    alpha_tail = {alpha_tail:.6f} mm^-1   "
          f"R^2 = {r2_tail:.4f}{se_s}")
    if r2_tail < 0.98:
        print("    !! R^2 < 0.98 — tail may not be fully straight yet.")

    mu_nist = MU_NARROW_NIST.get((nuclide,barrier)) if nuclide else None
    if mu_nist:
        print(f"    alpha_tail/mu_NIST = {alpha_tail/mu_nist:.3f}  "
              f"(mu_NIST={mu_nist:.5f} mm^-1 — info only)")

    # Step 2 — ODR
    a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_odr(
        x, y, sy, alpha_tail, alpha_tol, thickness_unc_mm, nuclide, barrier)

    return (a, b, g, sd_a, sd_b, sd_g, fit_mask, alpha_tail, r2_tail, odr_res)


def compute_fit_quality(t_data, T_data, alpha, beta, gamma):
    valid = T_data > 0
    if not valid.any(): return float("nan"), float("nan")
    try:
        T_fit = archer_transmission(t_data[valid], alpha, beta, gamma)
        ok    = (T_fit>0) & np.isfinite(T_fit)
        if not ok.any(): return float("nan"),float("nan")
        log_r = np.log10(T_fit[ok]) - np.log10(T_data[valid][ok])
        rmse  = float(np.sqrt(np.mean(log_r**2)))
        pct_r = (T_fit-T_data[valid])/T_data[valid]*100
        return rmse, float(np.max(np.abs(pct_r)))
    except Exception:
        return float("nan"),float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_fvl_table(fvl_rows, nuclide, barrier, alpha, beta, gamma):
    pub  = TABLE2.get((nuclide,barrier))
    ok_s = "ALL ACCEPTED" if all_fvl_accepted(fvl_rows) else "SOME REJECTED"
    print(f"\n{'='*72}")
    print(f"  {nuclide} / {barrier}   FVL comparison ({ok_s})")
    print(f"  alpha={alpha:.6f}  beta={beta:.7f}  gamma={gamma:.6f}")
    print(f"{'─'*72}")
    hdr = (f"  {'Lyr':<5}  {'T_tgt':>5}  {'Archer':>8}  "
           f"{'Local':>8}  {'Meth':>4}  {'R2':>6}  {'D%':>6}  {'OK':>2}")
    if pub: hdr += f"  {'Pub':>8}"
    print(hdr)
    print(f"  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*8}  "
          f"{'─'*4}  {'─'*6}  {'─'*6}  {'─'*2}"
          + (f"  {'─'*8}" if pub else ""))
    for r in fvl_rows:
        r2v = r["r2_poly"] if r["method"]=="poly" else r["r2_exp"]
        row = (f"  {r['label']:<5}  {r['T_target']:>5.3f}  "
               f"{r['x_archer']:>8.2f}  {r['x_local']:>8.2f}  "
               f"{r['method']:>4}  {r2v:>6.4f}  "
               f"{r['delta_pct']:>5.1f}%  "
               f"{'v' if r['accepted'] else 'X':>2}")
        if pub:
            try: row += f"  {archer_thickness(r['T_target'],*pub):>8.2f}"
            except: row += f"  {'err':>8}"
        print(row)
    print(f"{'='*72}")


def write_transmission_csv(nuclide, barrier, thicknesses, transmissions,
                            doses_barrier, dose_air, alpha, beta, gamma,
                            unc_results, n_estimates, output_dir, fit_mask):
    csv_path = output_dir/f"{nuclide}_{barrier}_transmission_data.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["thickness_mm","dose_transmitted","dose_reference",
                    "T_simulated","T_fitted","pct_residual",
                    "sigma_rel_bar%","sigma_rel_air%",
                    "sigma_rel_T%","sigma_abs_T","used_in_fit"])
        for t,d_bar,T_sim,unc,n_est,in_fit in zip(
                thicknesses,doses_barrier,transmissions,
                unc_results,n_estimates,fit_mask):
            T_fit = archer_transmission(t,alpha,beta,gamma)
            pct   = 100*(T_sim-T_fit)/T_sim if T_sim>0 else float("nan")
            a_T   = unc["sigma_abs_T"]
            w.writerow([f"{t:.4f}",f"{d_bar:.6e}",f"{dose_air:.6e}",
                        f"{T_sim:.6e}",f"{T_fit:.6e}",f"{pct:.2f}",
                        f"{unc['sigma_rel_bar']*100:.3f}",
                        f"{unc['sigma_rel_air']*100:.3f}",
                        f"{unc['sigma_rel_T']*100:.3f}",f"{a_T:.4e}",
                        "1" if in_fit else "0"])
    print(f"  CSV => {csv_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE TUNER
# ═══════════════════════════════════════════════════════════════════════════════

def launch_interactive_tuner(nuclide, barrier,
                              thicknesses, transmissions, unc_results,
                              auto_alpha, auto_beta, auto_gamma,
                              fit_mask, output_dir,
                              alpha_tail, r2_tail, alpha_tol,
                              sd_alpha=float("nan"),
                              sd_beta=float("nan"),
                              sd_gamma=float("nan")):
    pub   = TABLE2.get((nuclide,barrier))
    color = BARRIER_COLORS.get(barrier,"#333333")
    mu_nist = MU_NARROW_NIST.get((nuclide,barrier))
    ekev    = NUCLIDE_ENERGY_KEV.get(nuclide,"?")

    t_arr = np.asarray(thicknesses)
    T_arr = np.asarray(transmissions)
    used  = fit_mask if fit_mask is not None else np.ones(len(t_arr),bool)
    excl  = ~used & (T_arr>0)
    err   = np.array([u.get("sigma_abs_T",np.nan) for u in unc_results])
    t_sm  = np.linspace(0, t_arr.max()*1.08, 1200)

    # Tail highlight
    valid_m = (T_arr>0)&np.isfinite(T_arr)
    tail_n  = min(DEFAULT_ALPHA_TAIL_N, int(valid_m.sum()))
    tail_min= np.sort(t_arr[valid_m])[-tail_n] if tail_n>0 else t_arr.max()

    # Slider ranges
    a_span = max(alpha_tail*0.5, alpha_tail*alpha_tol*3)
    a_min  = max(1e-6, alpha_tail-a_span); a_max = alpha_tail+a_span
    b_span = max(5.0*alpha_tail, abs(auto_beta)*1.5)
    b_min,b_max = -b_span,b_span
    g_min,g_max = 1e-4,GAMMA_MAX_FIT

    fig = plt.figure(figsize=(18,10)); fig.patch.set_facecolor("#f2f4f8")
    gs  = gridspec.GridSpec(1,2,figure=fig,left=0.04,right=0.99,
                            top=0.97,bottom=0.03,wspace=0.03,
                            width_ratios=[2.55,1.0])
    gs_l = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs[0],
               height_ratios=[3.4,1.5,1.1],hspace=0.10)
    ax_main  = fig.add_subplot(gs_l[0])
    ax_resid = fig.add_subplot(gs_l[1],sharex=ax_main)
    fig.add_subplot(gs_l[2]).set_visible(False)
    gs_r = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1],
               height_ratios=[1.2,1.0],hspace=0.25)
    ax_fvl   = fig.add_subplot(gs_r[0])
    ax_const = fig.add_subplot(gs_r[1])
    for ax in (ax_fvl,ax_const):
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)

    L,W,H = 0.06,0.565,0.022
    sl_a = Slider(fig.add_axes([L,0.135,W,H]),"alpha (tail slope, mm^-1)",
                  a_min,a_max,valinit=auto_alpha,color="#4a90d9")
    sl_b = Slider(fig.add_axes([L,0.100,W,H]),"beta  (shape, free)",
                  b_min,b_max,valinit=auto_beta, color="#e06c4a")
    sl_g = Slider(fig.add_axes([L,0.065,W,H]),"gamma (beam-hardening)",
                  g_min,g_max,valinit=auto_gamma,color="#4aae6e")
    for sl in (sl_a,sl_b,sl_g):
        sl.label.set_fontsize(9); sl.label.set_fontfamily("monospace")
        sl.valtext.set_fontsize(8.5); sl.valtext.set_fontfamily("monospace")

    # alpha_tail reference tick and tolerance band
    if a_min < alpha_tail < a_max:
        nx = (alpha_tail-a_min)/(a_max-a_min)
        sl_a.ax.axvline(nx,color="#cc0000",lw=2.0,alpha=0.75,zorder=6)
        sl_a.ax.text(nx,1.15,"alpha_tail",transform=sl_a.ax.transAxes,
                     ha="center",va="bottom",fontsize=7,
                     color="#cc0000",fontweight="bold",clip_on=False)
    lo_f = max((alpha_tail*(1-alpha_tol)-a_min)/(a_max-a_min),0)
    hi_f = min((alpha_tail*(1+alpha_tol)-a_min)/(a_max-a_min),1)
    sl_a.ax.axvspan(lo_f,hi_f,alpha=0.12,color="#cc0000",zorder=1)

    BTN = dict(color="#dde4f0",hovercolor="#b8ccec")
    by  = 0.018
    btn_auto = Button(fig.add_axes([0.06,by,0.12,0.030]),"Reset Auto",**BTN)
    btn_pub  = Button(fig.add_axes([0.19,by,0.12,0.030]),"Reset Pub", **BTN)
    btn_save = Button(fig.add_axes([0.32,by,0.10,0.030]),"Save Params",**BTN)
    btn_snap = Button(fig.add_axes([0.43,by,0.10,0.030]),"Save Figure",**BTN)
    for b2 in (btn_auto,btn_pub,btn_save,btn_snap): b2.label.set_fontsize(8.5)
    if pub is None: btn_pub.ax.set_alpha(0.35); btn_pub.label.set_color("#999")

    def _safe(x,a,b,g):
        try:
            y = archer_transmission(np.asarray(x,float),a,b,g)
            return np.where(np.isfinite(y)&(y>0)&(y<=2.0),y,np.nan)
        except: return None

    def _draw_main(a,b,g):
        ax=ax_main; ax.cla(); ax.set_facecolor("#fafbfc")
        ax.axvspan(tail_min,t_arr.max()*1.1,alpha=0.07,color="#cc0000",
                   zorder=0,label=f"alpha tail ({tail_n} pts)")
        he = ~np.isnan(err)&used; ne = np.isnan(err)&used
        if he.any():
            ax.errorbar(t_arr[he],T_arr[he],yerr=err[he],fmt="o",
                        color=color,ms=5.5,capsize=3.5,elinewidth=1,
                        zorder=5,label="Sim +/- sigma")
        if ne.any():
            ax.semilogy(t_arr[ne],T_arr[ne],"o",color=color,ms=5.5,
                        zorder=5,label="Sim")
        if excl.any():
            ax.semilogy(t_arr[excl],T_arr[excl],"s",color=color,ms=5,
                        alpha=0.28,zorder=3,label="Excluded")
        yc=_safe(t_sm,a,b,g)
        if yc is not None:
            ax.semilogy(t_sm,yc,"-",color=color,lw=2.4,zorder=6,
                        label="Manual (ODR)")
        ya=_safe(t_sm,auto_alpha,auto_beta,auto_gamma)
        if ya is not None:
            ax.semilogy(t_sm,ya,"--",color="#888",lw=1.4,alpha=0.75,
                        zorder=3,label="Auto-fit")
        if pub:
            yp=_safe(t_sm,*pub)
            if yp is not None:
                ax.semilogy(t_sm,yp,":",color="#333",lw=1.3,alpha=0.7,
                            zorder=3,label="Table 2")
        # Asymptote from current alpha
        valid_T = T_arr[T_arr>0]; valid_t = t_arr[T_arr>0]
        if len(valid_t)>0:
            C = valid_T[-1]*math.exp(a*valid_t[-1])
            ax.semilogy(t_sm,C*np.exp(-a*t_sm),"-.",color="#cc0000",
                        lw=1.0,alpha=0.5,zorder=2,
                        label=f"Asymptote a={a:.5f}")
        for lbl2,Tv in FVL_TARGETS.items():
            ax.axhline(Tv,lw=0.5,ls=":",color="#ccc",zorder=1)
            ax.text(t_arr.max()*1.005,Tv,lbl2,va="center",
                    fontsize=7,color="#aaa")
        r2s = f"  R2_tail={r2_tail:.4f}" if r2_tail else ""
        ax.set_title(
            f"{nuclide}  {barrier}\n"
            f"a={a:.6f}  b={b:.6f}  g={g:.6f}"
            f"    a_tail={alpha_tail:.6f}{r2s}",
            fontsize=9.5,pad=6)
        ax.set_ylabel("Transmission T",fontsize=10)
        ax.set_xlim(0, t_arr.max() * 1.10)
        ax.set_ylim(1e-5,2.0); ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.legend(fontsize=8,loc="upper right",framealpha=0.88)
        ax.grid(True,which="both",ls=":",alpha=0.35)
        plt.setp(ax.get_xticklabels(),visible=False)

    def _draw_resid(a,b,g):
        ax=ax_resid; ax.cla(); ax.set_facecolor("#fafbfc")
        ax.axhline(0,color="#555",lw=0.9,zorder=2)
        ax.axvspan(tail_min,t_arr.max()*1.1,alpha=0.07,color="#cc0000",zorder=0)
        valid2=T_arr>0
        if valid2.any():
            try:
                T_fit=archer_transmission(t_arr[valid2],a,b,g)
                pct=(T_fit-T_arr[valid2])/T_arr[valid2]*100
                cols=[color if used[i] else "#ccc" for i in np.where(valid2)[0]]
                ax.bar(t_arr[valid2],pct,width=t_arr.max()*0.016,
                       color=cols,alpha=0.8,zorder=3,
                       edgecolor="white",linewidth=0.4)
            except: pass
        for lv,lc,la in [(5,"#dd8800",0.6),(10,"#cc2222",0.45)]:
            ax.axhline(+lv,ls=":",color=lc,lw=0.9,alpha=la)
            ax.axhline(-lv,ls=":",color=lc,lw=0.9,alpha=la)
            ax.text(0,lv+0.4,f"+{lv}%",fontsize=6.5,color=lc,alpha=la)
        ax.set_ylabel("Residual %",fontsize=8.5)
        ax.set_xlabel("Barrier Thickness (mm)",fontsize=10)
        ax.grid(True,which="major",ls=":",alpha=0.35)
        ax.tick_params(axis="both",labelsize=8)

    def _draw_fvl(a,b,g):
        ax=ax_fvl; ax.cla()
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        ax.set_facecolor("#f0f4fa")
        fvl_rows = compute_fvl_with_local(t_arr,T_arr,a,b,g)
        all_ok   = all_fvl_accepted(fvl_rows)
        rmse,mxp = compute_fit_quality(t_arr[used],T_arr[used],a,b,g)

        sd_s = ""
        if all(math.isfinite(v) for v in (sd_alpha,sd_beta,sd_gamma)):
            sd_s = (f"\n  sd_a={sd_alpha:.2e}  "
                    f"sd_b={sd_beta:.2e}  sd_g={sd_gamma:.2e}")

        lines = [f"  -- ODR Parameters --\n",
                 f"  a = {a:.7f} mm^-1\n",
                 f"  b = {b:.7f}\n",
                 f"  g = {g:.7f}\n"]
        if sd_s: lines.append(f"{sd_s}\n")
        r2s = f"  R2={r2_tail:.4f}" if r2_tail else ""
        lines += [f"\n  -- Alpha from tail --\n",
                  f"  a_tail = {alpha_tail:.6f} mm^-1{r2s}\n",
                  f"  a/a_tail = {a/alpha_tail:.4f}\n",
                  f"  (+/-{alpha_tol*100:.0f}% band on slider)\n"]
        if mu_nist:
            lines += [f"\n  -- NIST mu (info only) --\n",
                      f"  mu={mu_nist:.6f} mm^-1 @ {ekev} keV\n",
                      f"  a_tail/mu={alpha_tail/mu_nist:.3f}\n"]
        if pub:
            lines += [f"\n  -- Table 2 --\n",
                      f"  a={pub[0]:.6f}  b={pub[1]:.6f}\n",
                      f"  g={pub[2]:.6f}\n"]
        accept_hdr = "ALL ACCEPTED (<=10%)" if all_ok else "SOME REJECTED"
        lines += [f"\n  -- FVL (mm) [{accept_hdr}] --\n",
                  f"  {'Lyr':<5}  {'Archer':>7}  {'Local':>7}  "
                  f"{'Meth':>4}  {'D%':>5}\n"]
        for r in fvl_rows:
            ok_c = "v" if r["accepted"] else "X"
            lines.append(
                f"  {r['label']:<5}  {r['x_archer']:>7.2f}  "
                f"{r['x_local']:>7.2f}  {r['method']:>4}  "
                f"{r['delta_pct']:>4.1f}% {ok_c}\n")

        rmse_s = f"{rmse:.5f}" if math.isfinite(rmse) else "n/a"
        mxp_s  = f"{mxp:.1f}%" if math.isfinite(mxp)  else "n/a"
        lines += [f"\n  RMSE(log10) = {rmse_s}\n",
                  f"  Max|resid|  = {mxp_s}\n"]

        ax.text(0.03,0.98,"".join(lines),transform=ax.transAxes,
                va="top",ha="left",fontsize=7.6,fontfamily="monospace",
                color="#1a1a1a",linespacing=1.3)
        ax.set_title("Parameters & FVL  (Section 2.4)",fontsize=9,pad=5,
                     fontweight="bold",color="#334466")

    def _draw_const(a,b,g):
        ax=ax_const; ax.cla()
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        items  = check_constraints(a,b,g,alpha_tail,alpha_tol)
        n_fail = sum(1 for _,ok2,_ in items if not ok2)
        all_ok2= n_fail==0
        ax.set_facecolor("#f0faf2" if all_ok2 else "#fef5f0")
        hc = "#1a7a1a" if all_ok2 else "#b83010"
        ht = ("All constraints OK" if all_ok2
              else f"{n_fail} violation{'s' if n_fail>1 else ''}")
        ax.text(0.5,0.97,ht,transform=ax.transAxes,ha="center",va="top",
                fontsize=9,fontweight="bold",color=hc)
        y2=0.85
        for name,ok2,det in items:
            if y2<0.02: break
            ic="v" if ok2 else "X"; cc="#1a8a1a" if ok2 else "#cc2200"
            ax.text(0.05,y2,f"{ic}  {name}",transform=ax.transAxes,
                    ha="left",va="top",fontsize=8.2,
                    fontfamily="monospace",color=cc)
            y2-=0.10
            if not ok2:
                ax.text(0.10,y2,det,transform=ax.transAxes,ha="left",
                        va="top",fontsize=6.8,fontfamily="monospace",
                        color="#884422")
                y2-=0.06
        ax.set_title("Physical Constraints",fontsize=9,pad=5,
                     fontweight="bold",color="#334466")

    _lock=[False]
    def redraw(_=None):
        if _lock[0]: return
        a=max(sl_a.val,1e-9); b=sl_b.val; g=max(sl_g.val,1e-9)
        _draw_main(a,b,g); _draw_resid(a,b,g)
        _draw_fvl(a,b,g);  _draw_const(a,b,g)
        fig.canvas.draw_idle()

    sl_a.on_changed(redraw); sl_b.on_changed(redraw); sl_g.on_changed(redraw)

    def _set(a,b,g):
        _lock[0]=True
        sl_a.set_val(float(np.clip(a,a_min,a_max)))
        sl_b.set_val(float(np.clip(b,b_min,b_max)))
        sl_g.set_val(float(np.clip(g,g_min,g_max)))
        _lock[0]=False; redraw()

    def reset_auto(_): _set(auto_alpha,auto_beta,auto_gamma)
    def reset_pub(_):
        if pub: _set(*pub)

    def save_params(_):
        a2,b2,g2=sl_a.val,sl_b.val,sl_g.val
        output_dir.mkdir(parents=True,exist_ok=True)
        log = output_dir/f"{nuclide}_{barrier}_manual_params.txt"
        fvl_rows = compute_fvl_with_local(t_arr,T_arr,a2,b2,g2)
        rm2,mp2 = compute_fit_quality(t_arr[used],T_arr[used],a2,b2,g2)
        with open(log,"a") as fp:
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.write(f"# {ts}  {nuclide}/{barrier}\n")
            fp.write(f"alpha={a2:.8f}  beta={b2:.8f}  gamma={g2:.8f}\n")
            fp.write(f"alpha_tail={alpha_tail:.8f}  r2_tail={r2_tail:.5f}  "
                     f"a/a_tail={a2/alpha_tail:.5f}\n")
            if mu_nist:
                fp.write(f"mu_NIST={mu_nist:.6f}  "
                         f"a_tail/mu={alpha_tail/mu_nist:.4f}\n")
            fp.write(f"rmse_log10={rm2:.6f}  max_resid_pct={mp2:.2f}\n")
            fp.write(f"FVL_accept={'all' if all_fvl_accepted(fvl_rows) else 'some_rejected'}\n")
            for r in fvl_rows:
                fp.write(f"  {r['label']}: archer={r['x_archer']:.4f}mm  "
                         f"local={r['x_local']:.4f}mm  "
                         f"delta={r['delta_pct']:.2f}%  "
                         f"accept={r['accepted']}\n")
            fp.write("\n")
        print(f"\n  Saved => {log}")
        old=ax_main.get_title()
        ax_main.set_title(f"Saved  a={a2:.6f}  b={b2:.6f}  g={g2:.6f}",
                          fontsize=10,color="green")
        fig.canvas.draw_idle()
        def _r():
            time.sleep(2); 
            try: ax_main.set_title(old,fontsize=9.5,color="black"); fig.canvas.draw_idle()
            except: pass
        threading.Thread(target=_r,daemon=True).start()

    def save_figure(_):
        a2,b2,g2=sl_a.val,sl_b.val,sl_g.val
        output_dir.mkdir(parents=True,exist_ok=True)
        fn=output_dir/f"{nuclide}_{barrier}_tuned_{a2:.5f}_{b2:.5f}_{g2:.4f}.png"
        fig.savefig(fn,dpi=150,bbox_inches="tight")
        print(f"  Saved figure => {fn}")

    btn_auto.on_clicked(reset_auto); btn_pub.on_clicked(reset_pub)
    btn_save.on_clicked(save_params); btn_snap.on_clicked(save_figure)

    fig.text(0.06,0.152,
             f"Alpha from data tail (last {DEFAULT_ALPHA_TAIL_N} pts, "
             f"R2={r2_tail:.4f})  |  "
             f"a_tail={alpha_tail:.6f} mm^-1  |  "
             f"Red band = +/-{alpha_tol*100:.0f}% tol  |  "
             f"Beta FREE (no NIST anchor)  |  "
             f"ODR weights: sx={0.5} mm, sy=sigma_T",
             fontsize=7.5,color="#440000",style="italic")

    redraw(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# NON-INTERACTIVE PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_static(nuclide, barrier, thicknesses, transmissions,
                alpha, beta, gamma, output_dir, unc_results=None, fit_mask=None):
    fig,ax = plt.subplots(figsize=(8,5))
    T_arr  = np.asarray(transmissions); t_arr=np.asarray(thicknesses)
    used   = fit_mask if fit_mask is not None else np.ones(len(T_arr),bool)
    col    = BARRIER_COLORS.get(barrier,"k")
    t_sm   = np.linspace(0,t_arr.max()*1.05,1000)
    if unc_results:
        err=np.array([u.get("sigma_abs_T",np.nan) for u in unc_results])
        he=~np.isnan(err)&used; ne=np.isnan(err)&used
        if he.any():
            ax.errorbar(t_arr[he],T_arr[he],yerr=err[he],fmt="o",
                        color=col,ms=5,capsize=3,elinewidth=1,label="Sim +/- s")
        if ne.any():
            ax.semilogy(t_arr[ne],T_arr[ne],"o",color=col,ms=5,label="Sim")
    else:
        ax.semilogy(t_arr[used],T_arr[used],"o",color=col,ms=5,label="Sim")
    ax.semilogy(t_sm,archer_transmission(t_sm,alpha,beta,gamma),
                "-",color=col,label="ODR fit",lw=1.8)
    ref=TABLE2.get((nuclide,barrier))
    if ref:
        ax.semilogy(t_sm,archer_transmission(t_sm,*ref),
                    "--",color="grey",label="Table 2",lw=1.2,alpha=0.7)
    for lbl2,Tv in FVL_TARGETS.items():
        ax.axhline(Tv,lw=0.5,ls=":",color="silver")
        ax.text(t_arr.max()*1.01,Tv,lbl2,va="center",fontsize=7,color="grey")
    ax.set_xlabel("Barrier Thickness (mm)",fontsize=12)
    ax.set_ylabel("Transmission Factor T",fontsize=12)
    ax.set_title(f"{nuclide}  {barrier}  [{int(used.sum())}/{int((T_arr>0).sum())} pts]")
    ax.set_ylim(1e-5,2.0)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.legend(fontsize=9); ax.grid(True,which="both",ls=":",alpha=0.4)
    fig.tight_layout()
    out=output_dir/f"{nuclide}_{barrier}_transmission.png"
    fig.savefig(out,dpi=150); plt.close(fig); print(f"  Plot => {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# WORKED EXAMPLE  Lu-177 (Section 4.1)
# ═══════════════════════════════════════════════════════════════════════════════

def lu177_room_example():
    a,b,g = TABLE2[("Lu177","Lead")]
    wl=5*200*4; d=240.0; lim=20.0
    D_unsh = wl*0.181*0.957e-2/d**2*1e6
    T_req  = lim/D_unsh
    x_lead = archer_thickness(T_req,a,b,g)
    print("\n"+"="*60)
    print("  Lu-177 DOTATATE Treatment Room  (Section 4.1)")
    print("="*60)
    print(f"  Workload    : {wl:,.0f} mCi.h/week")
    print(f"  Distance    : {d:.0f} cm")
    print(f"  Unshielded  : {D_unsh:.1f} uGy/week")
    print(f"  Required T  : {T_req:.4f}")
    print(f"  Pb thickness: {x_lead:.2f} mm  (paper: 1.48 mm)")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# N-PRIMARIES ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_estimate_n(nuclide, barrier, output_dir,
                   target_unc=DEFAULT_TARGET_UNC):
    print(f"\n{'='*60}")
    print(f"  N-Primaries Estimator   target sigma_T = {target_unc*100:.1f}%")
    print(f"  {nuclide} / {barrier}")
    print(f"{'='*60}")
    (t,T,_,_,unc_results,n_estimates,_) = collect_transmission(
        nuclide,barrier,output_dir,target_unc=target_unc)
    for ti,Ti,unc,n_est in zip(t,T,unc_results,n_estimates):
        a_T=unc["sigma_abs_T"]; s_T=unc["sigma_rel_T"]
        print(f"\n  t={ti:.3g}mm  T={Ti:.5f}  "
              f"sigma_T={s_T*100:.2f}%  "
              f"N_needed={n_est.get('N_needed','n/a')}")
    n_all=[e["N_needed"] for e in n_estimates if e.get("N_needed")]
    if n_all:
        n_max=max(n_all); t_max=t[[e.get("N_needed",0) for e in n_estimates].index(n_max)]
        print(f"\n  Worst case: t={t_max:.3g}mm  N_needed={n_max:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_one(nuclide, barrier, output_dir,
                interactive=True, make_plot=True,
                target_unc=DEFAULT_TARGET_UNC,
                fit_points=None, fit_min_T=None,
                alpha_tail_n=DEFAULT_ALPHA_TAIL_N,
                alpha_tol=DEFAULT_ALPHA_TOL,
                thickness_unc_mm=0.5):

    (thicknesses,transmissions,doses_barrier,dose_air,
     unc_results,n_estimates,sigma_T) = collect_transmission(
        nuclide,barrier,output_dir,target_unc=target_unc)

    (alpha,beta,gamma,
     sd_a,sd_b,sd_g,
     fit_mask,alpha_tail,r2_tail,_) = fit_archer_full(
        thicknesses,transmissions,sigma_T,
        n_points=fit_points,min_T=fit_min_T,
        alpha_tail_n=alpha_tail_n,alpha_tol=alpha_tol,
        thickness_unc_mm=thickness_unc_mm,
        nuclide=nuclide,barrier=barrier)

    fvl_rows = compute_fvl_with_local(thicknesses,transmissions,
                                       alpha,beta,gamma)
    print_fvl_table(fvl_rows,nuclide,barrier,alpha,beta,gamma)

    write_transmission_csv(nuclide,barrier,thicknesses,transmissions,
                           doses_barrier,dose_air,alpha,beta,gamma,
                           unc_results,n_estimates,output_dir,fit_mask)

    if interactive:
        launch_interactive_tuner(
            nuclide,barrier,thicknesses,transmissions,unc_results,
            alpha,beta,gamma,fit_mask,output_dir,
            alpha_tail,r2_tail,alpha_tol,sd_a,sd_b,sd_g)
    elif make_plot:
        plot_static(nuclide,barrier,thicknesses,transmissions,
                    alpha,beta,gamma,output_dir,unc_results,fit_mask)

    return alpha,beta,gamma,fvl_rows


def analyze_all(output_dir, target_unc=DEFAULT_TARGET_UNC,
                alpha_tail_n=DEFAULT_ALPHA_TAIL_N,
                alpha_tol=DEFAULT_ALPHA_TOL):
    rows=[]
    for nuclide in ["Lu177","Tc99m","I131","F18","Zr89"]:
        for barrier in ["Lead","LWConcrete","NWConcrete","Steel","Glass","Gypsum"]:
            if not (list(output_dir.glob(f"{nuclide}_{barrier}_*mm_dose.mhd"))+
                    list(output_dir.glob(f"{nuclide}_{barrier}_*mm_edep.mhd"))):
                continue
            try:
                a,b,g,fvl_rows = analyze_one(
                    nuclide,barrier,output_dir,interactive=False,
                    target_unc=target_unc,
                    alpha_tail_n=alpha_tail_n,alpha_tol=alpha_tol)
                row={"nuclide":nuclide,"barrier":barrier,
                     "alpha":a,"beta":b,"gamma":g,
                     "all_accepted":all_fvl_accepted(fvl_rows)}
                for r in fvl_rows:
                    row[f"{r['label']}_archer"]=r["x_archer"]
                    row[f"{r['label']}_local"] =r["x_local"]
                    row[f"{r['label']}_delta%"]=r["delta_pct"]
                rows.append(row)
            except Exception as e:
                print(f"  X {nuclide}/{barrier}: {e}")
    if rows:
        out=output_dir/"archer_parameters_summary.csv"
        with open(out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\nSummary => {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p=argparse.ArgumentParser(
        description="Oumano 2025 JACMP — ODR Archer fit + interactive tuner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_interactive.py --nuclide F18 --barrier Lead
  python analyze_interactive.py --nuclide I131 --barrier LWConcrete --fit-min-T 0.01
  python analyze_interactive.py --nuclide F18 --barrier Lead --alpha-tail-n 4
  python analyze_interactive.py --all --no-interactive
  python analyze_interactive.py --example
        """)
    p.add_argument("--nuclide",default="F18",
                   choices=["F18","Tc99m","I131","Lu177","Zr89"])
    p.add_argument("--barrier",default="Lead",
                   choices=["Lead","LWConcrete","NWConcrete",
                             "Steel","Glass","Gypsum"])
    p.add_argument("--output",default="output",
                   help="Directory with .mhd files  (default: output/)")
    p.add_argument("--target-unc",type=float,default=DEFAULT_TARGET_UNC)
    p.add_argument("--fit-points",type=int,default=None)
    p.add_argument("--fit-min-T",type=float,default=None)
    p.add_argument("--alpha-tail-n",type=int,default=DEFAULT_ALPHA_TAIL_N,
                   help=f"Tail points for alpha OLS (default {DEFAULT_ALPHA_TAIL_N})")
    p.add_argument("--alpha-tol",type=float,default=DEFAULT_ALPHA_TOL,
                   help=f"Alpha tolerance band (default {DEFAULT_ALPHA_TOL})")
    p.add_argument("--thickness-unc",type=float,default=0.5,
                   help="ODR thickness uncertainty sigma_x in mm (default 0.5)")
    p.add_argument("--estimate-n",action="store_true")
    p.add_argument("--all",action="store_true")
    p.add_argument("--example",action="store_true")
    p.add_argument("--no-interactive",action="store_true")
    p.add_argument("--no-plot",action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); output_dir=Path(args.output)
    if args.example:      lu177_room_example(); return
    if args.estimate_n:   run_estimate_n(args.nuclide,args.barrier,
                                         output_dir,args.target_unc); return
    if args.all:
        analyze_all(output_dir,args.target_unc,
                    args.alpha_tail_n,args.alpha_tol); return
    analyze_one(args.nuclide,args.barrier,output_dir,
                interactive=not args.no_interactive,
                make_plot=not args.no_plot,
                target_unc=args.target_unc,
                fit_points=args.fit_points,
                fit_min_T=args.fit_min_T,
                alpha_tail_n=args.alpha_tail_n,
                alpha_tol=args.alpha_tol,
                thickness_unc_mm=args.thickness_unc)

if __name__=="__main__":
    main()
