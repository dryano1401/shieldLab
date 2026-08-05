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

Compatibility note (no functional change made here)
-----------------------------------------------------
shieldLabSim.py's zero-flag defaults now reproduce Oumano's Dose Actor
scoring position directly: a slab centred 10 mm from the tissue face (4
planes, so arr[1:3] = 5-15 mm depth), 250x250 mm footprint (100x100 voxels
at 2.5 mm pitch) — comfortably larger than the 150 mm ROI, but deliberately
NOT the paper's literal full 2 m x 2 m block face. A full-footprint (800x800)
config was tried and measurably lowered T at matched depth (e.g. NW concrete
at 78.6 mm: 0.452 -> 0.399) despite the ROI mask below selecting the same
central voxels either way in principle; root cause not yet confirmed
(suspected interaction between --unc-goal's per-voxel early-stop and a
mostly-empty large array), so 250x250 mm is the supported default. Nothing in
THIS file changed either way: _is_original_actor() detects the actor from
its own array shape (nY>=100 and nX>=100 and nZ>=4 — true for 100x100x4,
800x800x4, or anything else that size) and _build_roi_mask() centres its
150 mm circle from the array's actual dimensions, so both configurations (and
anything explicitly overridden via --detector-size-x/y) are handled
correctly by this file regardless. Do not mix .mhd files scored at different
depths in one collect_transmission() directory — a run from before the
depth fix, or one made with --detector-centered, sits at ~240-260 mm and is
not comparable to files scored at 5-15 mm.
"""

import argparse, csv, json, math, sys, threading, time, warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for pop-up tuner window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit, minimize, least_squares, NonlinearConstraint
from scipy.odr import ODR, Model, RealData

# GSA (Grouped Spectral Archer) fitting -- optional sibling module
# (gsa_fit.py + nist_xcom_data.py). Soft-imported the same way this file's
# own GUI counterpart soft-imports shieldLabAnalyze itself (see AI_OK in
# shieldLabGUI.py) -- if gsa_fit.py isn't deployed alongside this file yet,
# every other fit method keeps working and only fit_method="gsa" degrades
# to a clear error instead of crashing the whole module at import time.
try:
    import gsa_fit as _gsa
    GSA_OK = True
except Exception:
    _gsa = None
    GSA_OK = False

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


def parse_fvl_layer_weights(spec):
    """
    Parse a CLI/GUI-facing per-layer FVL weight string into the dict form
    fit_archer_fvl_optimized()/fit_archer_odr_fvl_blend() expect (see
    fit_archer_full()'s fvl_layer_weights docstring).

    Format: comma-separated "LABEL=weight" pairs, e.g. "HVL=1,CVL=0.5".
    A layer named but omitted entirely from `spec` is NOT included in the
    returned dict (equivalent to weight 0 -- excluded from the objective),
    matching the "anchor to just the layer(s) you care about" use case
    (e.g. spec="HVL=1" optimizes purely for HVL agreement).

    Returns None if spec is None/empty/whitespace-only (caller's default
    "weight every resolvable layer equally" behavior applies). Raises
    ValueError on an unrecognized label or a value that doesn't parse as a
    float, so a typo fails loudly at parse time rather than silently
    dropping/misweighting a layer.
    """
    if spec is None or not spec.strip():
        return None
    weights = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid FVL layer weight entry {part!r} -- expected "
                f"LABEL=weight, e.g. 'HVL=1' or 'HVL=1,CVL=0.5'.")
        label, val = part.split("=", 1)
        label = label.strip().upper()
        if label not in FVL_TARGETS:
            raise ValueError(
                f"Unknown FVL layer {label!r} in weight spec {spec!r} -- "
                f"expected one of {sorted(FVL_TARGETS)}.")
        try:
            weights[label] = float(val.strip())
        except ValueError:
            raise ValueError(
                f"Invalid weight {val!r} for layer {label!r} in spec "
                f"{spec!r} -- expected a number.")
    if not weights:
        return None
    return weights

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
# ALTERNATE METHOD — STANDARD 3-PARAMETER FIT (plain weighted NLS, no alpha
# pinning, no tail pre-fit, no ODR x-uncertainty). Added as a selectable
# alternative to fit_archer_odr() above -- same archer_transmission() model,
# same (alpha,beta,gamma) parameterization, but fit as an ordinary generic
# nonlinear least-squares problem with all three parameters free and no
# multi-start grid: standard scipy.optimize.curve_fit (Levenberg-Marquardt),
# weighted by sigma_abs_T on y only (thickness treated as exact, no sigma_x).
# Useful as an independent cross-check against the paper-matched ODR method,
# since it makes no assumption that alpha must sit near the data-tail slope.
# ═══════════════════════════════════════════════════════════════════════════════

def fit_archer_standard(thicknesses, transmissions, sigma_T,
                         p0=None, gamma_max=GAMMA_MAX_FIT):
    """
    Standard (non-ODR, non-alpha-pinned) 3-parameter Archer fit.

    Plain scipy.optimize.curve_fit (Levenberg-Marquardt) on
    T = archer_transmission(x, alpha, beta, gamma), all three parameters
    free and fit simultaneously from a single generic starting guess --
    no data-tail alpha pre-fit, no alpha_tol bounding, no ODR treatment of
    thickness uncertainty, no multi-start grid search. This is the
    "textbook" way to fit a 3-parameter nonlinear model and is offered as
    an independent alternative to fit_archer_odr()'s paper-matched
    methodology, for cross-checking -- the two can disagree, particularly
    when the data has a long, sparsely-sampled tail (a single global fit
    then has to trade off fit quality between the well-sampled shallow
    region and the sparse deep region, exactly the situation that motivated
    adding this as a visible alternative rather than silently trusting one
    method).

    Parameters
    ----------
    thicknesses, transmissions : arrays
    sigma_T : array
        Absolute sigma_T per point (same array used elsewhere in this file,
        e.g. sigma_abs_T from write_transmission_csv/collect_transmission).
        Falls back to 1% of T where missing/nan/zero, same convention as
        fit_archer_odr().
    p0 : (alpha0, beta0, gamma0), optional
        Starting guess. Defaults to (0.02, -0.01, 1.0) -- generic, not
        derived from this data's tail.
    gamma_max : float
        Upper bound on gamma (kept finite for numerical stability, matches
        fit_archer_odr()'s GAMMA_MAX_FIT cap).

    Returns
    -------
    alpha, beta, gamma, sd_alpha, sd_beta, sd_gamma, pcov
    """
    x = np.asarray(thicknesses, float)
    y = np.asarray(transmissions, float)
    sig = np.asarray(sigma_T, float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.01 * y)
    sig = np.where(sig > 0, sig, 1e-6)

    valid = (y > 0) & np.isfinite(y)
    x, y, sig = x[valid], y[valid], sig[valid]
    if len(x) < 4:
        raise ValueError(
            f"Only {len(x)} valid point(s) — need >= 4 for a free "
            "3-parameter fit (3 parameters + at least 1 degree of freedom).")

    if p0 is None:
        p0 = (0.02, -0.01, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(
            archer_transmission, x, y, p0=p0,
            bounds=([1e-6, -50.0, 1e-4], [50.0, 50.0, gamma_max]),
            sigma=sig, absolute_sigma=True, maxfev=200_000)

    a, b, g = popt
    sd = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else (float("nan"),)*3
    sd_a, sd_b, sd_g = sd

    y_fit = archer_transmission(x, a, b, g)
    ok = (y_fit > 0) & np.isfinite(y_fit)
    rmse = (float(np.sqrt(np.mean((np.log10(y_fit[ok])-np.log10(y[ok]))**2)))
            if ok.any() else float("nan"))

    print(f"\n  Standard fit (plain weighted NLS, no alpha pinning): "
          f"log-RMSE = {rmse:.5f}")
    print(f"  alpha={a:.6f}  beta={b:.6f}  gamma={g:.6f}")

    return a, b, g, sd_a, sd_b, sd_g, pcov


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


def compute_fvl_with_local_gsa(thicknesses, transmissions, gsa_params,
                                n_bracket=3):
    """
    GSA analogue of compute_fvl_with_local(): FVL thicknesses via
      (a) GSA (weighted-sum-of-Archer-terms) inversion -- gsa_fit.gsa_thickness()
      (b) the SAME local-bracketing ground truth used for the standard
          Archer comparison (local_fvl()), so "x_local" here is identical
          to the standard table's -- only the model-side column differs.

    Returns [] if gsa_params is falsy (no GSA fit available for this
    material) or gsa_fit.py isn't importable -- callers should treat an
    empty list as "nothing to show", not an error.
    """
    if not gsa_params or not GSA_OK:
        return []
    rows = []
    for label, T_target in FVL_TARGETS.items():
        try:
            x_gsa = _gsa.gsa_thickness(T_target, gsa_params)
            if not math.isfinite(x_gsa) or x_gsa <= 0:
                x_gsa = float("nan")
        except Exception:
            x_gsa = float("nan")

        loc = local_fvl(thicknesses, transmissions, T_target, n_bracket)
        x_local = loc["thickness"]

        if math.isfinite(x_gsa) and math.isfinite(x_local) and x_local > 0:
            delta_pct = abs(x_gsa - x_local) / x_local * 100.0
            accepted  = delta_pct <= FVL_ACCEPT_THRESH * 100
        else:
            delta_pct = float("nan")
            accepted  = False

        rows.append({
            "label":    label,
            "T_target": T_target,
            "x_gsa":    x_gsa,
            "x_local":  x_local,
            "method":   loc["method"],
            "delta_pct":delta_pct,
            "accepted": accepted,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# PIECEWISE (THIN/THICK) ARCHER FIT -- two independently-invertible 3-param
# triples, continuity-constrained at a data-driven cutoff thickness x*
# ═══════════════════════════════════════════════════════════════════════════════
# Motivation: like GSA, this exists for materials whose transmission curve
# has a genuine two-regime shape that a single 3-parameter Archer term can't
# represent (see the Cu64/Lead investigation this was built for -- a sharp
# break in local attenuation coefficient around 30-40mm). Unlike GSA, this
# does NOT require nuclear-physics anchoring data (no PHOTON_SPECTRA lookup,
# no NIST XCOM material composition) -- it's a purely data-driven curve fit:
# grid-search the cutoff thickness x* that minimizes total fit error, then
# jointly fit a "thin" triple (x <= x*) and a "thick" triple (x > x*) with a
# hard equality constraint forcing T_thin(x*) == T_thick(x*), so the
# resulting piecewise curve has no visible jump at the seam.
#
# Trade-off vs GSA: simpler to compute and EASILY invertible (each side is
# just a plain archer_thickness() call on its own triple -- no Newton
# iteration needed), but the two triples carry no physical meaning of their
# own (nothing ties "thin alpha" or "thick alpha" to an actual emission line
# or attenuation coefficient the way GSA's per-group alpha does) -- the
# split is purely a numerical curve-fit choice, not a photon-population
# decomposition. Confirmed against real Cu64/Lead data in-session: best-RMSE
# cutoff landed close to (not identical to) GSA's physics-derived crossover
# x_c, and continuity-constrained piecewise beat GSA on raw RMSE(log10)
# while GSA still had smaller max FVL disagreement.

@dataclass
class PiecewiseFitResult:
    nuclide: str
    barrier: str
    n: int
    n_dropped: int
    x_star: float                # cutoff thickness (mm)
    thin_params: tuple           # (alpha, beta, gamma) for x <= x_star
    thick_params: tuple          # (alpha, beta, gamma) for x > x_star
    n_thin: int
    n_thick: int
    rmse_std: float              # plain single-term Archer RMSE(log10), for comparison
    rmse_piecewise: float
    maxerr_std_pct: float
    maxerr_piecewise_pct: float
    seam_continuous: bool        # True if the continuity constraint actually converged
    verdict: str                 # "piecewise needed" / "piecewise favored" / "std adequate" / "fit failed"
    std_params: tuple            # (alpha, beta, gamma) of the plain fit, for the dAICc-style comparison
    dAICc: float = float("nan")  # aicc_std - aicc_piecewise; >0 means piecewise wins after the
                                  # parameter-count penalty (6 params vs the plain fit's 3), NaN if
                                  # either side is low-DOF (see aicc()/low_dof below)
    low_dof: bool = False        # True if n is too small for AICc's correction term to be valid at
                                  # p=6 (piecewise) -- verdict falls back to plain RMSE comparison
                                  # rather than trusting a garbage AICc


def _piecewise_archer_resid(p, x_thin, lnT_thin, x_thick, lnT_thick):
    a1,b1,g1, a2,b2,g2 = p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tm_thin  = archer_transmission(x_thin,  a1,b1,g1)
        Tm_thick = archer_transmission(x_thick, a2,b2,g2)
    Tm_thin  = np.clip(Tm_thin,  1e-300, None)
    Tm_thick = np.clip(Tm_thick, 1e-300, None)
    return np.concatenate([np.log(Tm_thin)-lnT_thin, np.log(Tm_thick)-lnT_thick])


def _fit_piecewise_at_cutoff(x, y, lnT, x_star, min_side_n=3):
    """
    Fit thin/thick triples at a FIXED cutoff x_star, continuity-constrained
    at the seam (T_thin(x_star) == T_thick(x_star) exactly). Returns
    (p_thin, p_thick, sse, success) -- sse is total sum-of-squared log-space
    residuals across both sides combined, success=False (sse=inf) if either
    side has fewer than min_side_n points or the constrained solve fails.
    """
    thin_mask  = x <= x_star
    thick_mask = x >  x_star
    n_thin, n_thick = int(thin_mask.sum()), int(thick_mask.sum())
    if n_thin < min_side_n or n_thick < min_side_n:
        return None, None, np.inf, False

    x_thin, lnT_thin   = x[thin_mask],  lnT[thin_mask]
    x_thick, lnT_thick = x[thick_mask], lnT[thick_mask]

    # Unconstrained per-side starting guesses (cheap OLS slope), used only
    # to seed the constrained joint solve below.
    def _seed(xs, lnTs):
        slope = -np.polyfit(xs, lnTs, 1)[0]
        return [max(1e-6, min(9.9, slope)), 0.0, 1.0]
    p0 = np.array(_seed(x_thin, lnT_thin) + _seed(x_thick, lnT_thick))

    lo = np.array([1e-7,-10.0,1e-3]*2)
    hi = np.array([10.0, 10.0,60.0]*2)
    p0c = np.clip(p0, lo+1e-9, hi-1e-9)

    def _continuity(p):
        a1,b1,g1, a2,b2,g2 = p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T1 = archer_transmission(np.array([x_star]), a1,b1,g1)[0]
            T2 = archer_transmission(np.array([x_star]), a2,b2,g2)[0]
        if not (math.isfinite(T1) and math.isfinite(T2)) or T1<=0 or T2<=0:
            return 1e6  # far from zero -> heavily penalized, solver steers away
        return math.log(T1) - math.log(T2)

    def _objective(p):
        r = _piecewise_archer_resid(p, x_thin, lnT_thin, x_thick, lnT_thick)
        return float(np.sum(r**2))

    try:
        res = minimize(_objective, p0c, method="SLSQP",
                        bounds=list(zip(lo,hi)),
                        constraints=[{"type":"eq","fun":_continuity}],
                        options={"maxiter":500,"ftol":1e-14})
    except Exception:
        return None, None, np.inf, False

    if not res.success:
        return None, None, np.inf, False

    p_thin  = tuple(res.x[0:3])
    p_thick = tuple(res.x[3:6])
    seam_gap = abs(_continuity(res.x))
    sse = float(np.sum(_piecewise_archer_resid(
        res.x, x_thin, lnT_thin, x_thick, lnT_thick)**2))
    return p_thin, p_thick, sse, (seam_gap < 1e-4)


def _piecewise_aicc(sse, n, p):
    """
    Local copy of gsa_fit.aicc()'s exact formula -- kept in sync
    deliberately, same rationale as archer_transmission()'s duplication
    into gsa_fit.py: fit_archer_piecewise() must not depend on GSA_OK /
    gsa_fit.py being importable, since piecewise fitting needs no nuclear-
    physics anchoring data and should work even when GSA can't.

        AICc = n*ln(SSE/n) + 2p + 2p(p+1)/(n-p-1)

    Returns (aicc_value, low_dof_flag) -- low_dof=True if n-p-1 <= 0 (the
    correction term is undefined/negative-denominator; caller should not
    trust this AICc value for model comparison).
    """
    if n - p - 1 <= 0:
        return float("nan"), True
    if sse <= 0:
        sse = 1e-300
    val = n*math.log(sse/n) + 2*p + (2*p*(p+1))/(n - p - 1)
    return val, False


def fit_archer_piecewise(thicknesses, transmissions, nuclide=None, barrier=None,
                          min_side_n=3, x_star=None):
    """
    Grid-search a cutoff thickness x_star (unless explicitly given), fit
    continuity-constrained thin/thick Archer triples at that cutoff, and
    return a PiecewiseFitResult. Also fits the plain single-term Archer
    equation over the FULL curve for comparison (verdict/RMSE/maxerr), the
    same "diagnostic overlay vs plain fit" pattern fit_gsa_diagnostic() uses.

    thicknesses/transmissions should be the FULL (unmasked) arrays, matching
    fit_gsa_diagnostic()'s convention -- T<=0 points are dropped internally.

    x_star : float or None
        If given, skip the grid search and fit at this exact cutoff (e.g.
        to match a value the user already knows from a prior GSA run's
        x_c, or to reproduce a specific earlier result). If None (default),
        grid-search every interior cutoff that leaves >=min_side_n points
        on each side and keep whichever minimizes total log-space SSE.
    """
    x_all = np.asarray(thicknesses, float)
    y_all = np.asarray(transmissions, float)
    valid = y_all > 0
    n_dropped = int((~valid).sum())
    x = x_all[valid]
    y = y_all[valid]
    n = len(x)
    lnT = np.log(y)

    order = np.argsort(x)
    x_sorted = x[order]

    # --- plain single-term Archer fit over the full curve, for comparison ---
    try:
        slope = -np.polyfit(x, lnT, 1)[0]
        a0 = max(1e-6, min(9.9, slope))
    except Exception:
        a0 = 0.1
    p0_std = np.clip([a0, 0.0, 1.0], [1e-7,-10,1e-3], [10,10,60])
    def _resid_std(p):
        a,b,g = p
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tm = archer_transmission(x, a, b, g)
        Tm = np.clip(Tm, 1e-300, None)
        return np.log(Tm) - lnT
    try:
        res_std = least_squares(_resid_std, p0_std,
                                 bounds=([1e-7,-10,1e-3],[10,10,60]),
                                 method="trf", max_nfev=20000)
        std_params = tuple(res_std.x)
        sse_std = float(np.sum(res_std.fun**2))
        rmse_std = math.sqrt(sse_std/n)
        Tm_std = archer_transmission(x, *std_params)
        maxerr_std = float(np.max(np.abs(Tm_std/y - 1.0))*100)
    except Exception:
        std_params = (float("nan"),)*3
        rmse_std = maxerr_std = float("nan")

    if n < 2*min_side_n:
        return PiecewiseFitResult(
            nuclide=nuclide, barrier=barrier, n=n, n_dropped=n_dropped,
            x_star=float("nan"), thin_params=(float("nan"),)*3,
            thick_params=(float("nan"),)*3, n_thin=0, n_thick=0,
            rmse_std=rmse_std, rmse_piecewise=float("nan"),
            maxerr_std_pct=maxerr_std, maxerr_piecewise_pct=float("nan"),
            seam_continuous=False, verdict="std only (n<2*min_side_n)",
            std_params=std_params, dAICc=float("nan"), low_dof=True)

    # --- grid-search the cutoff (unless the caller pinned one) -------------
    if x_star is not None:
        candidates = [float(x_star)]
    else:
        candidates = [
            (x_sorted[i-1]+x_sorted[i])/2.0
            for i in range(min_side_n, len(x_sorted)-min_side_n+1)
        ]

    best = None  # (x_star, p_thin, p_thick, sse, n_thin, n_thick, seam_ok)
    for xs in candidates:
        p_thin, p_thick, sse, seam_ok = _fit_piecewise_at_cutoff(
            x, y, lnT, xs, min_side_n=min_side_n)
        if p_thin is None:
            continue
        n_thin = int((x <= xs).sum()); n_thick = int((x > xs).sum())
        if best is None or sse < best[3]:
            best = (xs, p_thin, p_thick, sse, n_thin, n_thick, seam_ok)

    if best is None:
        return PiecewiseFitResult(
            nuclide=nuclide, barrier=barrier, n=n, n_dropped=n_dropped,
            x_star=float("nan"), thin_params=(float("nan"),)*3,
            thick_params=(float("nan"),)*3, n_thin=0, n_thick=0,
            rmse_std=rmse_std, rmse_piecewise=float("nan"),
            maxerr_std_pct=maxerr_std, maxerr_piecewise_pct=float("nan"),
            seam_continuous=False, verdict="fit failed (no cutoff converged)",
            std_params=std_params, dAICc=float("nan"), low_dof=True)

    x_star_best, p_thin, p_thick, sse, n_thin, n_thick, seam_ok = best
    rmse_pw = math.sqrt(sse/n)
    thin_mask = x <= x_star_best
    Tm_pw = np.where(thin_mask,
                      archer_transmission(x, *p_thin),
                      archer_transmission(x, *p_thick))
    maxerr_pw = float(np.max(np.abs(Tm_pw/y - 1.0))*100)

    # --- AICc-penalized verdict (mirrors gsa_fit's dAICc gate) -------------
    # Plain Archer: p=3 free params. Piecewise: p=6 (two independent
    # 3-parameter triples; the continuity constraint removes one degree of
    # freedom in principle, but it's an equality constraint on the *model*,
    # not a reduction in fitted parameter count, so we charge the full 6 --
    # this is the conservative choice, i.e. it makes it HARDER for piecewise
    # to win, not easier.)
    aicc_std, low_dof_std = _piecewise_aicc(sse_std, n, 3) if math.isfinite(sse_std) else (float("nan"), True)
    aicc_pw, low_dof_pw = _piecewise_aicc(sse, n, 6)
    low_dof = bool(low_dof_std or low_dof_pw)
    if low_dof or not (math.isfinite(aicc_std) and math.isfinite(aicc_pw)):
        dAICc = float("nan")
        # Not enough degrees of freedom to trust AICc's small-sample
        # correction at p=6 -- fall back to the plain RMSE comparison rather
        # than silently calling it "adequate" on garbage statistics.
        verdict = ("piecewise needed (low-DOF fallback)"
                   if math.isfinite(rmse_std) and rmse_pw < rmse_std
                   else "std adequate")
    else:
        dAICc = aicc_std - aicc_pw  # >0 means piecewise wins after the parameter-count penalty
        verdict = "piecewise needed" if dAICc > 0 else "std adequate"

    return PiecewiseFitResult(
        nuclide=nuclide, barrier=barrier, n=n, n_dropped=n_dropped,
        x_star=float(x_star_best), thin_params=p_thin, thick_params=p_thick,
        n_thin=n_thin, n_thick=n_thick,
        rmse_std=rmse_std, rmse_piecewise=rmse_pw,
        maxerr_std_pct=maxerr_std, maxerr_piecewise_pct=maxerr_pw,
        seam_continuous=seam_ok, verdict=verdict, std_params=std_params,
        dAICc=dAICc, low_dof=low_dof)


def archer_transmission_piecewise(x, result):
    """
    Evaluate the piecewise model at thickness(es) x, using result.thin_params
    for x <= result.x_star and result.thick_params for x > result.x_star.
    Vectorized (x can be scalar or array).
    """
    x = np.atleast_1d(np.asarray(x, float))
    thin_mask = x <= result.x_star
    out = np.empty_like(x)
    if thin_mask.any():
        out[thin_mask] = archer_transmission(x[thin_mask], *result.thin_params)
    if (~thin_mask).any():
        out[~thin_mask] = archer_transmission(x[~thin_mask], *result.thick_params)
    return out if out.shape != (1,) else out  # keep array shape consistent


def archer_thickness_piecewise(T_target, result):
    """
    EASILY-invertible piecewise inversion: no Newton iteration required
    (unlike gsa_thickness()) -- just archer_thickness() on whichever side's
    triple governs T_target, decided by comparing T_target to the model's
    own transmission value AT the seam (T_target >= T(x_star) means the
    target is in the shallow/thin regime, since T(x) is monotonically
    decreasing).
    """
    T_at_seam = archer_transmission(np.array([result.x_star]), *result.thin_params)[0]
    if T_target >= T_at_seam:
        return archer_thickness(T_target, *result.thin_params)
    else:
        return archer_thickness(T_target, *result.thick_params)


def compute_fvl_with_local_piecewise(thicknesses, transmissions, pw_result,
                                      n_bracket=3):
    """
    Piecewise analogue of compute_fvl_with_local()/compute_fvl_with_local_gsa():
    FVL thicknesses via archer_thickness_piecewise() vs the same local-
    bracketing ground truth used everywhere else. Returns [] if pw_result is
    None or has no valid thin/thick params (grid search failed / n too small).
    """
    if pw_result is None or not math.isfinite(pw_result.x_star):
        return []
    rows = []
    for label, T_target in FVL_TARGETS.items():
        try:
            x_pw = archer_thickness_piecewise(T_target, pw_result)
            if not math.isfinite(x_pw) or x_pw <= 0:
                x_pw = float("nan")
        except Exception:
            x_pw = float("nan")

        loc = local_fvl(thicknesses, transmissions, T_target, n_bracket)
        x_local = loc["thickness"]

        if math.isfinite(x_pw) and math.isfinite(x_local) and x_local > 0:
            delta_pct = abs(x_pw - x_local) / x_local * 100.0
            accepted  = delta_pct <= FVL_ACCEPT_THRESH * 100
        else:
            delta_pct = float("nan")
            accepted  = False

        rows.append({
            "label":    label,
            "T_target": T_target,
            "x_piecewise": x_pw,
            "x_local":  x_local,
            "method":   loc["method"],
            "delta_pct":delta_pct,
            "accepted": accepted,
        })
    return rows


def fit_piecewise_diagnostic(thicknesses, transmissions, nuclide, barrier, **kwargs):
    """
    Run fit_archer_piecewise() and return its PiecewiseFitResult, or None if
    the fit itself raises -- logs a clear warning either way rather than
    raising, since (like fit_gsa_diagnostic()) this is a diagnostic add-on
    and should never be the reason a normal analyze_one() run aborts.
    """
    try:
        return fit_archer_piecewise(thicknesses, transmissions, nuclide,
                                     barrier, **kwargs)
    except Exception as exc:
        warnings.warn(f"fit_piecewise_diagnostic: piecewise fit failed for "
                       f"{nuclide}/{barrier}: {exc}", RuntimeWarning)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATE METHOD 2 — FVL-OPTIMIZED FIT (alpha pinned to hardened-beam tail,
# beta/gamma chosen to directly minimize Archer-vs-local-bracket FVL disagreement)
# ═══════════════════════════════════════════════════════════════════════════════
# Neither fit_archer_odr() nor fit_archer_standard() actually optimizes the
# quantity the paper's own 10% acceptance criterion judges: how closely the
# Archer-equation HVL/QVL/TVL/CVL/MVL match the model-free local-bracketing
# thicknesses. ODR minimizes weighted (x,y) residuals across all data points;
# standard minimizes unweighted y residuals. Both can leave a curve that fits
# the bulk of the data well in an RMSE sense while still landing outside 10%
# on one or two FVL layers (see session 22's HVL/QVL-only rejection pattern).
# This method instead treats FVL agreement as the direct objective, subject to
# alpha staying pinned near the data-tail slope for the same physical reason
# fit_archer_odr() pins it: alpha is the asymptotic (hardened-beam) linear
# attenuation coefficient, a real physical quantity determined by the deep,
# well-measured region once buildup has saturated -- letting it float to chase
# FVL agreement would sacrifice that physical meaning to fix disagreement that
# properly belongs to beta/gamma (the buildup-shape parameters), which is
# exactly the failure mode this method is meant to avoid, not reproduce.

def fit_archer_fvl_optimized(thicknesses, transmissions, sigma_T,
                              alpha_tail, alpha_tol=DEFAULT_ALPHA_TOL,
                              nuclide=None, barrier=None,
                              fvl_weights=None, n_bracket=3):
    """
    Fit (alpha, beta, gamma) by directly minimizing summed relative
    disagreement between the Archer-equation FVL thicknesses and the
    model-free local-bracketing FVL thicknesses (the same comparison
    compute_fvl_with_local()/the paper's 10% acceptance test already make),
    rather than minimizing point-by-point (x,y) or y-only residuals.

    Alpha is held within the SAME physical band as fit_archer_odr():
    [alpha_tail*(1-alpha_tol), alpha_tail*(1+alpha_tol)] -- pinned near the
    hardened-beam asymptotic slope measured directly from the data tail,
    not treated as a free optimization variable. Only beta and gamma are
    actually varied by the optimizer; alpha is swept over a small grid
    within its band and the best (alpha, beta*, gamma*) triple is kept, so
    the "alpha stays physically anchored" property holds by construction,
    not just as a soft preference.

    Objective (evaluated at each candidate alpha, then argmin over beta,
    gamma via a local optimizer per alpha, then argmin over the alpha grid):

        L(alpha, beta, gamma) = sum_i  w_i * ((x_archer_i - x_local_i) / x_local_i)^2

    summed over every FVL layer (HVL/QVL/TVL/CVL/MVL) whose local-bracketing
    thickness x_local_i is finite and positive -- a layer local_fvl() can't
    resolve (e.g. too few points straddling that T_target) simply drops out
    of the sum rather than being imputed or penalized. fvl_weights lets a
    layer be emphasized/de-emphasized (default: every resolvable layer
    weighted equally at 1.0 -- HVL is not privileged over MVL by default,
    unlike a plain point-density-weighted fit where the many thin-region
    points would naturally dominate).

    Multi-start: for each candidate alpha, beta/gamma optimization is
    attempted from several starting points (the tail-slope-implied guess,
    the ODR result if one is cheaply available via alpha_tail's own
    neighborhood, and a small generic grid) — same rationale as
    fit_archer_odr()'s multi-start grid: this objective surface is not
    convex and a single starting guess can land in a poor local optimum.

    Returns
    -------
    alpha, beta, gamma, sd_alpha, sd_beta, sd_gamma, opt_result
        sd_alpha/sd_beta/sd_gamma are NaN -- this objective is not a
        likelihood, so no standard-error/covariance interpretation applies
        (unlike ODR's or standard's sigma-weighted least-squares, whose
        parameter covariance has a real statistical meaning). Reported as
        NaN rather than a misleading number.
    """
    x = np.asarray(thicknesses, float)
    y = np.asarray(transmissions, float)
    valid = (y > 0) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        raise ValueError(f"Only {len(x)} valid point(s) — need >= 3.")

    if fvl_weights is None:
        fvl_weights = {label: 1.0 for label in FVL_TARGETS}

    # Pre-compute each FVL layer's local-bracketing thickness ONCE (it does
    # not depend on alpha/beta/gamma) -- the optimizer re-evaluates the
    # Archer side only, on every call.
    local_x = {}
    for label, T_target in FVL_TARGETS.items():
        loc = local_fvl(x, y, T_target, n_bracket)
        xl = loc["thickness"]
        if math.isfinite(xl) and xl > 0:
            local_x[label] = xl
    if not local_x:
        raise ValueError(
            "No FVL layer has a usable local-bracketing thickness -- cannot "
            "optimize against FVL agreement. Check data density/spacing.")

    def _fvl_loss(beta, gamma, alpha):
        if not (0 < gamma <= GAMMA_MAX_FIT):
            return np.inf
        total = 0.0
        for label, xl in local_x.items():
            w = fvl_weights.get(label, 1.0)
            if w <= 0:
                continue
            try:
                xa = archer_thickness(FVL_TARGETS[label], alpha, beta, gamma)
            except Exception:
                return np.inf
            if not math.isfinite(xa) or xa <= 0:
                return np.inf
            total += w * ((xa - xl) / xl) ** 2
        return total

    a_lo = alpha_tail * (1.0 - alpha_tol)
    a_hi = alpha_tail * (1.0 + alpha_tol)
    beta_max = 5.0 * alpha_tail
    beta_min = -0.90 * alpha_tail

    beta_starts  = np.array([-2, -0.5, -0.1, 0, 0.1, 0.5, 2]) * alpha_tail
    beta_starts  = beta_starts[(beta_starts >= beta_min) & (beta_starts <= beta_max)]
    gamma_starts = np.array([0.3, 0.8, 1.5, 3.0, 8.0])
    pub = TABLE2.get((nuclide, barrier)) if nuclide else None

    alpha_grid = np.unique(np.concatenate([
        [alpha_tail], np.linspace(a_lo, a_hi, 7)]))

    best = None  # (loss, alpha, beta, gamma, opt_res)
    for alpha_c in alpha_grid:
        starts = [(b0, g0) for b0 in beta_starts for g0 in gamma_starts]
        if pub:
            starts.insert(0, (pub[1], pub[2]))

        for b0, g0 in starts:
            b0c = float(np.clip(b0, beta_min + 1e-9, beta_max - 1e-9))
            g0c = float(np.clip(g0, 1e-3, GAMMA_MAX_FIT - 1e-3))
            try:
                res = minimize(
                    lambda p: _fvl_loss(p[0], p[1], alpha_c),
                    x0=[b0c, g0c], method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
            except Exception:
                continue
            if not res.success and not math.isfinite(res.fun):
                continue
            b_opt, g_opt = res.x
            b_opt = float(np.clip(b_opt, beta_min, beta_max))
            g_opt = float(np.clip(g_opt, 1e-4, GAMMA_MAX_FIT))
            loss  = _fvl_loss(b_opt, g_opt, alpha_c)
            if not math.isfinite(loss):
                continue
            if best is None or loss < best[0]:
                best = (loss, alpha_c, b_opt, g_opt, res)

    if best is None:
        raise ValueError(
            "FVL-optimized fit failed to converge from any starting point -- "
            "try widening --alpha-tol or check for degenerate FVL data.")

    best_loss, a, b, g, opt_res = best

    print(f"\n  FVL-optimized fit: alpha grid {len(alpha_grid)} pts x "
          f"{len(beta_starts)*len(gamma_starts)+(1 if pub else 0)} starts  "
          f"alpha in [{a_lo:.5f}, {a_hi:.5f}]")
    print(f"  best alpha={a:.6f} (alpha/alpha_tail={a/alpha_tail:.4f})  "
          f"beta={b:.6f}  gamma={g:.6f}  FVL-loss={best_loss:.6e}")
    layer_s = ", ".join(f"{lbl}={archer_thickness(FVL_TARGETS[lbl],a,b,g):.2f}"
                         f"/{xl:.2f}mm" for lbl, xl in local_x.items())
    print(f"  archer/local per layer: {layer_s}")

    return a, b, g, float("nan"), float("nan"), float("nan"), opt_res


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATE METHOD — ODR + FVL BLEND (fit_method="odr_fvl")
# ═══════════════════════════════════════════════════════════════════════════════
# fit_archer_odr() selects its multi-start winner purely by point-residual
# log-RMSE; fit_archer_fvl_optimized() ignores point residuals and sigma_T
# weighting entirely, selecting purely by archer-vs-local FVL agreement.
# This method is a genuine blend of the two, not a rename of either:
#   1. Run the SAME uncertainty-weighted ODR (scipy.odr.ODR, sx=thickness_unc,
#      sy=sigma_T, curve_fit-with-equivalent-weights fallback) multi-start
#      grid as fit_archer_odr(), so every candidate is a real ODR solution
#      that respects the measured uncertainties in both x and y.
#   2. Instead of keeping whichever candidate has the lowest point-residual
#      RMSE, score every converged candidate with a BLENDED objective:
#          score = rmse_log10 / rmse_ref  +  fvl_weight * (fvl_loss / fvl_ref)
#      where rmse_ref/fvl_ref are the best single-objective values seen
#      across the candidate grid (normalizes the two very differently-scaled
#      terms onto a comparable ~[0,~few] range instead of one silently
#      dominating because of raw magnitude).
#   3. Locally polish the blended winner: beta/gamma only (alpha stays
#      pinned to its ODR-fitted value from step 2, keeping the "alpha
#      anchored near the physical tail slope" property that both odr and
#      fvl_optimized already guarantee) via a bounded Nelder-Mead directly
#      against the same blended score, seeded from the winning candidate.
# Net effect: a fit that is still a real ODR solution (not just point-fit
# dressed up) but is explicitly pulled toward lower archer-vs-local FVL
# disagreement, with a tunable knob (fvl_weight) for how hard to pull.
# ═══════════════════════════════════════════════════════════════════════════════

def fit_archer_odr_fvl_blend(thicknesses, transmissions, sigma_T,
                              alpha_tail, alpha_tol=DEFAULT_ALPHA_TOL,
                              thickness_unc_mm=0.5,
                              nuclide=None, barrier=None,
                              fvl_weight=1.0, fvl_weights=None, n_bracket=3):
    """
    ODR fit whose candidate selection and final local polish are both driven
    by a blend of point-residual log-RMSE (the same quantity fit_archer_odr()
    uses) and archer-vs-local FVL disagreement (the same quantity
    fit_archer_fvl_optimized() uses) -- see module comment above for the
    full rationale. alpha is pinned to the ODR-fitted value from the winning
    candidate (itself constrained to [alpha_tail*(1-tol), alpha_tail*(1+tol)]
    exactly like fit_archer_odr()); only beta/gamma move during the local
    polish step.

    Parameters
    ----------
    fvl_weight : float
        Relative weight of the (normalized) FVL-agreement term vs. the
        (normalized) point-residual RMSE term in the blended score.
        1.0 (default) weights them equally after normalization. Larger
        values pull the fit harder toward FVL agreement at the possible
        cost of point-residual fit quality; 0.0 reduces to plain ODU
        candidate selection (equivalent to fit_archer_odr() plus a no-op
        polish step).
    fvl_weights : dict[str, float], optional
        Per-layer (HVL/QVL/TVL/CVL/MVL) weight within the FVL term, same
        convention as fit_archer_fvl_optimized(). Default: every resolvable
        layer weighted equally.
    n_bracket : int
        Local-bracketing window size for local_fvl(), same as
        fit_archer_fvl_optimized().

    Returns
    -------
    alpha, beta, gamma, sd_alpha, sd_beta, sd_gamma, odr_result
        sd_alpha/sd_beta/sd_gamma come from the winning ODR candidate's
        covariance when available (same as fit_archer_odr()); the
        subsequent beta/gamma polish step does not have a likelihood
        interpretation, so if the polish moves the parameters at all the
        standard errors are reported as NaN (misleading otherwise).
    """
    x   = np.asarray(thicknesses, float)
    y   = np.asarray(transmissions, float)
    sig = np.asarray(sigma_T, float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.01 * y)
    sig = np.where(sig > 0, sig, 1e-6)
    sigma_x = np.full_like(x, thickness_unc_mm)

    if fvl_weights is None:
        fvl_weights = {label: 1.0 for label in FVL_TARGETS}

    # Local-bracketing FVL thicknesses, precomputed once (data-only, does
    # not depend on alpha/beta/gamma) -- same approach as
    # fit_archer_fvl_optimized().
    local_x = {}
    for label, T_target in FVL_TARGETS.items():
        loc = local_fvl(x, y, T_target, n_bracket)
        xl = loc["thickness"]
        if math.isfinite(xl) and xl > 0:
            local_x[label] = xl

    def _fvl_loss(alpha, beta, gamma):
        if not local_x or not (0 < gamma <= GAMMA_MAX_FIT):
            return np.inf
        total = 0.0
        for label, xl in local_x.items():
            w = fvl_weights.get(label, 1.0)
            if w <= 0:
                continue
            try:
                xa = archer_thickness(FVL_TARGETS[label], alpha, beta, gamma)
            except Exception:
                return np.inf
            if not math.isfinite(xa) or xa <= 0:
                return np.inf
            total += w * ((xa - xl) / xl) ** 2
        return total

    a_lo = alpha_tail * (1.0 - alpha_tol)
    a_hi = alpha_tail * (1.0 + alpha_tol)
    beta_max = 5.0 * alpha_tail
    beta_min = -0.90 * alpha_tail

    beta_vals  = np.array([-5,-3,-1.5,-0.5,-0.1,0,0.1,0.5,1.5,3,5]) * alpha_tail
    beta_vals  = beta_vals[beta_vals >= beta_min]
    gamma_vals = np.unique(np.round(np.concatenate([
        np.linspace(0.05, 0.5,  5),
        np.linspace(0.5,  5.0,  8),
        np.linspace(5.0, 30.0,  5)]), 4))
    candidates = [(alpha_tail, b, g)
                  for b in beta_vals for g in gamma_vals]
    pub = TABLE2.get((nuclide, barrier)) if nuclide else None
    if pub:
        candidates.insert(0, pub)

    print(f"\n  ODR+FVL blend: {len(candidates)} candidates  "
          f"alpha in [{a_lo:.5f}, {a_hi:.5f}]  fvl_weight={fvl_weight}")

    def _clip(p0):
        a = float(np.clip(p0[0], a_lo + 1e-10, a_hi - 1e-10))
        b = float(np.clip(p0[1], beta_min + 1e-10, beta_max - 1e-10))
        g = float(np.clip(p0[2], 1e-4, GAMMA_MAX_FIT - 1e-4))
        return [a, b, g]

    # Pass 1: collect every converged ODR candidate's (params, rmse, fvl_loss,
    # odr_result) -- same solve as fit_archer_odr(), but nothing is discarded
    # yet so the blended score can be computed after seeing the full range.
    conv = []  # list of dict(popt, rmse, fvl, odr_res)
    for p0 in candidates:
        p0c = _clip(p0)
        popt = None; odr_res = None
        try:
            model = Model(archer_odr_func)
            data  = RealData(x, y, sx=sigma_x, sy=sig)
            odr   = ODR(data, model, beta0=p0c, ifixb=[0, 0, 0], maxit=1000)
            res   = odr.run()
            popt  = res.beta
            if not (a_lo <= popt[0] <= a_hi and
                    beta_min <= popt[1] <= beta_max and
                    0 < popt[2] <= GAMMA_MAX_FIT):
                raise ValueError("ODR solution outside bounds")
            odr_res = res
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        archer_transmission, x, y, p0=p0c,
                        bounds=([a_lo, beta_min, 1e-4],
                                [a_hi, +beta_max, GAMMA_MAX_FIT]),
                        sigma=sig, absolute_sigma=True, maxfev=200_000)
            except Exception:
                continue
        y_fit = archer_transmission(x, *popt)
        ok = (y > 0) & (y_fit > 0) & np.isfinite(y_fit)
        if not ok.any():
            continue
        rmse = float(np.sqrt(np.mean(
            (np.log10(y_fit[ok]) - np.log10(y[ok]))**2)))
        fvl = _fvl_loss(*popt)
        conv.append({"popt": popt, "rmse": rmse, "fvl": fvl, "odr_res": odr_res})

    if not conv:
        raise ValueError(
            f"All {len(candidates)} ODR+FVL candidates failed to converge. "
            "Try widening --alpha-tol or check for degenerate data.")

    # Normalize each term by the best single-objective value actually seen,
    # so rmse (typically ~0.01-0.1) and fvl_loss (can be anywhere from ~1e-4
    # to 10+, being a sum over up to 5 squared relative errors) sit on a
    # comparable scale before blending -- otherwise fvl_weight=1.0 would not
    # mean "equal weight" in practice.
    finite_fvl = [c["fvl"] for c in conv if math.isfinite(c["fvl"])]
    rmse_ref = min(c["rmse"] for c in conv) or 1e-12
    fvl_ref  = (min(finite_fvl) or 1e-12) if finite_fvl else 1.0

    for c in conv:
        fvl_term = (c["fvl"] / fvl_ref) if math.isfinite(c["fvl"]) else np.inf
        c["score"] = (c["rmse"] / rmse_ref) + fvl_weight * fvl_term

    conv.sort(key=lambda c: c["score"])
    winner = conv[0]
    a, b, g = winner["popt"]
    odr_res = winner["odr_res"]

    print(f"  ODR+FVL blend: {len(conv)}/{len(candidates)} candidates converged  "
          f"winner score={winner['score']:.5f}  "
          f"(rmse={winner['rmse']:.5f}, fvl_loss={winner['fvl']:.6e})")

    # Local polish: beta/gamma only, alpha pinned at the winner's ODR value,
    # directly minimizing the SAME blended score (rmse term recomputed at
    # each trial point, fvl term via _fvl_loss) -- lets the fit nudge closer
    # to FVL agreement beyond what the discrete candidate grid landed on,
    # without drifting alpha off its physically-anchored value.
    def _blended_score(params):
        beta_c, gamma_c = params
        if not (0 < gamma_c <= GAMMA_MAX_FIT):
            return np.inf
        y_fit = archer_transmission(x, a, beta_c, gamma_c)
        ok = (y > 0) & (y_fit > 0) & np.isfinite(y_fit)
        if not ok.any():
            return np.inf
        rmse_c = float(np.sqrt(np.mean(
            (np.log10(y_fit[ok]) - np.log10(y[ok]))**2)))
        fvl_c = _fvl_loss(a, beta_c, gamma_c)
        fvl_term = (fvl_c / fvl_ref) if math.isfinite(fvl_c) else np.inf
        return (rmse_c / rmse_ref) + fvl_weight * fvl_term

    # Skip the local polish entirely when fvl_weight<=0 -- with no FVL term,
    # _blended_score reduces to plain point-residual log-RMSE evaluated via
    # a straight forward archer_transmission() call, which is NOT the same
    # objective the winning candidate's own ODR solve minimized (ODR
    # minimizes weighted ORTHOGONAL distance in (x,y), accounting for
    # thickness uncertainty too -- not vertical-only log-residual RMSE). A
    # Nelder-Mead polish against that different objective could legitimately
    # nudge beta/gamma away from the true ODR answer even at fvl_weight=0,
    # which would silently break the documented guarantee that fvl_weight=0
    # reduces this method to plain fit_archer_odr(). Skipping the polish
    # step entirely (not just the FVL term inside it) is what makes that
    # guarantee actually hold.
    polished = False
    if fvl_weight > 0:
        try:
            b_lo, b_hi = beta_min, beta_max
            g_lo, g_hi = 1e-4, GAMMA_MAX_FIT
            b0 = float(np.clip(b, b_lo + 1e-9, b_hi - 1e-9))
            g0 = float(np.clip(g, g_lo + 1e-9, g_hi - 1e-9))
            # Defensive finite-check BEFORE handing control to SciPy's
            # compiled Nelder-Mead code. An aggressive/unusual fvl_weights
            # combination (e.g. a layer weighted well above 1.0) can in
            # principle drive the winning candidate's (a,b,g) into a region
            # where _blended_score()'s intermediate archer_transmission()
            # values are NaN/inf despite every individual guard inside it
            # looking satisfied on paper -- a Python-level try/except around
            # minimize() does NOT reliably catch a crash originating inside
            # native optimizer code fed non-finite objective values (this
            # can present as the whole process dying/freezing with no
            # Python traceback at all, rather than a normal exception).
            # Refuse to even start the polish if the seed point itself
            # isn't finite and if _blended_score doesn't return a finite
            # value at that seed -- fall back to the un-polished ODR winner
            # instead, which is always safe.
            seed_score = _blended_score([b0, g0])
            if not (math.isfinite(b0) and math.isfinite(g0)
                    and math.isfinite(seed_score)):
                print(f"  ODR+FVL blend: skipping local polish -- seed point "
                      f"(beta={b0:.6g}, gamma={g0:.6g}) is non-finite or "
                      f"gives a non-finite blended score "
                      f"(score={seed_score!r}). Keeping the un-polished ODR "
                      f"winner instead.")
            else:
                res_polish = minimize(
                    _blended_score, x0=[b0, g0], method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
                if res_polish.success or math.isfinite(res_polish.fun):
                    b_new = float(np.clip(res_polish.x[0], b_lo, b_hi))
                    g_new = float(np.clip(res_polish.x[1], g_lo, g_hi))
                    if (math.isfinite(res_polish.fun)
                            and res_polish.fun < winner["score"]):
                        b, g = b_new, g_new
                        polished = True
        except Exception as e:
            msg = str(e) or repr(e) or "(no exception message)"
            print(f"  ODR+FVL blend: local polish failed "
                  f"({type(e).__name__}: {msg}) -- keeping the un-polished "
                  f"ODR winner instead.")

    if polished:
        final_fvl = _fvl_loss(a, b, g)
        print(f"  ODR+FVL blend: local polish improved score "
              f"(beta={b:.6f}, gamma={g:.6f}, fvl_loss={final_fvl:.6e})")
        sd_a = sd_b = sd_g = float("nan")  # polish step has no likelihood
    elif odr_res is not None and hasattr(odr_res, "sd_beta"):
        sd_a, sd_b, sd_g = odr_res.sd_beta
    else:
        sd_a = sd_b = sd_g = float("nan")

    if local_x:
        # archer_thickness() can hit a math domain error (log of a
        # non-positive argument) for a pathological (a,b,g) triple -- e.g.
        # a winner/polish result driven toward an unusual fvl_weights
        # combination can land beta/gamma somewhere that's valid for the
        # layers _fvl_loss() actually optimized against but degenerate for
        # a different layer being reported here. This is purely a cosmetic
        # summary line -- the fit itself (a,b,g) is already finalized above
        # -- so a bad layer must not be allowed to raise all the way out of
        # fit_archer_odr_fvl_blend() and abort the whole (nuclide,barrier)
        # fit. Report "n/a" for any layer that fails instead.
        parts = []
        for lbl, xl in local_x.items():
            try:
                xa = archer_thickness(FVL_TARGETS[lbl], a, b, g)
                parts.append(f"{lbl}={xa:.2f}/{xl:.2f}mm")
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                parts.append(f"{lbl}=n/a({type(e).__name__})/{xl:.2f}mm")
        print(f"  archer/local per layer: {', '.join(parts)}")

    print(f"  alpha={a:.6f} (alpha/alpha_tail={a/alpha_tail:.4f})  "
          f"beta={b:.6f}  gamma={g:.6f}")

    return a, b, g, sd_a, sd_b, sd_g, odr_res


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
    unc_results=[]; n_estimates=[]; sigma_T=[]; nonzero_pct=[]

    print(f"\n  {'t(mm)':>8}  {'T':>10}  {'sbar%':>6}  {'sair%':>6}  "
          f"{'sT%':>6}  {'nz%':>6}  {'T+-s':>14}")
    print(f"  {'--':>8}  {'--':>10}  {'--':>6}  {'--':>6}  {'--':>6}  "
          f"{'--':>6}  {'--':>14}")

    for f in files:
        t       = parse_thickness(f)
        arr_bar = load_dose(f)
        unc_bar = load_uncertainty(f)
        n_bar   = read_n_primaries(f)
        d_bar   = roi_mean_dose(arr_bar,z_flipped)/n_bar
        T       = d_bar/d_air

        # Same definition the MHD-viewer tab in shieldLabGUI.py already shows
        # per-file (np.count_nonzero(flat)/flat.size*100) -- whole array, not
        # ROI-masked, so this number matches what a user sees there for the
        # exact same file and can be used as a familiar, already-trusted
        # sparsity/statistics sanity check on top of sigma_rel_T.
        nz_pct = float(np.count_nonzero(arr_bar)) / arr_bar.size * 100.0

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
              f"{_f(unc['sigma_rel_air'])}  {_f(s_T_val)}  {nz_pct:6.1f}  "
              f"{T_pm:>14}")

        thicknesses.append(t); transmissions.append(T)
        doses_barrier.append(d_bar); unc_results.append(unc)
        n_estimates.append(n_est)
        sigma_T.append(a_T_val if math.isfinite(a_T_val) else float("nan"))
        nonzero_pct.append(nz_pct)

    return (np.array(thicknesses), np.array(transmissions),
            np.array(doses_barrier), float(d_air),
            unc_results, n_estimates, np.array(sigma_T),
            np.array(nonzero_pct))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def select_fit_points(thicknesses, transmissions, n_points=None, min_T=None,
                       start_idx=None, end_idx=None,
                       nonzero_pct=None, min_nonzero_pct=None):
    """
    Select points for Archer fitting.

    Parameters
    ----------
    thicknesses : array
        Barrier thickness values
    transmissions : array
        Transmission factor values
    n_points : int, optional
        Maximum number of points from the front (legacy)
    min_T : float, optional
        Minimum transmission value threshold
    start_idx : int, optional
        Starting index for range selection (0-based, inclusive)
    end_idx : int, optional
        Ending index for range selection (0-based, inclusive)
        Use -1 or None for last point
    nonzero_pct : array, optional
        Per-point % of non-zero voxels in that thickness's dose .mhd array
        (same definition as the MHD-viewer tab in shieldLabGUI.py:
        np.count_nonzero(arr)/arr.size*100, whole array). Required if
        min_nonzero_pct is given.
    min_nonzero_pct : float, optional
        Drop any point whose nonzero_pct falls below this threshold, e.g.
        90 to require at least 90% of voxels to have registered a nonzero
        score. This is a sparsity/statistics floor independent of min_T --
        a point can have T above min_T yet still be too sparse at the voxel
        level to trust (see the In111/Lu177 Gypsum sparse-buildup-region
        cases from the material-attenuation investigation, where a badly
        undersampled thin/buildup point skewed the whole Archer fit even
        though its T value itself looked fine).

    Returns
    -------
    mask : bool array
        Boolean mask of selected points
    """
    mask = transmissions > 0
    if min_T is not None:
        mask &= transmissions >= min_T
    if min_nonzero_pct is not None:
        if nonzero_pct is None:
            raise ValueError("min_nonzero_pct given but nonzero_pct array "
                              "not provided to select_fit_points().")
        mask &= nonzero_pct >= min_nonzero_pct

    # Range-based selection (new method)
    if start_idx is not None or end_idx is not None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return mask
        
        # Handle start index
        if start_idx is None:
            start_idx = 0
        elif start_idx < 0:
            start_idx = len(idx) + start_idx
        start_idx = max(0, min(start_idx, len(idx) - 1))
        
        # Handle end index
        if end_idx is None or end_idx == -1:
            end_idx = len(idx) - 1
        elif end_idx < 0:
            end_idx = len(idx) + end_idx
        end_idx = max(0, min(end_idx, len(idx) - 1))
        
        # Ensure start <= end
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        # Create new mask with only the range
        range_mask = np.zeros_like(mask)
        range_mask[idx[start_idx:end_idx+1]] = True
        mask = mask & range_mask
    
    # Legacy n_points selection (from front)
    elif n_points is not None and n_points > 0:
        idx = np.where(mask)[0]
        if len(idx) > n_points: 
            mask[idx[n_points:]] = False
    
    return mask

def fit_archer_full(thicknesses, transmissions, sigma_T,
                    n_points=None, min_T=None,
                    start_idx=None, end_idx=None,
                    alpha_tail_n=DEFAULT_ALPHA_TAIL_N,
                    alpha_tol=DEFAULT_ALPHA_TOL,
                    thickness_unc_mm=0.5,
                    nuclide=None, barrier=None,
                    nonzero_pct=None, min_nonzero_pct=None,
                    fit_method="odr",
                    anchor_alpha_global=False,
                    fvl_weight=1.0,
                    fvl_layer_weights=None):
    """
    Full Archer fit per Section 2.4, with a choice of fitting method:

      fit_method="odr" (default, paper-matched):
        Step 1: alpha from OLS on ln T vs x (last alpha_tail_n valid points)
        Step 2: ODR with alpha pinned near alpha_tail, beta+gamma free
        (fit_archer_odr() — matches the OriginPro ODR methodology described
        in the paper, including the multi-start grid over beta/gamma.)

      anchor_alpha_global (bool, default False):
        By default, Step 1's "last alpha_tail_n valid points" are the last
        points of whatever subset survives point selection (n_points/min_T/
        start_idx/end_idx/min_nonzero_pct) — so a shallow selection (e.g.
        min_T=0.01 or min_nonzero_pct=90 that excludes the deepest measured
        points) silently anchors alpha to whatever happens to be deepest
        WITHIN that selection, not the deepest points actually measured for
        this (nuclide, barrier) pair. Set anchor_alpha_global=True to instead
        always compute alpha_tail from the last alpha_tail_n valid points of
        the FULL thicknesses/transmissions arrays passed to this function
        (every simulated point for this pair, before n_points/min_T/
        start_idx/end_idx/min_nonzero_pct are applied), regardless of which
        subset is selected for the actual curve fit. The fit itself (Step 2)
        still only sees the selected subset — only alpha's anchor point
        changes. Only affects fit_method in {"odr","fvl_optimized","gsa"}
        (all three pin alpha to alpha_tail); "standard" ignores alpha_tail
        entirely and is unaffected either way.

      fit_method="standard" (alternative, for cross-checking):
        A single plain weighted nonlinear least-squares fit
        (fit_archer_standard()) with all 3 parameters free simultaneously
        from one generic starting guess — no alpha pinning to the data
        tail, no ODR treatment of thickness uncertainty, no multi-start
        grid. The tail alpha (Step 1 above) is still computed either way
        purely for reporting/comparison (alpha_tail, r2_tail are always
        returned) — it just isn't used to constrain the standard fit.

      fit_method="fvl_optimized" (alternative, FVL-agreement-driven):
        Alpha stays pinned to the same [alpha_tail*(1-alpha_tol),
        alpha_tail*(1+alpha_tol)] physical band as the ODR fit (evaluated
        over a small grid within that band, not floated freely) — beta and
        gamma are chosen to directly minimize the disagreement between the
        Archer-equation-derived FVL thickness (HVL/QVL/TVL/CVL/MVL) and the
        model-free local-bracketing-interpolated FVL thickness, summed in
        relative-squared-error over every FVL layer with usable data
        (fit_archer_fvl_optimized()). Unlike ODR/standard, whose objective
        is point-by-point transmission residuals, this method's objective
        is exactly the paper's own Section 2.4 acceptance criterion (<=10%
        Archer-vs-local-bracket FVL disagreement) — so it directly targets
        the metric the fit will ultimately be judged on, at the cost of
        being a non-likelihood-based optimization (no meaningful parameter
        standard errors; sd_a/sd_b/sd_g are returned as NaN).

      fit_method="gsa" (diagnostic overlay, Grouped Spectral Archer):
        Runs gsa_fit.fit_transmission() — a physics-anchored fit of a
        WEIGHTED SUM of K single-term Archer curves (one per photon-energy
        group in the nuclide's real emission spectrum, grouped by linear
        attenuation coefficient in THIS barrier material via NIST XCOM
        data), selected against a plain single-term Archer fit using both a
        physical crossover criterion and AICc. This exists because a single
        Archer term can only approximate a multi-energy-group spectrum's
        true transmission curve, and does worst exactly where it matters —
        deep in the barrier, once the harder-attenuating group has died off
        and the curve is really governed by the softer group's own
        asymptotic slope.

        IMPORTANT — this method does NOT change what gets returned/plotted/
        used for FVL: the (alpha,beta,gamma) returned by this branch is
        STILL the plain single-term Archer fit (identical to fit_method=
        "odr"'s alpha-pinned result), so plot_static(), compute_fvl_with_
        local(), write_transmission_csv(), and the interactive tuner all
        keep working completely unchanged. GSA's own result (verdict, K,
        per-group weights/alpha/beta/gamma, dAICc, crossover x_c) is
        computed alongside as a DIAGNOSTIC — call fit_gsa_diagnostic()
        directly (or use analyze_one(..., fit_method="gsa")) to also
        persist/print it. This design choice (report standard triple, GSA
        data on the side, rather than trying to force a multi-term GSA
        result through single-triple-shaped fields it can't accurately
        represent) was confirmed with the user rather than assumed.

        Requires gsa_fit.py + nist_xcom_data.py to be importable (see
        GSA_OK at module level) — raises RuntimeError with a clear message
        if they aren't, rather than silently falling back to a different
        method.

      fit_method="piecewise" (diagnostic overlay, thin/thick Archer):
        Runs fit_archer_piecewise() — grid-searches a cutoff thickness x*
        and fits TWO independently-invertible 3-parameter Archer triples,
        one for x<=x* ("thin") and one for x>x* ("thick"), continuity-
        constrained so the two curves meet exactly at x* (no visible jump
        at the seam). This is a simpler alternative to GSA for the same
        two-regime-transmission-curve problem: no nuclear-physics anchoring
        needed (no PHOTON_SPECTRA/NIST XCOM lookup), and inversion is a
        plain archer_thickness() call on whichever side's triple governs a
        given target transmission (see archer_thickness_piecewise()) — no
        Newton iteration required, unlike gsa_thickness(). The trade-off:
        the two triples carry no physical meaning of their own (nothing
        ties "thin alpha" to an actual emission line the way GSA's
        per-group alpha does) — purely a numerical curve-fit split, and in
        testing against Cu64/Lead it beat GSA on raw point-fit RMSE but had
        larger FVL disagreement at some layers than GSA's physics-anchored
        fit.

        IMPORTANT — same "diagnostic overlay" contract as fit_method="gsa"
        above: the (alpha,beta,gamma) returned by this branch is STILL the
        plain single-term Archer fit, so every existing downstream consumer
        of this function's return value keeps working unchanged. The
        piecewise result (x_star, thin/thick triples, RMSE comparison,
        verdict) is computed alongside as a DIAGNOSTIC — call
        fit_piecewise_diagnostic() directly (or use analyze_one(...,
        fit_method="piecewise")) to also persist/print it.

        No availability gate (unlike GSA's GSA_OK) since this needs no
        external physics data -- fit_piecewise_diagnostic() only returns
        None on an actual fit failure (e.g. too few points on one side of
        every candidate cutoff), logged via warnings.warn.

      fit_method="odr_fvl" (blend, fit_archer_odr_fvl_blend()):
        A genuine ODR fit (same uncertainty-weighted scipy.odr.ODR
        multi-start grid as fit_method="odr", sx=thickness_unc_mm,
        sy=sigma_T) whose candidate SELECTION and a subsequent beta/gamma-
        only local polish are both driven by a blended objective combining
        point-residual log-RMSE (what "odr" alone optimizes) with
        archer-vs-local FVL disagreement (what "fvl_optimized" alone
        optimizes), normalized onto a comparable scale and combined via
        fvl_weight (see below). Unlike "fvl_optimized", every candidate
        this method considers is a real ODR solution respecting the
        measured x/y uncertainties -- the FVL term only influences WHICH
        converged ODR solution is kept and how far beta/gamma are nudged
        from it afterward, alpha stays pinned to the winning candidate's
        ODR-fitted value throughout.

        fvl_weight (float, default 1.0): relative weight of the normalized
        FVL-agreement term vs. the normalized RMSE term. 0.0 reduces this
        to plain ODR candidate selection (equivalent to fit_method="odr"
        plus a no-op polish); larger values pull harder toward FVL
        agreement, potentially at some cost to point-residual fit quality.
        Use this to reconcile an "odr" fit that already matches a published
        reference well against an "fvl_optimized" fit that disagrees with
        it by a few percent -- start near 0 and increase gradually to see
        how much FVL-agreement improvement costs in RMSE, rather than
        jumping straight to fvl_optimized's fully FVL-driven answer.

    fvl_layer_weights (dict[str, float], optional; only affects
    "fvl_optimized" and "odr_fvl"):
        Per-layer weight within the FVL-agreement term, keyed by FVL_TARGETS
        label ("HVL","QVL","TVL","CVL","MVL"). Default (None) weights every
        resolvable layer equally at 1.0 -- e.g. HVL is not privileged over
        MVL. Set this to anchor the fit to whichever layer(s) you actually
        care about matching -- e.g. {"HVL":1.0} (everything else implicitly
        0) to optimize purely for HVL agreement, or {"HVL":2.0,"CVL":1.0}
        to weight HVL twice as heavily as CVL while ignoring QVL/TVL/MVL
        entirely. A layer set to 0 (or simply omitted) drops out of the
        objective the same way an unresolvable layer (too few points
        straddling its T_target) already does. See parse_fvl_layer_weights()
        for the CLI/GUI string format ("HVL=1,CVL=0.5").
    """
    fit_mask = select_fit_points(thicknesses, transmissions, n_points, min_T,
                                   start_idx, end_idx,
                                   nonzero_pct, min_nonzero_pct)
    x  = thicknesses[fit_mask]
    y  = transmissions[fit_mask]
    sy = sigma_T[fit_mask]

    if len(x) < 3:
        raise ValueError(f"Only {len(x)} point(s) selected — need >= 3.")

    # Step 1 — tail alpha, computed regardless of fit_method (used to
    # constrain the ODR fit; kept purely informational for the standard fit
    # so the two methods' printed/saved reports stay directly comparable).
    #
    # anchor_alpha_global=True: anchor to the deepest alpha_tail_n points of
    # the FULL (pre-selection) dataset instead of the selected subset (x,y),
    # so alpha stays pinned to the true hardened-beam tail regardless of
    # what fit range/threshold is chosen (e.g. min_T=0.01 or
    # min_nonzero_pct=90 that would otherwise exclude the deepest measured
    # points from Step 1's own tail-slope calculation).
    if anchor_alpha_global:
        all_valid = (np.asarray(transmissions) > 0) & np.isfinite(transmissions)
        n_all_valid = int(all_valid.sum())
        if n_all_valid >= 2:
            alpha_tail, r2_tail, tail_x, tail_lnT, slope_se = \
                fit_alpha_from_tail(thicknesses, transmissions, n_tail=alpha_tail_n)
            anchor_note = "GLOBAL (full dataset, ignores fit-range selection)"
        else:
            # Not enough valid points in the full dataset (shouldn't happen
            # if the selected subset itself had >=3, but guard anyway) --
            # fall back to the selected-subset tail rather than crash.
            alpha_tail, r2_tail, tail_x, tail_lnT, slope_se = \
                fit_alpha_from_tail(x, y, n_tail=alpha_tail_n)
            anchor_note = "selected subset (global anchor requested but full dataset had <2 valid points)"
    else:
        alpha_tail, r2_tail, tail_x, tail_lnT, slope_se = \
            fit_alpha_from_tail(x, y, n_tail=alpha_tail_n)
        anchor_note = "selected subset"

    se_s = f"  SE={slope_se:.5f}" if math.isfinite(slope_se) else ""
    print(f"\n  Step 1 — alpha from tail ({alpha_tail_n} pts, anchor={anchor_note}: "
          f"x={','.join(f'{v:.1f}' for v in tail_x)} mm)")
    print(f"    alpha_tail = {alpha_tail:.6f} mm^-1   "
          f"R^2 = {r2_tail:.4f}{se_s}")
    if r2_tail < 0.98:
        print("    !! R^2 < 0.98 — tail may not be fully straight yet.")

    mu_nist = MU_NARROW_NIST.get((nuclide,barrier)) if nuclide else None
    if mu_nist:
        print(f"    alpha_tail/mu_NIST = {alpha_tail/mu_nist:.3f}  "
              f"(mu_NIST={mu_nist:.5f} mm^-1 — info only)")

    # Step 2 — fit method
    if fit_method == "standard":
        print(f"\n  Step 2 — STANDARD fit (plain weighted NLS, alpha NOT "
              f"pinned to tail)")
        a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_standard(x, y, sy)
    elif fit_method == "odr":
        a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_odr(
            x, y, sy, alpha_tail, alpha_tol, thickness_unc_mm, nuclide, barrier)
    elif fit_method == "fvl_optimized":
        layer_s = (f"  layer weights={fvl_layer_weights}"
                   if fvl_layer_weights else "  (all layers weighted equally)")
        print(f"\n  Step 2 — FVL-OPTIMIZED fit (alpha pinned to tail, "
              f"beta/gamma chosen to minimize Archer-vs-local FVL "
              f"disagreement){layer_s}")
        a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_fvl_optimized(
            x, y, sy, alpha_tail, alpha_tol, nuclide, barrier,
            fvl_weights=fvl_layer_weights)
    elif fit_method == "odr_fvl":
        layer_s = (f"  layer weights={fvl_layer_weights}"
                   if fvl_layer_weights else "  (all layers weighted equally)")
        print(f"\n  Step 2 — ODR+FVL BLEND fit (real ODR candidates, "
              f"selected/polished by blended RMSE+FVL score, fvl_weight="
              f"{fvl_weight}){layer_s}")
        a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_odr_fvl_blend(
            x, y, sy, alpha_tail, alpha_tol, thickness_unc_mm, nuclide, barrier,
            fvl_weight=fvl_weight, fvl_weights=fvl_layer_weights)
    elif fit_method == "gsa":
        # GSA is a DIAGNOSTIC OVERLAY, not a replacement for the returned
        # triple (see docstring above) -- the plotted/reported/FVL-table
        # fit stays the plain alpha-pinned single-term Archer result
        # (identical to fit_method="odr"), so every downstream consumer of
        # this function's return value keeps working unchanged. Run the
        # actual GSA comparison via fit_gsa_diagnostic() separately (already
        # done automatically by analyze_one() when fit_method="gsa").
        print(f"\n  Step 2 — GSA fit_method requested: reporting the plain "
              f"alpha-pinned single-term Archer fit here (same as 'odr') "
              f"-- the actual GSA (multi-group) comparison is computed "
              f"separately as a diagnostic; see fit_gsa_diagnostic() / the "
              f"printed/saved GSA DIAGNOSTIC block.")
        a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_odr(
            x, y, sy, alpha_tail, alpha_tol, thickness_unc_mm, nuclide, barrier)
    elif fit_method == "piecewise":
        # Same "diagnostic overlay" pattern as fit_method="gsa" (see above):
        # the returned triple STAYS the plain alpha-pinned single-term
        # Archer fit, so plot_static(), compute_fvl_with_local(),
        # write_transmission_csv(), and the interactive tuner all keep
        # working unchanged. The actual piecewise (thin/thick) result is
        # computed separately as a diagnostic by analyze_one() when
        # fit_method="piecewise" -- see fit_piecewise_diagnostic().
        print(f"\n  Step 2 — PIECEWISE fit_method requested: reporting the "
              f"plain alpha-pinned single-term Archer fit here (same as "
              f"'odr') -- the actual piecewise (thin/thick) comparison is "
              f"computed separately as a diagnostic; see "
              f"fit_piecewise_diagnostic() / the printed/saved PIECEWISE "
              f"DIAGNOSTIC block.")
        a, b, g, sd_a, sd_b, sd_g, odr_res = fit_archer_odr(
            x, y, sy, alpha_tail, alpha_tol, thickness_unc_mm, nuclide, barrier)
    else:
        raise ValueError(
            f"Unknown fit_method={fit_method!r} — expected 'odr', "
            f"'standard', 'fvl_optimized', 'odr_fvl', 'gsa', or 'piecewise'.")

    return (a, b, g, sd_a, sd_b, sd_g, fit_mask, alpha_tail, r2_tail, odr_res)


def fit_gsa_diagnostic(thicknesses, transmissions, nuclide, barrier, **kwargs):
    """
    Run the actual GSA (Grouped Spectral Archer) comparison via
    gsa_fit.fit_transmission() and return its GSAFitResult, or None if
    gsa_fit.py isn't available (GSA_OK False) or the fit itself fails --
    logs a clear warning either way rather than raising, since this is a
    diagnostic add-on and should never be the reason a normal analyze_one()
    run aborts. thicknesses/transmissions should be the FULL (unmasked)
    arrays -- gsa_fit.fit_transmission() does its own point-selection
    (dropping T<=0) and reports n/n_dropped based on the full curve, not
    whatever subset fit_archer_full()'s point-selection UI settings chose.
    """
    if not GSA_OK:
        print("  !! GSA diagnostic skipped -- gsa_fit.py not importable "
              "(see GSA_OK at module level). Fit method 'gsa' still ran "
              "the plain alpha-pinned Archer fit above; only the GSA "
              "comparison itself is unavailable.")
        return None
    try:
        return _gsa.fit_transmission(thicknesses, transmissions, nuclide,
                                      barrier, verbose=True, **kwargs)
    except Exception as exc:
        warnings.warn(f"fit_gsa_diagnostic: GSA comparison failed for "
                       f"{nuclide}/{barrier}: {exc}", RuntimeWarning)
        return None


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
        # Use Archer equation as fallback when local bracketing fails
        x_display = r['x_local'] if math.isfinite(r['x_local']) else r['x_archer']
        method_display = r['method'] if math.isfinite(r['x_local']) else "arch"
        row = (f"  {r['label']:<5}  {r['T_target']:>5.3f}  "
               f"{r['x_archer']:>8.2f}  {x_display:>8.2f}  "
               f"{method_display:>4}  {r2v:>6.4f}  "
               f"{r['delta_pct']:>5.1f}%  "
               f"{'v' if r['accepted'] else 'X':>2}")
        if pub:
            try: row += f"  {archer_thickness(r['T_target'],*pub):>8.2f}"
            except: row += f"  {'err':>8}"
        print(row)
    print(f"{'='*72}")


def write_transmission_csv(nuclide, barrier, thicknesses, transmissions,
                            doses_barrier, dose_air, alpha, beta, gamma,
                            unc_results, n_estimates, output_dir, fit_mask,
                            nonzero_pct=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir/f"{nuclide}_{barrier}_transmission_data.csv"
    if nonzero_pct is None:
        nonzero_pct = [float("nan")]*len(thicknesses)
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["thickness_mm","dose_transmitted","dose_reference",
                    "T_simulated","T_fitted","pct_residual",
                    "sigma_rel_bar%","sigma_rel_air%",
                    "sigma_rel_T%","sigma_abs_T","nonzero_voxel_pct",
                    "used_in_fit"])
        for t,d_bar,T_sim,unc,n_est,in_fit,nz in zip(
                thicknesses,doses_barrier,transmissions,
                unc_results,n_estimates,fit_mask,nonzero_pct):
            T_fit = archer_transmission(t,alpha,beta,gamma)
            pct   = 100*(T_sim-T_fit)/T_sim if T_sim>0 else float("nan")
            a_T   = unc["sigma_abs_T"]
            w.writerow([f"{t:.4f}",f"{d_bar:.6e}",f"{dose_air:.6e}",
                        f"{T_sim:.6e}",f"{T_fit:.6e}",f"{pct:.2f}",
                        f"{unc['sigma_rel_bar']*100:.3f}",
                        f"{unc['sigma_rel_air']*100:.3f}",
                        f"{unc['sigma_rel_T']*100:.3f}",f"{a_T:.4e}",
                        f"{nz:.2f}" if math.isfinite(nz) else "n/a",
                        "1" if in_fit else "0"])
    print(f"  CSV => {csv_path.name}")


_FIT_INFO_MARKER = "# === FIT INFO ==="

def write_fit_info(nuclide, barrier, output_dir, alpha, beta, gamma,
                    fvl_rows, source="auto",
                    sd_alpha=float("nan"), sd_beta=float("nan"),
                    sd_gamma=float("nan"), alpha_tail=float("nan"),
                    r2_tail=float("nan"), rmse_log10=float("nan"),
                    max_resid_pct=float("nan"), mu_nist=None,
                    fit_method="odr", gsa_result=None, pw_result=None,
                    thicknesses=None, transmissions=None):
    """
    Always-on persistence of everything the interactive tuner's "Save
    Params" button used to require a manual click to capture: alpha/beta/
    gamma, their ODR standard errors, fit-quality metrics, and the full
    HVL/TVL/CVL/MVL (Archer + local-bracket + delta%/accept) table.

    Appended directly onto the SAME {nuclide}_{barrier}_transmission_data.csv
    file that already holds the per-thickness transmission rows (per user
    request -- one file to open in Excel, not a separate JSON alongside it).
    write_transmission_csv() must have already been called for this
    (nuclide,barrier) in this output_dir before this function runs, since
    this appends rather than creates the file.

    Layout: the existing per-point data rows are left as-is (row 1 = header,
    rows 2..N = data, unchanged column count so nothing that already parses
    this CSV as a flat table breaks). Below that: one blank row, then a
    "# === FIT INFO ===" marker row, then key/value rows (alpha, beta,
    gamma, their std errors, alpha_tail, r2_tail, rmse_log10,
    max_resid_pct, all_fvl_accepted, source, saved_at), then a blank row,
    then a small FVL table (label, T_target, x_archer_mm, x_local_mm,
    delta_pct, accepted). Re-running a material re-reads the data rows,
    rewrites them unchanged, and re-appends a FRESH fit-info block --
    calling this twice never leaves two stacked fit-info sections.

    thicknesses/transmissions (optional): the FULL per-point arrays for this
    material. When gsa_result/pw_result is given AND these are also given,
    this function computes and persists the model-specific per-layer
    HVL/QVL/TVL/CVL/MVL table (GSA-vs-local / piecewise-vs-local) into the
    GSA/PIECEWISE DIAGNOSTIC block, via compute_fvl_with_local_gsa()/
    compute_fvl_with_local_piecewise() -- the same tables the interactive
    tuner's FVL panel shows on screen, previously NEVER persisted anywhere
    (a real gap: re-opening a saved "piecewise" or "gsa" fit_method result
    from fit_info_summary.csv only ever showed the plain single-term
    Archer-vs-local HVL/TVL/CVL/MVL numbers, never the GSA/piecewise ones,
    even though those are the numbers that actually matter for a fit_method
    the user deliberately chose because the standard fit wasn't good enough).
    If thicknesses/transmissions are omitted (e.g. an old call site not yet
    updated), gsa_fvl_rows/pw_fvl_rows are simply not written -- backward
    compatible, no crash.

    source: "auto" (written automatically at the end of analyze_one(), using
    the auto-fit alpha/beta/gamma — no user action required) or "manual"
    (written by the interactive tuner, either via the Save Params button or
    on window close, using whatever alpha/beta/gamma the sliders showed at
    that moment). A "manual" write always overwrites an "auto" one for the
    same material, since it reflects a human's deliberate final choice.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir/f"{nuclide}_{barrier}_transmission_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"write_fit_info() expects {csv_path.name} to already exist "
            f"(write_transmission_csv() runs first) -- got a missing file.")

    # Strip any previously-appended fit-info block (everything from the
    # marker onward), keeping only the original data rows, so re-saving
    # never stacks multiple fit-info sections in the same file.
    with open(csv_path, "r", newline="") as f:
        lines = f.readlines()
    marker_idx = next((i for i,l in enumerate(lines)
                        if l.startswith(_FIT_INFO_MARKER)), None)
    if marker_idx is not None:
        # also drop the blank separator row immediately before the marker,
        # if present, so trimming + re-appending doesn't grow a trail of
        # blank lines each time a material is re-saved.
        end = marker_idx
        if end > 0 and lines[end-1].strip() == "":
            end -= 1
        lines = lines[:end]
    data_text = "".join(lines).rstrip("\n") + "\n"

    all_ok = all_fvl_accepted(fvl_rows)
    alpha_over_tail = (alpha/alpha_tail
                        if math.isfinite(alpha_tail) and alpha_tail
                        else float("nan"))
    tail_over_mu = (alpha_tail/mu_nist if mu_nist else float("nan"))

    def _fmt(v, spec=".8f"):
        try:
            if v is None or (isinstance(v,float) and not math.isfinite(v)):
                return ""
        except TypeError:
            pass
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, float, np.floating, np.integer)):
            return format(float(v), spec)
        return str(v)

    with open(csv_path, "w", newline="") as f:
        f.write(data_text)
        w = csv.writer(f)
        w.writerow([])
        w.writerow([_FIT_INFO_MARKER])
        w.writerow(["source", source])
        w.writerow(["fit_method", fit_method])
        w.writerow(["saved_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        w.writerow(["alpha", _fmt(alpha)])
        w.writerow(["beta", _fmt(beta)])
        w.writerow(["gamma", _fmt(gamma)])
        w.writerow(["sd_alpha", _fmt(sd_alpha)])
        w.writerow(["sd_beta", _fmt(sd_beta)])
        w.writerow(["sd_gamma", _fmt(sd_gamma)])
        w.writerow(["alpha_tail", _fmt(alpha_tail)])
        w.writerow(["r2_tail", _fmt(r2_tail,".5f")])
        w.writerow(["alpha_over_alpha_tail", _fmt(alpha_over_tail,".5f")])
        w.writerow(["mu_nist", _fmt(mu_nist)])
        w.writerow(["alpha_tail_over_mu_nist", _fmt(tail_over_mu,".5f")])
        w.writerow(["rmse_log10", _fmt(rmse_log10,".6f")])
        w.writerow(["max_resid_pct", _fmt(max_resid_pct,".2f")])
        w.writerow(["all_fvl_accepted", "True" if all_ok else "False"])
        w.writerow([])
        w.writerow(["fvl_label","T_target","x_archer_mm","x_local_mm",
                    "method","r2_exp","r2_poly","delta_pct","accepted"])
        for r in fvl_rows:
            w.writerow([r["label"], _fmt(r["T_target"],".4f"),
                        _fmt(r["x_archer"],".4f"), _fmt(r["x_local"],".4f"),
                        r["method"], _fmt(r["r2_exp"],".5f"),
                        _fmt(r["r2_poly"],".5f"), _fmt(r["delta_pct"],".2f"),
                        "True" if r["accepted"] else "False"])

        # GSA diagnostic block -- only written when fit_method="gsa" was
        # requested AND the comparison actually produced a result (gsa_result
        # is None if gsa_fit.py wasn't importable or the fit failed; either
        # way this block is simply omitted, not written with blank/garbage
        # values). Falls INSIDE the same marker-delimited section as
        # everything above, so the "strip from marker onward, then
        # re-append fresh" logic at the top of this function replaces this
        # block too on every re-save -- no separate stacking risk.
        if gsa_result is not None:
            w.writerow([])
            w.writerow(["# === GSA DIAGNOSTIC ==="])
            w.writerow(["gsa_verdict", gsa_result.verdict])
            w.writerow(["gsa_K", gsa_result.K])
            w.writerow(["gsa_anchored", "True" if gsa_result.anchored else "False"])
            w.writerow(["gsa_n", gsa_result.n])
            w.writerow(["gsa_n_dropped", gsa_result.n_dropped])
            w.writerow(["gsa_x_c_mm",
                        "" if math.isinf(gsa_result.x_c) else _fmt(gsa_result.x_c,".3f")])
            w.writerow(["gsa_x_max_data_mm", _fmt(gsa_result.x_max_data,".3f")])
            w.writerow(["gsa_rmse_std", _fmt(gsa_result.rmse_std,".6f")])
            w.writerow(["gsa_rmse_gsa", _fmt(gsa_result.rmse_gsa,".6f")])
            w.writerow(["gsa_maxerr_std_pct", _fmt(gsa_result.maxerr_std_pct,".2f")])
            w.writerow(["gsa_maxerr_gsa_pct", _fmt(gsa_result.maxerr_gsa_pct,".2f")])
            w.writerow(["gsa_dAICc", _fmt(gsa_result.dAICc,".3f")])
            w.writerow(["gsa_std_degenerate",
                        "True" if gsa_result.std_degenerate else "False"])
            w.writerow([])
            if gsa_result.gsa_params:
                w.writerow(["gsa_group","weight","alpha_mm-1","beta","gamma"])
                for k, (wk, ak, bk, gk) in enumerate(gsa_result.gsa_params):
                    w.writerow([k, _fmt(wk,".5f"), _fmt(ak,".6f"),
                                _fmt(bk,".6f"), _fmt(gk,".4f")])
                # Blank separator so _read_fit_info_block() knows the
                # gsa_group table ends here -- without it, the dx_at_B rows
                # below (2 columns each) get misread as more group rows.
                w.writerow([])
            for B, (dstd, dgsa) in gsa_result.dx_at_B.items():
                w.writerow([f"gsa_dx_at_B{B:g}_std_mm", _fmt(dstd,".4f")])
                w.writerow([f"gsa_dx_at_B{B:g}_gsa_mm", _fmt(dgsa,".4f")])

            # Per-layer GSA-vs-local HVL/QVL/TVL/CVL/MVL table -- the same
            # numbers the interactive tuner's FVL panel shows on screen
            # (compute_fvl_with_local_gsa()), now actually persisted. Only
            # written when thicknesses/transmissions were supplied AND the
            # comparison produces rows (gsa_params non-empty, GSA_OK True).
            if thicknesses is not None and transmissions is not None:
                gsa_fvl_rows = compute_fvl_with_local_gsa(
                    thicknesses, transmissions, gsa_result.gsa_params)
                if gsa_fvl_rows:
                    w.writerow([])
                    w.writerow(["gsa_fvl_label","T_target","x_gsa_mm",
                                "x_local_mm","method","delta_pct","accepted"])
                    for r in gsa_fvl_rows:
                        w.writerow([r["label"], _fmt(r["T_target"],".4f"),
                                    _fmt(r["x_gsa"],".4f"), _fmt(r["x_local"],".4f"),
                                    r["method"], _fmt(r["delta_pct"],".2f"),
                                    "True" if r["accepted"] else "False"])
                    # Blank separator, same convention as gsa_group above --
                    # keeps _read_fit_info_block() from reading anything
                    # appended after this as more gsa_fvl_label rows.
                    w.writerow([])

        # Piecewise diagnostic block -- same "only written when requested
        # and successful" convention as the GSA block above, same marker-
        # delimited placement so it's replaced (not stacked) on every
        # re-save.
        if pw_result is not None:
            w.writerow([])
            w.writerow(["# === PIECEWISE DIAGNOSTIC ==="])
            w.writerow(["pw_verdict", pw_result.verdict])
            w.writerow(["pw_n", pw_result.n])
            w.writerow(["pw_n_dropped", pw_result.n_dropped])
            w.writerow(["pw_x_star_mm", _fmt(pw_result.x_star,".3f")])
            w.writerow(["pw_n_thin", pw_result.n_thin])
            w.writerow(["pw_n_thick", pw_result.n_thick])
            w.writerow(["pw_seam_continuous",
                        "True" if pw_result.seam_continuous else "False"])
            w.writerow(["pw_rmse_std", _fmt(pw_result.rmse_std,".6f")])
            w.writerow(["pw_rmse_piecewise", _fmt(pw_result.rmse_piecewise,".6f")])
            w.writerow(["pw_maxerr_std_pct", _fmt(pw_result.maxerr_std_pct,".2f")])
            w.writerow(["pw_maxerr_piecewise_pct", _fmt(pw_result.maxerr_piecewise_pct,".2f")])
            w.writerow(["pw_dAICc", _fmt(pw_result.dAICc,".3f")])
            w.writerow(["pw_low_dof", "True" if pw_result.low_dof else "False"])
            w.writerow([])
            if all(math.isfinite(v) for v in pw_result.thin_params):
                w.writerow(["pw_side","alpha_mm-1","beta","gamma"])
                w.writerow(["thin", _fmt(pw_result.thin_params[0],".6f"),
                            _fmt(pw_result.thin_params[1],".6f"),
                            _fmt(pw_result.thin_params[2],".4f")])
                w.writerow(["thick", _fmt(pw_result.thick_params[0],".6f"),
                            _fmt(pw_result.thick_params[1],".6f"),
                            _fmt(pw_result.thick_params[2],".4f")])
                # Blank separator, matching the gsa_group table's convention
                # -- keeps _read_fit_info_block() from mis-reading anything
                # appended after this block as more pw_side rows.
                w.writerow([])

            # Per-layer piecewise-vs-local HVL/QVL/TVL/CVL/MVL table -- the
            # same numbers the interactive tuner's FVL panel shows on screen
            # (compute_fvl_with_local_piecewise()), now actually persisted.
            if (thicknesses is not None and transmissions is not None
                    and math.isfinite(pw_result.x_star)):
                pw_fvl_rows = compute_fvl_with_local_piecewise(
                    thicknesses, transmissions, pw_result)
                if pw_fvl_rows:
                    w.writerow(["pw_fvl_label","T_target","x_piecewise_mm",
                                "x_local_mm","method","delta_pct","accepted"])
                    for r in pw_fvl_rows:
                        w.writerow([r["label"], _fmt(r["T_target"],".4f"),
                                    _fmt(r["x_piecewise"],".4f"), _fmt(r["x_local"],".4f"),
                                    r["method"], _fmt(r["delta_pct"],".2f"),
                                    "True" if r["accepted"] else "False"])
                    # Blank separator, same convention as pw_side above --
                    # keeps _read_fit_info_block() from reading anything
                    # appended after this as more pw_fvl_label rows.
                    w.writerow([])

    print(f"  Fit info appended => {csv_path.name}  ({source})")
    if gsa_result is not None:
        print(f"  GSA diagnostic => verdict={gsa_result.verdict!r}  "
              f"K={gsa_result.K}  dAICc={gsa_result.dAICc}")
    if pw_result is not None:
        print(f"  Piecewise diagnostic => verdict={pw_result.verdict!r}  "
              f"x_star={pw_result.x_star:.2f}mm  "
              f"rmse_std={pw_result.rmse_std:.5f}  "
              f"rmse_piecewise={pw_result.rmse_piecewise:.5f}  "
              f"dAICc={pw_result.dAICc}")

    _rebuild_fit_info_summary(output_dir)
    return {"nuclide":nuclide,"barrier":barrier,"source":source,
            "alpha":alpha,"beta":beta,"gamma":gamma,
            "all_fvl_accepted":all_ok,"fvl_rows":fvl_rows}


def _read_fit_info_block(csv_path):
    """
    Parse the fit-info block appended by write_fit_info() back out of a
    {nuclide}_{barrier}_transmission_data.csv file. Returns None if the
    file has no fit-info section yet (e.g. write_transmission_csv() ran
    but write_fit_info() hasn't, or an old-format file without one).
    """
    with open(csv_path, "r", newline="") as f:
        lines = f.readlines()
    marker_idx = next((i for i,l in enumerate(lines)
                        if l.startswith(_FIT_INFO_MARKER)), None)
    if marker_idx is None:
        return None

    rest = "".join(lines[marker_idx+1:])
    reader = csv.reader(rest.splitlines())
    kv = {}
    fvl_rows = []
    gsa_group_rows = []
    pw_side_rows = []
    gsa_fvl_rows = []
    pw_fvl_rows = []
    in_fvl_table = False
    in_gsa_group_table = False
    in_pw_side_table = False
    in_gsa_fvl_table = False
    in_pw_fvl_table = False
    fvl_header = None
    gsa_group_header = None
    pw_side_header = None
    gsa_fvl_header = None
    pw_fvl_header = None
    for row in reader:
        if not row:
            # A blank row ends whichever sub-table (if any) is currently
            # open -- without this, in_fvl_table/in_gsa_group_table stay
            # True forever once set, so every later row (including the
            # "# === GSA DIAGNOSTIC ===" marker, its own key/value rows,
            # and the gsa_group header) gets mis-appended into fvl_rows
            # keyed against the stale fvl_header instead of being read as
            # the distinct sections they are.
            in_fvl_table = False
            in_gsa_group_table = False
            in_pw_side_table = False
            in_gsa_fvl_table = False
            in_pw_fvl_table = False
            continue
        if row[0] in ("# === GSA DIAGNOSTIC ===", "# === PIECEWISE DIAGNOSTIC ==="):
            in_fvl_table = False
            in_gsa_group_table = False
            in_pw_side_table = False
            in_gsa_fvl_table = False
            in_pw_fvl_table = False
            continue
        if row[0] == "fvl_label":
            in_fvl_table = True
            in_gsa_group_table = False
            in_pw_side_table = False
            in_gsa_fvl_table = False
            in_pw_fvl_table = False
            fvl_header = row
            continue
        if row[0] == "gsa_group":
            in_gsa_group_table = True
            in_fvl_table = False
            in_pw_side_table = False
            in_gsa_fvl_table = False
            in_pw_fvl_table = False
            gsa_group_header = row
            continue
        if row[0] == "pw_side":
            in_pw_side_table = True
            in_fvl_table = False
            in_gsa_group_table = False
            in_gsa_fvl_table = False
            in_pw_fvl_table = False
            pw_side_header = row
            continue
        if row[0] == "gsa_fvl_label":
            in_gsa_fvl_table = True
            in_fvl_table = False
            in_gsa_group_table = False
            in_pw_side_table = False
            in_pw_fvl_table = False
            gsa_fvl_header = row
            continue
        if row[0] == "pw_fvl_label":
            in_pw_fvl_table = True
            in_fvl_table = False
            in_gsa_group_table = False
            in_pw_side_table = False
            in_gsa_fvl_table = False
            pw_fvl_header = row
            continue
        if in_fvl_table:
            fvl_rows.append(dict(zip(fvl_header, row)))
        elif in_gsa_group_table:
            # Defensive check, not just reliant on the blank-row separator
            # write_fit_info() now emits: a valid group row's first cell is
            # always an integer group index. Anything else (e.g. a
            # "gsa_dx_at_B..." key/value row immediately following the
            # table, written by an older file without the separator) ends
            # the table instead of being mis-parsed as a group row.
            try:
                int(row[0])
                gsa_group_rows.append(dict(zip(gsa_group_header, row)))
            except (ValueError, IndexError):
                in_gsa_group_table = False
                if len(row) >= 2:
                    kv[row[0]] = row[1]
        elif in_pw_side_table:
            # Same defensive pattern: a valid pw_side row's first cell is
            # always "thin" or "thick".
            if row and row[0] in ("thin", "thick"):
                pw_side_rows.append(dict(zip(pw_side_header, row)))
            else:
                in_pw_side_table = False
                if len(row) >= 2:
                    kv[row[0]] = row[1]
        elif in_gsa_fvl_table:
            # Defensive check: a valid gsa_fvl row's first cell is always
            # one of the FVL_TARGETS labels (HVL/QVL/TVL/CVL/MVL).
            if row and row[0] in FVL_TARGETS:
                gsa_fvl_rows.append(dict(zip(gsa_fvl_header, row)))
            else:
                in_gsa_fvl_table = False
                if len(row) >= 2:
                    kv[row[0]] = row[1]
        elif in_pw_fvl_table:
            if row and row[0] in FVL_TARGETS:
                pw_fvl_rows.append(dict(zip(pw_fvl_header, row)))
            else:
                in_pw_fvl_table = False
                if len(row) >= 2:
                    kv[row[0]] = row[1]
        elif len(row) >= 2:
            kv[row[0]] = row[1]

    def _num(v):
        try: return float(v)
        except (TypeError, ValueError): return float("nan")

    info = {
        "nuclide": None, "barrier": None,  # filled in by caller from filename
        "source": kv.get("source"),
        "fit_method": kv.get("fit_method"),
        "saved_at": kv.get("saved_at"),
        "alpha": _num(kv.get("alpha")), "beta": _num(kv.get("beta")),
        "gamma": _num(kv.get("gamma")),
        "sd_alpha": _num(kv.get("sd_alpha")), "sd_beta": _num(kv.get("sd_beta")),
        "sd_gamma": _num(kv.get("sd_gamma")),
        "rmse_log10": _num(kv.get("rmse_log10")),
        "max_resid_pct": _num(kv.get("max_resid_pct")),
        "all_fvl_accepted": kv.get("all_fvl_accepted") == "True",
        "fvl": {
            r["fvl_label"]: {
                "x_archer_mm": _num(r.get("x_archer_mm")),
                "x_local_mm": _num(r.get("x_local_mm")),
                "delta_pct": _num(r.get("delta_pct")),
                "accepted": r.get("accepted") == "True",
            }
            for r in fvl_rows
        },
        # GSA diagnostic (only present when fit_method="gsa" was used and
        # the comparison succeeded -- see write_fit_info()'s "GSA
        # DIAGNOSTIC" block). "groups" holds every (weight, alpha, beta,
        # gamma) group actually written, in group-index order -- NOT
        # truncated to two, so a future K=3 fit still round-trips fully.
        "gsa_verdict": kv.get("gsa_verdict"),
        "gsa_K": kv.get("gsa_K"),
        "gsa_anchored": kv.get("gsa_anchored") == "True",
        "gsa_x_c_mm": _num(kv.get("gsa_x_c_mm")),
        "gsa_x_max_data_mm": _num(kv.get("gsa_x_max_data_mm")),
        "gsa_rmse_std": _num(kv.get("gsa_rmse_std")),
        "gsa_rmse_gsa": _num(kv.get("gsa_rmse_gsa")),
        "gsa_maxerr_std_pct": _num(kv.get("gsa_maxerr_std_pct")),
        "gsa_maxerr_gsa_pct": _num(kv.get("gsa_maxerr_gsa_pct")),
        "gsa_dAICc": _num(kv.get("gsa_dAICc")),
        "gsa_groups": [
            {
                "k": int(_num(r.get("gsa_group"))),
                "weight": _num(r.get("weight")),
                "alpha_mm-1": _num(r.get("alpha_mm-1")),
                "beta": _num(r.get("beta")),
                "gamma": _num(r.get("gamma")),
            }
            for r in gsa_group_rows
        ],
        # GSA-vs-local per-layer FVL table (only present when
        # thicknesses/transmissions were passed to write_fit_info() AND the
        # GSA fit produced groups -- see the "gsa_fvl_label" table above).
        # Keyed by label (HVL/QVL/TVL/CVL/MVL), same shape as "fvl" above
        # but with "x_gsa_mm" instead of "x_archer_mm".
        "gsa_fvl": {
            r["gsa_fvl_label"]: {
                "x_gsa_mm": _num(r.get("x_gsa_mm")),
                "x_local_mm": _num(r.get("x_local_mm")),
                "delta_pct": _num(r.get("delta_pct")),
                "accepted": r.get("accepted") == "True",
            }
            for r in gsa_fvl_rows
        },
        # Piecewise diagnostic (only present when fit_method="piecewise"
        # was used and the fit succeeded -- see write_fit_info()'s
        # "PIECEWISE DIAGNOSTIC" block). "sides" holds the thin/thick
        # triples in that fixed order when present.
        "pw_verdict": kv.get("pw_verdict"),
        "pw_x_star_mm": _num(kv.get("pw_x_star_mm")),
        "pw_n_thin": kv.get("pw_n_thin"),
        "pw_n_thick": kv.get("pw_n_thick"),
        "pw_seam_continuous": kv.get("pw_seam_continuous") == "True",
        "pw_rmse_std": _num(kv.get("pw_rmse_std")),
        "pw_rmse_piecewise": _num(kv.get("pw_rmse_piecewise")),
        "pw_maxerr_std_pct": _num(kv.get("pw_maxerr_std_pct")),
        "pw_maxerr_piecewise_pct": _num(kv.get("pw_maxerr_piecewise_pct")),
        "pw_dAICc": _num(kv.get("pw_dAICc")),
        "pw_low_dof": kv.get("pw_low_dof") == "True",
        "pw_sides": {
            r["pw_side"]: {
                "alpha_mm-1": _num(r.get("alpha_mm-1")),
                "beta": _num(r.get("beta")),
                "gamma": _num(r.get("gamma")),
            }
            for r in pw_side_rows
        },
        # Piecewise-vs-local per-layer FVL table (only present when
        # thicknesses/transmissions were passed to write_fit_info() AND the
        # cutoff grid search converged -- see the "pw_fvl_label" table
        # above). Keyed by label, same shape as "fvl"/"gsa_fvl" but with
        # "x_piecewise_mm".
        "pw_fvl": {
            r["pw_fvl_label"]: {
                "x_piecewise_mm": _num(r.get("x_piecewise_mm")),
                "x_local_mm": _num(r.get("x_local_mm")),
                "delta_pct": _num(r.get("delta_pct")),
                "accepted": r.get("accepted") == "True",
            }
            for r in pw_fvl_rows
        },
    }
    return info


def _rebuild_fit_info_summary(output_dir):
    """
    Recompute {output_dir}/fit_info_summary.csv from the fit-info block
    appended to every *_transmission_data.csv present in output_dir.
    Rebuilding from scratch each time (rather than appending) means the
    summary always reflects each material's latest saved fit — auto or
    manual — with no stale/duplicate rows even across multiple
    analyze_one()/tuner sessions run in any order.
    """
    output_dir = Path(output_dir)
    csv_files = sorted(output_dir.glob("*_transmission_data.csv"))
    if not csv_files:
        return

    rows = []
    fvl_labels = []
    gsa_fvl_labels = []
    pw_fvl_labels = []
    max_gsa_groups = 0
    for cf in csv_files:
        try:
            info = _read_fit_info_block(cf)
        except Exception as e:
            print(f"  ! skipping {cf.name} in fit_info_summary: {e}")
            continue
        if info is None:
            continue
        # {nuclide}_{barrier}_transmission_data.csv -- barrier names are
        # single tokens (Lead, Glass, ...) so this split is unambiguous.
        stem_parts = cf.stem.replace("_transmission_data","").split("_")
        nuclide = stem_parts[0]; barrier = "_".join(stem_parts[1:])
        row = {
            "nuclide": nuclide,
            "barrier": barrier,
            "source": info.get("source"),
            "fit_method": info.get("fit_method"),
            "saved_at": info.get("saved_at"),
            "alpha": info.get("alpha"), "beta": info.get("beta"),
            "gamma": info.get("gamma"),
            "sd_alpha": info.get("sd_alpha"), "sd_beta": info.get("sd_beta"),
            "sd_gamma": info.get("sd_gamma"),
            "rmse_log10": info.get("rmse_log10"),
            "max_resid_pct": info.get("max_resid_pct"),
            "all_fvl_accepted": info.get("all_fvl_accepted"),
        }
        for label, fvl in info.get("fvl", {}).items():
            if label not in fvl_labels: fvl_labels.append(label)
            row[f"{label}_archer_mm"] = fvl.get("x_archer_mm")
            row[f"{label}_local_mm"]  = fvl.get("x_local_mm")
            row[f"{label}_delta_pct"] = fvl.get("delta_pct")
            row[f"{label}_accepted"]  = fvl.get("accepted")

        # GSA diagnostic columns -- present only for materials fit with
        # fit_method="gsa" where the comparison succeeded; other rows just
        # get blanks for these fields (DictWriter fills missing keys with
        # restval="", the default). Previously this whole block was
        # missing, so fit_info_summary.csv silently dropped every GSA
        # group's (weight, alpha, beta, gamma) even though write_fit_info()
        # had correctly written all of them into the per-material
        # *_transmission_data.csv -- the summary just never read them back.
        if info.get("gsa_verdict") is not None:
            row["gsa_verdict"] = info.get("gsa_verdict")
            row["gsa_K"] = info.get("gsa_K")
            row["gsa_anchored"] = info.get("gsa_anchored")
            row["gsa_x_c_mm"] = info.get("gsa_x_c_mm")
            row["gsa_x_max_data_mm"] = info.get("gsa_x_max_data_mm")
            row["gsa_rmse_std"] = info.get("gsa_rmse_std")
            row["gsa_rmse_gsa"] = info.get("gsa_rmse_gsa")
            row["gsa_maxerr_std_pct"] = info.get("gsa_maxerr_std_pct")
            row["gsa_maxerr_gsa_pct"] = info.get("gsa_maxerr_gsa_pct")
            row["gsa_dAICc"] = info.get("gsa_dAICc")
            groups = info.get("gsa_groups") or []
            max_gsa_groups = max(max_gsa_groups, len(groups))
            for grp in groups:
                k = grp["k"]
                row[f"gsa_g{k}_weight"] = grp["weight"]
                row[f"gsa_g{k}_alpha_mm-1"] = grp["alpha_mm-1"]
                row[f"gsa_g{k}_beta"] = grp["beta"]
                row[f"gsa_g{k}_gamma"] = grp["gamma"]

            # GSA-vs-local per-layer HVL/QVL/TVL/CVL/MVL columns -- these
            # were previously computed by the interactive tuner for on-
            # screen display only and never made it into fit_info_summary
            # (or even the per-material CSV) at all. Same "{label}_..."
            # naming convention as the plain fvl_labels columns above, just
            # prefixed "gsa_" and using "x_gsa_mm" as the model-side value.
            for label, gfvl in info.get("gsa_fvl", {}).items():
                if label not in gsa_fvl_labels: gsa_fvl_labels.append(label)
                row[f"gsa_{label}_x_gsa_mm"] = gfvl.get("x_gsa_mm")
                row[f"gsa_{label}_local_mm"] = gfvl.get("x_local_mm")
                row[f"gsa_{label}_delta_pct"] = gfvl.get("delta_pct")
                row[f"gsa_{label}_accepted"] = gfvl.get("accepted")

        # Piecewise diagnostic columns -- same "present only when that fit
        # method was used and succeeded" convention as the GSA block above.
        # _blank_nan() converts a NaN float (e.g. x_star when the cutoff
        # grid search failed / n was too small) to "" rather than letting
        # csv.DictWriter stringify it as the literal text "nan" -- without
        # this, a degenerate piecewise result would round-trip through
        # fit_info_summary.csv as the string "nan" instead of a genuinely
        # empty cell, which reads as a real (bogus) value in a spreadsheet.
        def _blank_nan(v):
            return "" if isinstance(v, float) and math.isnan(v) else v
        if info.get("pw_verdict") is not None:
            row["pw_verdict"] = info.get("pw_verdict")
            row["pw_x_star_mm"] = _blank_nan(info.get("pw_x_star_mm"))
            row["pw_n_thin"] = info.get("pw_n_thin")
            row["pw_n_thick"] = info.get("pw_n_thick")
            row["pw_seam_continuous"] = info.get("pw_seam_continuous")
            row["pw_rmse_std"] = _blank_nan(info.get("pw_rmse_std"))
            row["pw_rmse_piecewise"] = _blank_nan(info.get("pw_rmse_piecewise"))
            row["pw_maxerr_std_pct"] = _blank_nan(info.get("pw_maxerr_std_pct"))
            row["pw_maxerr_piecewise_pct"] = _blank_nan(info.get("pw_maxerr_piecewise_pct"))
            row["pw_dAICc"] = _blank_nan(info.get("pw_dAICc"))
            row["pw_low_dof"] = info.get("pw_low_dof")
            sides = info.get("pw_sides") or {}
            for side in ("thin","thick"):
                s = sides.get(side, {})
                row[f"pw_{side}_alpha_mm-1"] = _blank_nan(s.get("alpha_mm-1"))
                row[f"pw_{side}_beta"] = _blank_nan(s.get("beta"))
                row[f"pw_{side}_gamma"] = _blank_nan(s.get("gamma"))

            # Piecewise-vs-local per-layer HVL/QVL/TVL/CVL/MVL columns --
            # same gap/fix as the gsa_{label}_* columns above: the tuner
            # computed these for display only, they were never persisted.
            for label, pfvl in info.get("pw_fvl", {}).items():
                if label not in pw_fvl_labels: pw_fvl_labels.append(label)
                row[f"pw_{label}_x_piecewise_mm"] = _blank_nan(pfvl.get("x_piecewise_mm"))
                row[f"pw_{label}_local_mm"] = _blank_nan(pfvl.get("x_local_mm"))
                row[f"pw_{label}_delta_pct"] = _blank_nan(pfvl.get("delta_pct"))
                row[f"pw_{label}_accepted"] = pfvl.get("accepted")
        rows.append(row)

    if not rows:
        return

    fieldnames = ["nuclide","barrier","source","fit_method","saved_at",
                  "alpha","beta","gamma","sd_alpha","sd_beta","sd_gamma",
                  "rmse_log10","max_resid_pct","all_fvl_accepted"]
    for label in fvl_labels:
        fieldnames += [f"{label}_archer_mm", f"{label}_local_mm",
                       f"{label}_delta_pct", f"{label}_accepted"]
    if max_gsa_groups > 0:
        fieldnames += ["gsa_verdict","gsa_K","gsa_anchored","gsa_x_c_mm",
                       "gsa_x_max_data_mm","gsa_rmse_std","gsa_rmse_gsa",
                       "gsa_maxerr_std_pct","gsa_maxerr_gsa_pct","gsa_dAICc"]
        for k in range(max_gsa_groups):
            fieldnames += [f"gsa_g{k}_weight", f"gsa_g{k}_alpha_mm-1",
                           f"gsa_g{k}_beta", f"gsa_g{k}_gamma"]
        for label in gsa_fvl_labels:
            fieldnames += [f"gsa_{label}_x_gsa_mm", f"gsa_{label}_local_mm",
                           f"gsa_{label}_delta_pct", f"gsa_{label}_accepted"]
    if any(r.get("pw_verdict") is not None for r in rows):
        fieldnames += ["pw_verdict","pw_x_star_mm","pw_n_thin","pw_n_thick",
                       "pw_seam_continuous","pw_rmse_std","pw_rmse_piecewise",
                       "pw_maxerr_std_pct","pw_maxerr_piecewise_pct",
                       "pw_dAICc","pw_low_dof"]
        for side in ("thin","thick"):
            fieldnames += [f"pw_{side}_alpha_mm-1", f"pw_{side}_beta",
                           f"pw_{side}_gamma"]
        for label in pw_fvl_labels:
            fieldnames += [f"pw_{label}_x_piecewise_mm", f"pw_{label}_local_mm",
                           f"pw_{label}_delta_pct", f"pw_{label}_accepted"]

    out = output_dir/"fit_info_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Fit info summary => {out.name}  ({len(rows)} material(s))")


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
                              sd_gamma=float("nan"),
                              gsa_result=None, pw_result=None):
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

    # Clean up any previous figures and force interactive backend
    plt.close('all')
    plt.switch_backend('TkAgg')

    # ── hover-to-inspect-a-point infrastructure ─────────────────────────
    # ax_main/ax_resid are fully torn down and rebuilt (ax.cla()) on every
    # slider move via _draw_main()/_draw_resid() below, so the hover
    # annotation artist can't be created once up front -- it has to be
    # (re)created inside each draw function, right after cla(), and the
    # data each axis's hover handler searches has to come from a small
    # mutable registry (_hover_data) that _draw_main()/_draw_resid()
    # refresh every redraw rather than a fixed dict captured at setup time.
    _hover_data = {}   # ax -> dict(x=array, y=array, fmt=callable(i)->str)
    _hover_ann  = {}   # ax -> the annotation artist currently live on that ax

    def _make_hover_annotation(ax):
        ann = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#997700",
                      alpha=0.95, lw=0.8),
            fontsize=8, fontfamily="monospace", zorder=50, visible=False)
        _hover_ann[ax] = ann
        return ann

    def _on_hover_move(event):
        ax = event.inaxes
        if ax is None or ax not in _hover_data or ax not in _hover_ann:
            # cursor left every tracked axis (or is over a slider/button) --
            # hide any annotation that's currently showing, then bail.
            changed = False
            for a2, ann2 in _hover_ann.items():
                if ann2.get_visible():
                    ann2.set_visible(False)
                    changed = True
            if changed:
                fig.canvas.draw_idle()
            return

        data = _hover_data[ax]
        xs, ys = data["x"], data["y"]
        ann = _hover_ann[ax]
        if len(xs) == 0 or event.xdata is None or event.ydata is None:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return

        # Nearest point in DISPLAY (pixel) space, not data space -- data
        # space is wrong here because the Y axis is log-scaled and X/Y
        # units are wildly different (mm vs dimensionless T), so naive
        # data-space distance would make the log-compressed low-T points
        # nearly impossible to hover accurately. Transform both the cursor
        # and every candidate point through the same axes transform first.
        try:
            pts_disp = ax.transData.transform(np.column_stack([xs, ys]))
        except Exception:
            return
        cursor_disp = np.array([event.x, event.y])
        d2 = np.sum((pts_disp - cursor_disp) ** 2, axis=1)
        i_near = int(np.argmin(d2))
        # 15 px hit radius -- close enough to feel responsive, far enough
        # not to fight with the fit-curve lines drawn through the same region.
        if d2[i_near] > 15 ** 2:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return

        ann.xy = (xs[i_near], ys[i_near])
        ann.set_text(data["fmt"](i_near))
        ann.set_visible(True)
        fig.canvas.draw_idle()

    # motion_notify_event is connected once, further down, right after
    # `fig` exists and the other mpl_connect('close_event', ...) call.

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
        # GSA overlay -- only drawn when this material was fit with
        # fit_method="gsa" and the diagnostic comparison actually produced
        # group parameters (gsa_result.gsa_params is None for K=1/"std
        # adequate" verdicts, a failed fit, or n<7 points -- see
        # gsa_fit.fit_transmission()). Previously launch_interactive_tuner()
        # had no gsa_result parameter at all, so nothing GSA-related could
        # ever be drawn here regardless of what analyze_one() had already
        # computed -- this is the fix for that gap.
        if gsa_result is not None and gsa_result.gsa_params:
            y_gsa = None
            try:
                y_gsa = _gsa.gsa_transmission(t_sm, gsa_result.gsa_params)
                y_gsa = np.where(np.isfinite(y_gsa)&(y_gsa>0)&(y_gsa<=2.0),
                                  y_gsa, np.nan)
            except Exception:
                y_gsa = None
            if y_gsa is not None:
                ax.semilogy(t_sm,y_gsa,"-",color="#0066cc",lw=2.2,
                            zorder=7,
                            label=f"GSA (K={gsa_result.K}, "
                                  f"{gsa_result.verdict})")
                # Faint per-group component curves -- shows WHY the blend
                # bends where it does (e.g. the hard, low-weight group
                # taking over once the soft group has died off), which the
                # single blended line alone doesn't make visually obvious.
                for k,(w,ga,gb,gg) in enumerate(gsa_result.gsa_params):
                    yk = _safe(t_sm,ga,gb,gg)
                    if yk is not None:
                        ax.semilogy(t_sm,w*yk,"--",color="#66a3e0",lw=1.0,
                                    alpha=0.65,zorder=4,
                                    label=f"  GSA group {k} "
                                          f"(w={w:.4f}, a={ga:.5f})")
                # Asymptotic breakpoint (crossover thickness x_c) -- the
                # physics-derived thickness beyond which the harder/more-
                # penetrating group's transmission exceeds the softer
                # group's, i.e. where the curve's controlling term switches
                # and a single-Archer-term description stops being
                # physically adequate (see gsa_fit.crossover_xc()). This is
                # x_c as computed from kerma fractions/mu BEFORE fitting --
                # it's what decided whether K=2 was even attempted, not a
                # post-hoc read off the fitted curve, so it can legitimately
                # sit outside the fitted curve's own visual "bend" a little.
                x_c = gsa_result.x_c
                if math.isfinite(x_c) and 0 < x_c <= t_arr.max()*1.10:
                    ax.axvline(x_c,color="#009966",lw=1.4,ls="-.",
                                alpha=0.75,zorder=6,
                                label=f"GSA breakpoint x_c={x_c:.2f}mm")
                    ax.text(x_c,1.4,f" x_c={x_c:.2f}mm",rotation=90,
                            fontsize=7,color="#006644",va="top",ha="left",
                            alpha=0.85)
                elif math.isfinite(x_c):
                    # x_c falls outside the plotted/data range -- still
                    # worth surfacing (e.g. in the title) rather than
                    # silently omitting it, since "no marker shown" could
                    # otherwise read as "no breakpoint exists" when really
                    # it's just off-screen relative to this material's data.
                    print(f"  (GSA breakpoint x_c={x_c:.2f}mm falls outside "
                          f"the plotted thickness range "
                          f"[0, {t_arr.max()*1.10:.1f}mm] -- not marked on "
                          f"plot)")
        # Piecewise (thin/thick) overlay -- only drawn when this material
        # was fit with fit_method="piecewise" and the grid search converged
        # (pw_result.x_star finite). Same rationale as the GSA overlay
        # above: shows both independently-invertible triples plus the
        # cutoff x_star where the model switches from one to the other, so
        # the tuner communicates not just "there's a piecewise fit" but
        # exactly where it breaks and how well each side tracks the data.
        if pw_result is not None and math.isfinite(pw_result.x_star):
            thin_mask_x = t_sm <= pw_result.x_star
            thick_mask_x = ~thin_mask_x
            y_pw = np.full_like(t_sm, np.nan)
            if thin_mask_x.any():
                yk = _safe(t_sm[thin_mask_x], *pw_result.thin_params)
                if yk is not None:
                    y_pw[thin_mask_x] = yk
            if thick_mask_x.any():
                yk = _safe(t_sm[thick_mask_x], *pw_result.thick_params)
                if yk is not None:
                    y_pw[thick_mask_x] = yk
            if np.isfinite(y_pw).any():
                ax.semilogy(t_sm,y_pw,"-",color="#cc6600",lw=2.2,zorder=7,
                            label=f"Piecewise ({pw_result.verdict})")
            if 0 < pw_result.x_star <= t_arr.max()*1.10:
                ax.axvline(pw_result.x_star,color="#cc6600",lw=1.4,ls=":",
                            alpha=0.8,zorder=6,
                            label=f"Piecewise cutoff x*={pw_result.x_star:.2f}mm")
                ax.text(pw_result.x_star,0.6,f" x*={pw_result.x_star:.2f}mm",
                        rotation=90,fontsize=7,color="#994400",va="top",
                        ha="left",alpha=0.85)
            else:
                print(f"  (Piecewise cutoff x*={pw_result.x_star:.2f}mm falls "
                      f"outside the plotted thickness range "
                      f"[0, {t_arr.max()*1.10:.1f}mm] -- not marked on plot)")
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
        gsa_s = ""
        if gsa_result is not None:
            dstr = (f"  dAICc={gsa_result.dAICc:.2f}"
                    if gsa_result.dAICc is not None else "")
            xc_str = (f"  x_c={gsa_result.x_c:.2f}mm"
                      if math.isfinite(gsa_result.x_c) else "  x_c=inf")
            gsa_s = (f"\nGSA: verdict={gsa_result.verdict!r}  "
                     f"K={gsa_result.K}{dstr}{xc_str}")
        pw_s = ""
        if pw_result is not None:
            xstar_str = (f"  x*={pw_result.x_star:.2f}mm"
                         if math.isfinite(pw_result.x_star) else "  x*=n/a")
            pw_dstr = (f"  dAICc={pw_result.dAICc:.2f}"
                       if math.isfinite(pw_result.dAICc) else "")
            pw_s = (f"\nPiecewise: verdict={pw_result.verdict!r}"
                    f"{xstar_str}{pw_dstr}  RMSE std/pw="
                    f"{pw_result.rmse_std:.4f}/{pw_result.rmse_piecewise:.4f}")
        ax.set_title(
            f"{nuclide}  {barrier}\n"
            f"a={a:.6f}  b={b:.6f}  g={g:.6f}"
            f"    a_tail={alpha_tail:.6f}{r2s}{gsa_s}{pw_s}",
            fontsize=9.5,pad=6)
        ax.set_ylabel("Transmission T",fontsize=10)
        ax.set_xlim(0, t_arr.max() * 1.10)
        ax.set_ylim(1e-5,2.0); ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.legend(fontsize=8,loc="upper right",framealpha=0.88)
        ax.grid(True,which="both",ls=":",alpha=0.35)
        plt.setp(ax.get_xticklabels(),visible=False)

        # Hover data: every REAL plotted simulation point (both used-in-fit
        # and excluded markers -- both are visible on this axis), keyed by
        # their exact (thickness, T) values, NOT the smooth fit curves --
        # hovering shows what was actually simulated, not an interpolation.
        plotted = (T_arr > 0)
        idx_plotted = np.where(plotted)[0]
        def _fmt_main(i_local, _idx=idx_plotted):
            i = _idx[i_local]
            tag = "used in fit" if used[i] else "excluded from fit"
            e = err[i] if i < len(err) and np.isfinite(err[i]) else None
            e_s = f"\n  sigma_T = {e:.4e}" if e is not None else ""
            return (f"x = {t_arr[i]:.4f} mm\n"
                    f"T = {T_arr[i]:.6e}{e_s}\n"
                    f"({tag})")
        _hover_data[ax] = {"x": t_arr[plotted], "y": T_arr[plotted],
                            "fmt": _fmt_main}
        _make_hover_annotation(ax)

    def _draw_resid(a,b,g):
        ax=ax_resid; ax.cla(); ax.set_facecolor("#fafbfc")
        ax.axhline(0,color="#555",lw=0.9,zorder=2)
        ax.axvspan(tail_min,t_arr.max()*1.1,alpha=0.07,color="#cc0000",zorder=0)
        valid2=T_arr>0
        pct = None
        if valid2.any():
            try:
                T_fit=archer_transmission(t_arr[valid2],a,b,g)
                pct=(T_fit-T_arr[valid2])/T_arr[valid2]*100
                idxs = np.where(valid2)[0]
                cols  = [color if used[i] else "#ccc" for i in idxs]
                # Excluded points (gray) are visually de-emphasized -- lower
                # alpha and a hatch pattern -- so they read as "outside the
                # fit" at a glance rather than looking like ordinary bars
                # that just happen to be tall. Without this, an excluded
                # point's residual (which can be much larger than anything
                # inside the fit -- e.g. a sparse deep point at 15-20%+) is
                # easy to mistake for the fit's actual worst residual, even
                # though the printed max_resid_pct below correctly excludes
                # it. The used/excluded boundary is also marked with a
                # vertical line at the deepest FITTED point's thickness.
                alphas = [0.85 if used[i] else 0.30 for i in idxs]
                hatches = [None if used[i] else "//" for i in idxs]
                bars = ax.bar(t_arr[valid2],pct,width=t_arr.max()*0.016,
                               color=cols,zorder=3,
                               edgecolor="white",linewidth=0.4)
                for bar_, a_, h_ in zip(bars, alphas, hatches):
                    bar_.set_alpha(a_)
                    if h_: bar_.set_hatch(h_)

                # Mark the deepest point actually used in the fit -- the
                # boundary max_resid_pct is computed up to, and no further.
                used_idx = np.where(used & valid2)[0]
                if len(used_idx):
                    x_last_fit = t_arr[used_idx].max()
                    ax.axvline(x_last_fit, color="#333", lw=1.1, ls="-",
                               alpha=0.55, zorder=4)
                    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else max(
                        (abs(v) for v in pct), default=10) * 1.15
                    ax.text(x_last_fit, y_top, " max fitted pt",
                            fontsize=6.5, color="#333", alpha=0.7,
                            va="top", ha="left", rotation=90)
            except: pass
        for lv,lc,la in [(5,"#dd8800",0.6),(10,"#cc2222",0.45)]:
            ax.axhline(+lv,ls=":",color=lc,lw=0.9,alpha=la)
            ax.axhline(-lv,ls=":",color=lc,lw=0.9,alpha=la)
            ax.text(0,lv+0.4,f"+{lv}%",fontsize=6.5,color=lc,alpha=la)
        ax.set_ylabel("Residual %",fontsize=8.5)
        ax.set_xlabel("Barrier Thickness (mm)  (gray/hatched = excluded from fit)",fontsize=8.5)
        ax.grid(True,which="major",ls=":",alpha=0.35)
        ax.tick_params(axis="both",labelsize=8)

        # Hover data: each bar's exact thickness + residual% (only if the
        # residual computation above actually succeeded -- pct stays None
        # on the rare archer_transmission() failure, e.g. an out-of-range
        # slider combination, and hover simply has nothing to show then).
        if pct is not None:
            idx_valid2 = np.where(valid2)[0]
            def _fmt_resid(i_local, _idx=idx_valid2, _pct=pct):
                i = _idx[i_local]
                tag = "used in fit" if used[i] else "excluded from fit (not counted in max resid %)"
                return (f"x = {t_arr[i]:.4f} mm\n"
                        f"residual = {_pct[i_local]:+.2f}%\n"
                        f"({tag})")
            _hover_data[ax] = {"x": t_arr[valid2], "y": pct, "fmt": _fmt_resid}
            _make_hover_annotation(ax)
        else:
            _hover_data.pop(ax, None)
            _hover_ann.pop(ax, None)

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
        # Both are computed from t_arr[used]/T_arr[used] above -- fitted
        # points only, matching the residual plot's used-vs-excluded (gray/
        # hatched) styling and "max fitted pt" marker. Label says so
        # explicitly since the residual plot still SHOWS excluded points
        # (as de-emphasized bars) for context, which could otherwise read
        # as inconsistent with these two summary numbers.
        lines += [f"\n  RMSE(log10) = {rmse_s}  (fitted pts only)\n",
                  f"  Max|resid|  = {mxp_s}  (fitted pts only)\n"]

        # GSA Archer-vs-local FVL block -- same layout/columns as the
        # standard Archer FVL table above, but inverted through the GSA
        # (weighted-sum) model instead of the single-term Archer equation,
        # against the SAME local-bracketing ground truth. Only shown when
        # this material was fit with fit_method="gsa" and produced group
        # parameters (gsa_result.gsa_params truthy) -- mirrors the
        # gsa_result-gated overlay already drawn in _draw_main().
        if gsa_result is not None and gsa_result.gsa_params:
            gsa_fvl_rows = compute_fvl_with_local_gsa(
                t_arr, T_arr, gsa_result.gsa_params)
            if gsa_fvl_rows:
                gsa_all_ok = all(r["accepted"] for r in gsa_fvl_rows)
                gsa_hdr = ("ALL ACCEPTED (<=10%)" if gsa_all_ok
                           else "SOME REJECTED")
                xc_line_s = (f"{gsa_result.x_c:.2f}mm"
                             if math.isfinite(gsa_result.x_c) else "inf")
                lines += [f"\n  -- GSA FVL (mm) [{gsa_hdr}] --\n",
                          f"  K={gsa_result.K}  verdict="
                          f"{gsa_result.verdict}\n",
                          f"  breakpoint x_c = {xc_line_s}\n",
                          f"  {'Lyr':<5}  {'GSA':>7}  {'Local':>7}  "
                          f"{'Meth':>4}  {'D%':>5}\n"]
                for r in gsa_fvl_rows:
                    ok_c = "v" if r["accepted"] else "X"
                    xg_s = (f"{r['x_gsa']:>7.2f}"
                            if math.isfinite(r["x_gsa"]) else f"{'n/a':>7}")
                    lines.append(
                        f"  {r['label']:<5}  {xg_s}  "
                        f"{r['x_local']:>7.2f}  {r['method']:>4}  "
                        f"{r['delta_pct']:>4.1f}% {ok_c}\n")
                gsa_rmse_s = (f"{gsa_result.rmse_gsa:.5f}"
                              if gsa_result.rmse_gsa is not None
                              and math.isfinite(gsa_result.rmse_gsa)
                              else "n/a")
                gsa_mxp_s = (f"{gsa_result.maxerr_gsa_pct:.1f}%"
                             if gsa_result.maxerr_gsa_pct is not None
                             and math.isfinite(gsa_result.maxerr_gsa_pct)
                             else "n/a")
                lines += [f"  RMSE(log10) = {gsa_rmse_s}  (GSA, all pts)\n",
                          f"  Max|resid|  = {gsa_mxp_s}  (GSA, all pts)\n"]

        # Piecewise Archer-vs-local FVL block -- same layout/columns as the
        # standard/GSA FVL tables above, inverted via
        # archer_thickness_piecewise() (plain closed-form archer_thickness()
        # on whichever side's triple governs each T_target -- no Newton
        # iteration, unlike GSA). Only shown when this material was fit
        # with fit_method="piecewise" and the cutoff grid search converged.
        if pw_result is not None and math.isfinite(pw_result.x_star):
            pw_fvl_rows = compute_fvl_with_local_piecewise(
                t_arr, T_arr, pw_result)
            if pw_fvl_rows:
                pw_all_ok = all(r["accepted"] for r in pw_fvl_rows)
                pw_hdr = ("ALL ACCEPTED (<=10%)" if pw_all_ok
                          else "SOME REJECTED")
                seam_s = "continuous" if pw_result.seam_continuous else "DISCONTINUOUS"
                pw_daicc_s = (f"  dAICc={pw_result.dAICc:.2f}"
                              if math.isfinite(pw_result.dAICc) else "")
                lines += [f"\n  -- Piecewise FVL (mm) [{pw_hdr}] --\n",
                          f"  verdict={pw_result.verdict}{pw_daicc_s}\n",
                          f"  cutoff x* = {pw_result.x_star:.2f}mm  "
                          f"(seam {seam_s}, n_thin={pw_result.n_thin}, "
                          f"n_thick={pw_result.n_thick})\n",
                          f"  thin : a={pw_result.thin_params[0]:.5f} "
                          f"b={pw_result.thin_params[1]:.5f} "
                          f"g={pw_result.thin_params[2]:.4f}\n",
                          f"  thick: a={pw_result.thick_params[0]:.5f} "
                          f"b={pw_result.thick_params[1]:.5f} "
                          f"g={pw_result.thick_params[2]:.4f}\n",
                          f"  {'Lyr':<5}  {'PW':>7}  {'Local':>7}  "
                          f"{'Meth':>4}  {'D%':>5}\n"]
                for r in pw_fvl_rows:
                    ok_c = "v" if r["accepted"] else "X"
                    xp_s = (f"{r['x_piecewise']:>7.2f}"
                            if math.isfinite(r["x_piecewise"]) else f"{'n/a':>7}")
                    lines.append(
                        f"  {r['label']:<5}  {xp_s}  "
                        f"{r['x_local']:>7.2f}  {r['method']:>4}  "
                        f"{r['delta_pct']:>4.1f}% {ok_c}\n")
                pw_rmse_s = (f"{pw_result.rmse_piecewise:.5f}"
                             if math.isfinite(pw_result.rmse_piecewise)
                             else "n/a")
                pw_mxp_s = (f"{pw_result.maxerr_piecewise_pct:.1f}%"
                            if math.isfinite(pw_result.maxerr_piecewise_pct)
                            else "n/a")
                lines += [f"  RMSE(log10) = {pw_rmse_s}  (piecewise, all pts)\n",
                          f"  Max|resid|  = {pw_mxp_s}  (piecewise, all pts)\n"]

        ax.text(0.03,0.98,"".join(lines),transform=ax.transAxes,
                va="top",ha="left",fontsize=7.0,fontfamily="monospace",
                color="#1a1a1a",linespacing=1.25)
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

    _persisted=[False]  # tracks whether THIS session has already written a
                         # "manual" fit_info for this material, so the
                         # close_event handler below doesn't redundantly
                         # rewrite an unchanged state after an explicit Save.

    def _persist_manual(a2,b2,g2,log_legacy_txt=True):
        output_dir.mkdir(parents=True,exist_ok=True)
        fvl_rows = compute_fvl_with_local(t_arr,T_arr,a2,b2,g2)
        rm2,mp2 = compute_fit_quality(t_arr[used],T_arr[used],a2,b2,g2)

        if log_legacy_txt:
            # Kept for backward compatibility with anything already parsing
            # the old append-only text log -- the fit-info section appended
            # to {nuclide}_{barrier}_transmission_data.csv by write_fit_info()
            # below is now the source of truth (that section gets REPLACED
            # per material on every save, so there's exactly one current
            # block to read instead of having to grep the tail of an
            # ever-growing .txt).
            log = output_dir/f"{nuclide}_{barrier}_manual_params.txt"
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

        # Pass gsa_result/pw_result/thicknesses/transmissions through here
        # too -- without this, clicking "Save Params" (or closing the tuner
        # window, which also calls _persist_manual) would silently WIPE the
        # GSA/PIECEWISE DIAGNOSTIC block that analyze_one()'s earlier "auto"
        # write_fit_info() call had written, since write_fit_info() always
        # strips-and-rewrites the entire fit-info section from scratch. The
        # gsa_result/pw_result objects passed into launch_interactive_tuner()
        # were computed once from the ORIGINAL auto-fit alpha/beta/gamma and
        # don't change as the user drags the a/b/g sliders here, so re-
        # persisting the same objects on manual save is correct -- they're
        # diagnostics about the data's shape, not about whatever a2/b2/g2
        # happen to be at save time.
        write_fit_info(nuclide, barrier, output_dir, a2, b2, g2, fvl_rows,
                       source="manual",
                       sd_alpha=sd_alpha, sd_beta=sd_beta, sd_gamma=sd_gamma,
                       alpha_tail=alpha_tail, r2_tail=r2_tail,
                       rmse_log10=rm2, max_resid_pct=mp2, mu_nist=mu_nist,
                       gsa_result=gsa_result, pw_result=pw_result,
                       thicknesses=t_arr, transmissions=T_arr)
        _persisted[0]=True
        return fvl_rows

    def save_params(_):
        a2,b2,g2=sl_a.val,sl_b.val,sl_g.val
        _persist_manual(a2,b2,g2)
        print(f"\n  Saved => {nuclide}_{barrier}_transmission_data.csv "
              f"(fit-info section) / {nuclide}_{barrier}_manual_params.txt")
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

    def on_close(_evt):
        # Persist whatever the sliders showed at close time, even if the
        # user never clicked "Save Params" -- this is what removes the
        # manual step: closing the tuner (its normal end-of-review action)
        # is now sufficient on its own to save the current HVL/TVL/CVL/MVL
        # and fit parameters. If Save Params was already clicked with the
        # sliders unchanged since, this just rewrites the same values.
        try:
            _persist_manual(sl_a.val, sl_b.val, sl_g.val, log_legacy_txt=False)
            print(f"  (fit info saved on window close: {nuclide}/{barrier})")
        except Exception as e:
            print(f"  ! could not save fit info on close for "
                  f"{nuclide}/{barrier}: {e}")

        # Also refresh the SAME clean, standard "{nuclide}_{barrier}_
        # transmission.png" that Fit All / non-interactive Fit Single
        # produce via plot_static() -- without this, running Fit Single
        # with "Interactive" checked never touched that file at all (only
        # "Save Figure" did, and that saves a screenshot of the whole tuner
        # window -- sliders, buttons, everything -- under a completely
        # different filename, {nuclide}_{barrier}_tuned_{a}_{b}_{g}.png).
        # Regenerating it here means the plain plot on disk always reflects
        # whatever fit was actually saved, regardless of which Fit Single
        # mode (interactive or not) produced it.
        try:
            plot_static(nuclide, barrier, thicknesses, transmissions,
                        sl_a.val, sl_b.val, sl_g.val, output_dir,
                        unc_results=unc_results, fit_mask=fit_mask)
        except Exception as e:
            msg = str(e) or repr(e) or "(no exception message)"
            print(f"  ! could not refresh {nuclide}_{barrier}_transmission.png "
                  f"on close: {type(e).__name__}: {msg}")

    btn_auto.on_clicked(reset_auto); btn_pub.on_clicked(reset_pub)
    btn_save.on_clicked(save_params); btn_snap.on_clicked(save_figure)
    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect('motion_notify_event', _on_hover_move)

    fig.text(0.06,0.152,
             f"Alpha from data tail (last {DEFAULT_ALPHA_TAIL_N} pts, "
             f"R2={r2_tail:.4f})  |  "
             f"a_tail={alpha_tail:.6f} mm^-1  |  "
             f"Red band = +/-{alpha_tol*100:.0f}% tol  |  "
             f"Beta FREE (no NIST anchor)  |  "
             f"ODR weights: sx={0.5} mm, sy=sigma_T",
             fontsize=7.5,color="#440000",style="italic")

    redraw()
    plt.show()  # Display interactive tuner window


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
# ALL-MATERIALS SUMMARY PLOT  (one figure per isotope, every barrier overlaid)
# ═══════════════════════════════════════════════════════════════════════════════
# Matches the style of the user's own reference chart (e.g. "Lu-177": log-scale
# transmission on Y, thickness in mm on X, one colored line per material,
# legend in the corner) -- but built from THIS codebase's actual fitted Archer
# curves (alpha/beta/gamma per material, read back from the fit-info section
# every *_transmission_data.csv already carries) rather than from raw data
# points, so every barrier's line extends smoothly from 0 out to that
# material's own deepest simulated thickness, exactly like the reference plot.

def _discover_materials_for_nuclide(nuclide, output_dir):
    """
    Return sorted list of barrier names for which
    {nuclide}_{barrier}_transmission_data.csv exists in output_dir (any
    barrier, not just the six BARRIER_COLORS keys, so a custom/renamed
    material still gets picked up and just falls back to a generic color).
    """
    output_dir = Path(output_dir)
    out = []
    prefix = f"{nuclide}_"
    suffix = "_transmission_data.csv"
    for p in sorted(output_dir.glob(f"{prefix}*{suffix}")):
        stem = p.stem[:-len("_transmission_data")]
        if not stem.startswith(prefix):
            continue
        barrier = stem[len(prefix):]
        if barrier:
            out.append(barrier)
    return out


def plot_isotope_summary(nuclide, output_dir, barriers=None,
                          show_data_points=False, log_ymin=1e-5):
    """
    Build ONE figure per isotope with every material's fitted Archer curve
    overlaid -- the "all materials for this isotope" chart (see the user's
    reference Lu-177 image: Transmission (log Y) vs Thickness mm (linear X),
    one line per barrier, legend listing each material).

    Data source: reads back each {nuclide}_{barrier}_transmission_data.csv's
    already-saved fit-info block (alpha/beta/gamma, via
    _read_fit_info_block()) rather than re-fitting -- this is deliberately a
    REPORTING function, not a fitting one, so it always reflects whatever the
    most recent analyze_one()/analyze_all()/interactive-tuner save left in
    place (auto or manual, whichever is newer) with no risk of silently
    re-deriving different parameters than what's on record. A material with
    no saved fit-info block yet (write_transmission_csv() ran but
    write_fit_info() hasn't) is skipped with a warning rather than crashing
    the whole isotope's plot.

    Each material's line is drawn from x=0 out to that material's own
    deepest simulated thickness (read from the same CSV's data rows) --
    materials are NOT forced to a common x-range, so a fast-attenuating
    barrier (e.g. Lead) naturally stops far short of a slow one (e.g.
    Gypsum), matching the reference chart's own visual convention.

    Parameters
    ----------
    nuclide : str
    output_dir : Path
    barriers : list[str], optional
        Explicit barrier list/order for the legend. Defaults to
        BARRIER_COLORS' own key order filtered to materials actually present
        for this nuclide, then any other barriers found appended after (so
        the legend order matches every other plot in this file when
        possible, without silently dropping an unexpected material name).
    show_data_points : bool
        Overlay the actual simulated (thickness, T) points as faint markers
        on top of each fitted curve -- off by default to keep the figure as
        clean as the reference image, but useful for a sanity-check pass.
    log_ymin : float
        Lower bound of the log Y axis (default 1e-5, matching plot_static()).

    Returns
    -------
    Path to the saved PNG, or None if no material had a usable fit.
    """
    output_dir = Path(output_dir)
    found = _discover_materials_for_nuclide(nuclide, output_dir)
    if not found:
        print(f"  ! No {nuclide}_*_transmission_data.csv files found in "
              f"{output_dir} -- run analyze_one()/analyze_all() for this "
              f"isotope first.")
        return None

    if barriers is None:
        preferred = [b for b in BARRIER_COLORS if b in found]
        extra     = [b for b in found if b not in BARRIER_COLORS]
        barriers  = preferred + sorted(extra)
    else:
        barriers = [b for b in barriers if b in found]

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted_any = False
    x_max_overall = 0.0

    for barrier in barriers:
        csv_path = output_dir / f"{nuclide}_{barrier}_transmission_data.csv"
        info = _read_fit_info_block(csv_path)
        if info is None or not all(math.isfinite(v) for v in
                                    (info["alpha"], info["beta"], info["gamma"])):
            print(f"  ! {nuclide}/{barrier}: no saved fit-info block yet "
                  f"(run analyze_one()/analyze_all() or save from the "
                  f"interactive tuner first) -- skipped.")
            continue

        a, b, g = info["alpha"], info["beta"], info["gamma"]

        # Deepest simulated thickness for this material, from the CSV's own
        # data rows (not from the fit-info block) -- this is what makes each
        # material's line stop at its own natural extent, matching the
        # reference chart.
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            t_max = 0.0
            for row in reader:
                if row.get("thickness_mm", "").startswith("#") or not row.get("thickness_mm"):
                    break  # hit the blank/fit-info section
                try:
                    t_max = max(t_max, float(row["thickness_mm"]))
                except (ValueError, TypeError):
                    break
        if t_max <= 0:
            print(f"  ! {nuclide}/{barrier}: could not determine a max "
                  f"thickness from its data rows -- skipped.")
            continue

        color = BARRIER_COLORS.get(barrier, None)  # None -> mpl auto-cycles
        t_sm  = np.linspace(0, t_max, 400)
        y_sm  = archer_transmission(t_sm, a, b, g)
        y_sm  = np.where(np.isfinite(y_sm) & (y_sm > 0), y_sm, np.nan)

        line, = ax.semilogy(t_sm, y_sm, "-", color=color, lw=2.0,
                             label=barrier)
        if color is None:
            color = line.get_color()  # capture the auto-assigned color
        plotted_any = True
        x_max_overall = max(x_max_overall, t_max)

        if show_data_points:
            (thicknesses, transmissions, *_rest) = collect_transmission(
                nuclide, barrier, output_dir)
            valid = transmissions > 0
            ax.semilogy(thicknesses[valid], transmissions[valid], "o",
                        color=color, ms=3.5, alpha=0.5, zorder=5)

    if not plotted_any:
        print(f"  ! {nuclide}: no material had a usable saved fit -- no "
              f"summary plot written.")
        plt.close(fig)
        return None

    ekev = NUCLIDE_ENERGY_KEV.get(nuclide)
    title = f"{nuclide}" + (f"  ({ekev:.0f} keV)" if ekev else "")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Thickness (mm)", fontsize=12)
    ax.set_ylabel("Transmission", fontsize=12)
    ax.set_xlim(0, x_max_overall * 1.03)
    ax.set_ylim(log_ymin, 2.0)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls="-", alpha=0.2)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92)
    fig.tight_layout()

    out = output_dir / f"{nuclide}_all_materials_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  All-materials summary plot => {out}")
    return out


def plot_all_isotope_summaries(output_dir, nuclides=None, **kwargs):
    """
    Call plot_isotope_summary() for every isotope that has at least one
    {nuclide}_*_transmission_data.csv present in output_dir. Convenience
    wrapper for --all-style batch use (CLI --summary-plots / GUI "All
    Summary Plots" button) -- does not re-fit anything, purely reads back
    whatever fit-info blocks already exist on disk.
    """
    output_dir = Path(output_dir)
    if nuclides is None:
        seen = set()
        for p in sorted(output_dir.glob("*_transmission_data.csv")):
            stem = p.stem.replace("_transmission_data", "")
            seen.add(stem.split("_")[0])
        nuclides = sorted(seen)

    written = []
    failed = []
    for nuclide in nuclides:
        # One bad isotope's summary plot (backend issue, degenerate/NaN
        # curve from a sparse-tail fit, etc.) must not abort every isotope
        # that sorts after it alphabetically -- previously an uncaught
        # exception here propagated straight out of this loop and silently
        # stopped the entire batch partway through (e.g. everything from
        # "Rb82" onward never got its summary plot, with no indication why).
        try:
            out = plot_isotope_summary(nuclide, output_dir, **kwargs)
            if out is not None:
                written.append(out)
        except Exception as e:
            msg = str(e) or repr(e) or "(no exception message)"
            print(f"  ! Summary plot failed for {nuclide}: "
                  f"{type(e).__name__}: {msg}")
            failed.append((nuclide, f"{type(e).__name__}: {msg}"))
    print(f"\n  {len(written)}/{len(nuclides)} isotope summary plot(s) written.")
    if failed:
        print(f"  {len(failed)} isotope summary plot(s) failed:")
        for nuc, err in failed:
            print(f"      X {nuc}: {err}")
    return written


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
    (t,T,_,_,unc_results,n_estimates,_,_) = collect_transmission(
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
                start_idx=None, end_idx=None,
                alpha_tail_n=DEFAULT_ALPHA_TAIL_N,
                alpha_tol=DEFAULT_ALPHA_TOL,
                thickness_unc_mm=0.5,
                min_nonzero_pct=None,
                fit_method="odr",
                anchor_alpha_global=False,
                fvl_weight=1.0,
                fvl_layer_weights=None):

    (thicknesses,transmissions,doses_barrier,dose_air,
     unc_results,n_estimates,sigma_T,nonzero_pct) = collect_transmission(
        nuclide,barrier,output_dir,target_unc=target_unc)

    (alpha,beta,gamma,
     sd_a,sd_b,sd_g,
     fit_mask,alpha_tail,r2_tail,_) = fit_archer_full(
        thicknesses,transmissions,sigma_T,
        n_points=fit_points,min_T=fit_min_T,
        start_idx=start_idx,end_idx=end_idx,
        alpha_tail_n=alpha_tail_n,alpha_tol=alpha_tol,
        thickness_unc_mm=thickness_unc_mm,
        nuclide=nuclide,barrier=barrier,
        nonzero_pct=nonzero_pct,min_nonzero_pct=min_nonzero_pct,
        fit_method=fit_method,anchor_alpha_global=anchor_alpha_global,
        fvl_weight=fvl_weight,fvl_layer_weights=fvl_layer_weights)

    fvl_rows = compute_fvl_with_local(thicknesses,transmissions,
                                       alpha,beta,gamma)
    print_fvl_table(fvl_rows,nuclide,barrier,alpha,beta,gamma)

    write_transmission_csv(nuclide,barrier,thicknesses,transmissions,
                           doses_barrier,dose_air,alpha,beta,gamma,
                           unc_results,n_estimates,output_dir,fit_mask,
                           nonzero_pct=nonzero_pct)

    # Always persist the auto-fit HVL/TVL/CVL/MVL + fit parameters --
    # no manual "Save Params" click required. If the interactive tuner runs
    # afterward and the user adjusts/saves/closes it, that will overwrite
    # this "auto" entry with a "manual" one for the same (nuclide,barrier).
    rmse_log10, max_resid_pct = compute_fit_quality(
        thicknesses[fit_mask], transmissions[fit_mask], alpha, beta, gamma)
    mu_nist = MU_NARROW_NIST.get((nuclide,barrier))

    # GSA diagnostic (only computed when fit_method="gsa" was requested --
    # see fit_archer_full()'s docstring: the returned alpha/beta/gamma above
    # is ALWAYS the plain single-term Archer fit; this is an add-on report,
    # not a replacement). Uses the full unmasked curve, not fit_mask, so the
    # comparison reflects every real data point collected for this material.
    gsa_result = None
    if fit_method == "gsa":
        gsa_result = fit_gsa_diagnostic(thicknesses, transmissions,
                                         nuclide, barrier)

    # Piecewise (thin/thick) diagnostic -- same pattern as gsa_result above,
    # only computed when fit_method="piecewise" was requested. No nuclear-
    # physics anchoring data needed (unlike GSA), so no GSA_OK-style
    # availability gate -- fit_piecewise_diagnostic() only returns None on
    # an actual fit failure (e.g. too few points), logged via warnings.warn.
    pw_result = None
    if fit_method == "piecewise":
        pw_result = fit_piecewise_diagnostic(thicknesses, transmissions,
                                              nuclide, barrier)

    write_fit_info(nuclide, barrier, output_dir, alpha, beta, gamma,
                   fvl_rows, source="auto",
                   sd_alpha=sd_a, sd_beta=sd_b, sd_gamma=sd_g,
                   alpha_tail=alpha_tail, r2_tail=r2_tail,
                   rmse_log10=rmse_log10, max_resid_pct=max_resid_pct,
                   mu_nist=mu_nist, fit_method=fit_method,
                   gsa_result=gsa_result, pw_result=pw_result,
                   thicknesses=thicknesses, transmissions=transmissions)

    if interactive:
        launch_interactive_tuner(
            nuclide,barrier,thicknesses,transmissions,unc_results,
            alpha,beta,gamma,fit_mask,output_dir,
            alpha_tail,r2_tail,alpha_tol,sd_a,sd_b,sd_g,
            gsa_result=gsa_result, pw_result=pw_result)
    elif make_plot:
        # Plotting is cosmetic -- the fit itself (alpha/beta/gamma, the
        # transmission CSV, and fit_info_summary.csv) is already fully
        # computed and written above. A plotting failure (bad backend,
        # weird all-zero/degenerate tail data, a transient matplotlib
        # error, etc.) must NOT cost the caller a good fit result -- under
        # analyze_all(), an exception raised here used to propagate all the
        # way up and get this whole (nuclide,barrier) pair dropped from
        # archer_parameters_summary.csv even though the real fit succeeded
        # and was already saved to disk. Catch and report instead of
        # letting it kill the pair.
        try:
            plot_static(nuclide,barrier,thicknesses,transmissions,
                        alpha,beta,gamma,output_dir,unc_results,fit_mask)
        except Exception as e:
            msg = str(e) or repr(e) or "(no exception message)"
            print(f"  ! Plot failed for {nuclide}/{barrier} (fit itself "
                  f"succeeded and was saved): {type(e).__name__}: {msg}")

    return alpha,beta,gamma,fvl_rows


_KNOWN_NUCLIDES = ["Lu177","Tc99m","I131","F18","Zr89","I123","Ga68","I124",
                   "Rb82","In111","Cu64","Ac225","Xe133"]
_KNOWN_BARRIERS = ["Lead","LWConcrete","NWConcrete","Steel","Glass","Gypsum"]


def _discover_nuclide_barrier_pairs(output_dir):
    """Scan output_dir for every {nuclide}_{barrier}_<thickness>mm_(dose|edep).mhd
    file actually present and return the set of (nuclide,barrier) pairs found,
    by matching filenames against the known nuclide/barrier label lists.

    This replaces the old fixed double-for-loop over hardcoded nuclide/barrier
    lists, which could silently skip a pair with ZERO log output in two ways:
    (1) a nuclide added to PHOTON_SPECTRA (e.g. Ac225, Xe133) but never added
    to analyze_all's own separate hardcoded list -- the loop would never even
    look for its files, no error, nothing printed; (2) any barrier/nuclide
    label typo'd differently between the sim output and the scan list.
    Scanning the directory directly and matching against the known-label
    lists means a pair is only ever missing from the fit because its files
    genuinely aren't there -- not because this function forgot about it.
    """
    found=set()
    for mhd in list(output_dir.glob("*_edep.mhd"))+list(output_dir.glob("*_dose.mhd")):
        stem=mhd.name
        for suffix in ("_edep.mhd","_dose.mhd"):
            if stem.endswith(suffix):
                stem=stem[:-len(suffix)]; break
        # stem now looks like "{nuclide}_{barrier}_{thickness}mm[_a{angle}]"
        parts=stem.split("_")
        nuclide=parts[0] if parts else None
        if nuclide not in _KNOWN_NUCLIDES or nuclide=="Air" or len(parts)<2:
            continue
        barrier=parts[1]
        if barrier not in _KNOWN_BARRIERS:
            continue
        found.add((nuclide,barrier))
    return found


def analyze_all(output_dir, target_unc=DEFAULT_TARGET_UNC,
                alpha_tail_n=DEFAULT_ALPHA_TAIL_N,
                alpha_tol=DEFAULT_ALPHA_TOL,
                min_nonzero_pct=None,
                fit_method="odr",
                make_summary_plots=True,
                anchor_alpha_global=False,
                fvl_weight=1.0,
                fvl_layer_weights=None):
    # ── THREAD SAFETY: force the non-interactive 'Agg' backend for the
    # duration of this batch run, restoring whatever was active afterward.
    # ------------------------------------------------------------------
    # shieldLabGUI.py's "Fit All" button runs this entire function inside a
    # background thread (threading.Thread), NOT the main GUI thread -- see
    # its _run_archer_all(). Every plot this function triggers below
    # (plot_static() inside analyze_one(), plot_all_isotope_summaries())
    # goes through matplotlib.pyplot, which module-level defaults to the
    # 'TkAgg' backend here (see top of file) -- and launch_interactive_
    # tuner() explicitly calls plt.switch_backend('TkAgg') every time the
    # single-fit interactive tuner runs, which is process-wide global state,
    # not scoped to that call. TkAgg creates and manipulates real Tk
    # widgets/canvases, and Tkinter is NOT thread-safe: doing that from any
    # thread other than the main one is undefined behavior at the Tcl/Tk C
    # level, not something a Python try/except can catch -- it can silently
    # kill or freeze the entire GUI process with no traceback at all. This
    # is the confirmed cause of "Fit All crashes/freezes the GUI with no
    # visible error" reports. The fix: batch/background-thread plotting
    # must never use TkAgg. 'Agg' is the pure-raster, no-GUI-toolkit
    # backend -- explicitly documented as safe to use off the main thread
    # since it never touches a windowing toolkit, only draws into an
    # in-memory image buffer before fig.savefig().
    _prev_backend = matplotlib.get_backend()
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    try:
        rows=[]
        nuclides_seen=set()
        failed=[]   # list of (nuclide, barrier, error_str) -- attempted but raised
        pairs=sorted(_discover_nuclide_barrier_pairs(output_dir))
        if not pairs:
            print(f"  No {{nuclide}}_{{barrier}}_*mm_(dose|edep).mhd files found "
                  f"under {output_dir} matching a known nuclide/barrier label "
                  f"-- nothing to fit.")
        for nuclide,barrier in pairs:
            try:
                a,b,g,fvl_rows = analyze_one(
                    nuclide,barrier,output_dir,interactive=False,
                    target_unc=target_unc,
                    alpha_tail_n=alpha_tail_n,alpha_tol=alpha_tol,
                    min_nonzero_pct=min_nonzero_pct,fit_method=fit_method,
                    anchor_alpha_global=anchor_alpha_global,
                    fvl_weight=fvl_weight,fvl_layer_weights=fvl_layer_weights)
                row={"nuclide":nuclide,"barrier":barrier,
                     "alpha":a,"beta":b,"gamma":g,
                     "all_accepted":all_fvl_accepted(fvl_rows)}
                for r in fvl_rows:
                    row[f"{r['label']}_archer"]=r["x_archer"]
                    row[f"{r['label']}_local"] =r["x_local"]
                    row[f"{r['label']}_delta%"]=r["delta_pct"]
                rows.append(row)
                nuclides_seen.add(nuclide)
            except Exception as e:
                # str(e) can be empty for some exception types (certain
                # matplotlib/backend errors in particular) -- fall back to
                # repr()/the exception type name so a failed pair is never
                # printed as a blank, easy-to-miss "X nuclide/barrier: " line.
                msg = str(e) or repr(e) or "(no exception message)"
                print(f"  X {nuclide}/{barrier}: {type(e).__name__}: {msg}")
                failed.append((nuclide,barrier,f"{type(e).__name__}: {msg}"))
        if rows:
            out=output_dir/"archer_parameters_summary.csv"
            with open(out,"w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            print(f"\nSummary => {out}")

        # Explicit, impossible-to-miss accounting of what happened to every
        # pair that was FOUND in output_dir, so "some materials didn't fit"
        # always has a concrete answer instead of relying on the user to
        # notice a single scrolled-past "X ..." line during the run.
        fitted_pairs=set((r["nuclide"],r["barrier"]) for r in rows)
        print(f"\n  ── Fit-all summary ──────────────────────────────")
        print(f"  Pairs found in {output_dir}: {len(pairs)}")
        print(f"  Fitted successfully:        {len(fitted_pairs)}")
        if failed:
            print(f"  Failed (exception during fit): {len(failed)}")
            for nuc,bar,err in failed:
                print(f"      X {nuc}/{bar}: {err}")
        print(f"  ─────────────────────────────────────────────────")

        # One "all materials for this isotope" overlay plot per nuclide that
        # had at least one material successfully fit above -- reads back the
        # fit-info blocks just written, does not re-fit. plot_all_isotope_
        # summaries() already guards each individual isotope internally, but
        # wrap the call too as a last-resort safety net -- this step is
        # purely cosmetic reporting and must never be able to make
        # analyze_all() itself look like it failed/hung when every fit above
        # already succeeded.
        if make_summary_plots and nuclides_seen:
            print(f"\n  Building all-materials summary plot(s) for "
                  f"{len(nuclides_seen)} isotope(s)...")
            try:
                plot_all_isotope_summaries(output_dir, nuclides=sorted(nuclides_seen))
            except Exception as e:
                msg = str(e) or repr(e) or "(no exception message)"
                print(f"  ! Summary-plot batch failed: {type(e).__name__}: {msg} "
                      f"(all fits above were already saved successfully)")
    finally:
        # ALWAYS restore whatever backend was active before this batch run,
        # regardless of how the function exits (normal return or exception),
        # so a later interactive tuner launch on the main thread still gets
        # TkAgg as expected -- see the thread-safety comment above this
        # function for why Agg was forced in the first place.
        try:
            matplotlib.use(_prev_backend, force=True)
        except Exception:
            pass


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
                   choices=["F18","Tc99m","I131","Lu177","Zr89","Ga68","I124","I123","Rb82","Cu64","In111"])
    p.add_argument("--barrier",default="Lead",
                   choices=["Lead","LWConcrete","NWConcrete",
                             "Steel","Glass","Gypsum"])
    p.add_argument("--output",default="output",
                   help="Directory with .mhd files  (default: output/)")
    p.add_argument("--target-unc",type=float,default=DEFAULT_TARGET_UNC)
    p.add_argument("--fit-points",type=int,default=None)
    p.add_argument("--fit-min-T",type=float,default=None)
    p.add_argument("--fit-start-idx",type=int,default=None,
                   help="Start index for fit range (0-based, inclusive)")
    p.add_argument("--fit-end-idx",type=int,default=None,
                   help="End index for fit range (0-based, inclusive, -1 for last)")
    p.add_argument("--min-nonzero-pct",type=float,default=None,
                   help="Drop any point whose dose .mhd array has fewer than "
                        "this %% of voxels non-zero (same definition as the "
                        "MHD-viewer tab in shieldLabGUI.py). E.g. 90 to "
                        "require >=90%% non-zero voxels.")
    p.add_argument("--fit-method",choices=["odr","standard","fvl_optimized","odr_fvl","gsa"],default="odr",
                   help="'odr' (default): paper-matched two-step fit, alpha "
                        "pinned near the data-tail slope, multi-start ODR. "
                        "'standard': plain weighted nonlinear least squares, "
                        "all 3 parameters free from one generic guess, no "
                        "alpha pinning -- an independent cross-check. "
                        "'fvl_optimized': alpha pinned to the same tail band "
                        "as odr, but beta/gamma are chosen to directly "
                        "minimize Archer-vs-local-bracket FVL (HVL/TVL/etc.) "
                        "disagreement instead of point-transmission residuals. "
                        "'odr_fvl': blend of the two above -- real "
                        "uncertainty-weighted ODR candidates, selected/"
                        "polished by a combined RMSE+FVL score (see "
                        "--fvl-weight). Use this to nudge an 'odr' fit "
                        "toward better FVL agreement without jumping all "
                        "the way to 'fvl_optimized's fully FVL-driven "
                        "answer. "
                        "'gsa': runs the SAME alpha-pinned fit as 'odr' for "
                        "the returned/plotted curve, PLUS a physics-anchored "
                        "Grouped Spectral Archer (multi-energy-group) "
                        "comparison as a diagnostic, printed and saved to "
                        "the transmission CSV's GSA DIAGNOSTIC block -- "
                        "requires gsa_fit.py/nist_xcom_data.py to be "
                        "deployed alongside this file.")
    p.add_argument("--fvl-weight",type=float,default=1.0,
                   help="Only used by --fit-method odr_fvl. Relative weight "
                        "of the (normalized) archer-vs-local FVL-agreement "
                        "term vs. the (normalized) point-residual RMSE term "
                        "when selecting/polishing the ODR candidate "
                        "(default 1.0 = equal weight after normalization). "
                        "0.0 reduces odr_fvl to plain 'odr'; try small "
                        "values (e.g. 0.1-0.5) first and increase gradually "
                        "to see how much FVL-agreement improvement costs in "
                        "RMSE, rather than jumping straight to a large "
                        "weight.")
    p.add_argument("--fvl-layer-weights",type=str,default=None,
                   help="Only used by --fit-method fvl_optimized/odr_fvl. "
                        "Per-layer weight within the FVL-agreement term, "
                        "as comma-separated LABEL=weight pairs, e.g. "
                        "'HVL=1' to optimize purely for HVL agreement, or "
                        "'HVL=2,CVL=1' to weight HVL twice as heavily as "
                        "CVL while ignoring QVL/TVL/MVL entirely. A layer "
                        "omitted from this string is excluded from the "
                        "objective (weight 0). Default (unset): every "
                        "resolvable layer (HVL/QVL/TVL/CVL/MVL) weighted "
                        "equally.")
    p.add_argument("--alpha-tail-n",type=int,default=DEFAULT_ALPHA_TAIL_N,
                   help=f"Tail points for alpha OLS (default {DEFAULT_ALPHA_TAIL_N})")
    p.add_argument("--anchor-alpha-global",action="store_true",
                   help="Anchor alpha to the deepest --alpha-tail-n points of "
                        "the FULL simulated dataset for this pair, regardless "
                        "of --fit-min-T/--fit-points/--fit-start-idx/"
                        "--fit-end-idx/--min-nonzero-pct. Without this flag "
                        "(default), alpha is anchored to the deepest points "
                        "WITHIN whatever fit range/threshold is selected -- "
                        "so e.g. --fit-min-T 0.01 or --min-nonzero-pct 90 can "
                        "silently anchor alpha to a shallower tail than the "
                        "deepest points actually measured. Only affects "
                        "fit-method odr/fvl_optimized/gsa (standard doesn't "
                        "pin alpha to the tail at all).")
    p.add_argument("--alpha-tol",type=float,default=DEFAULT_ALPHA_TOL,
                   help=f"Alpha tolerance band (default {DEFAULT_ALPHA_TOL})")
    p.add_argument("--thickness-unc",type=float,default=0.5,
                   help="ODR thickness uncertainty sigma_x in mm (default 0.5)")
    p.add_argument("--estimate-n",action="store_true")
    p.add_argument("--all",action="store_true")
    p.add_argument("--example",action="store_true")
    p.add_argument("--no-interactive",action="store_true")
    p.add_argument("--no-plot",action="store_true")
    p.add_argument("--no-summary-plots",action="store_true",
                   help="With --all, skip auto-generating the per-isotope "
                        "all-materials overlay plot after fitting.")
    p.add_argument("--summary-plots",action="store_true",
                   help="Generate the all-materials-for-this-isotope overlay "
                        "plot(s) from EXISTING saved fit-info (no re-fit). "
                        "With --nuclide, builds just that isotope's plot; "
                        "otherwise builds one for every isotope with saved "
                        "fit-info present in --output.")
    return p.parse_args()


def main():
    args=parse_args(); output_dir=Path(args.output)
    fvl_layer_weights = parse_fvl_layer_weights(args.fvl_layer_weights)
    if args.example:      lu177_room_example(); return
    if args.estimate_n:   run_estimate_n(args.nuclide,args.barrier,
                                         output_dir,args.target_unc); return
    if args.summary_plots:
        nuclide_explicit = "--nuclide" in sys.argv
        if nuclide_explicit:
            plot_isotope_summary(args.nuclide,output_dir)
        else:
            plot_all_isotope_summaries(output_dir)
        return
    if args.all:
        analyze_all(output_dir,args.target_unc,
                    args.alpha_tail_n,args.alpha_tol,
                    min_nonzero_pct=args.min_nonzero_pct,
                    fit_method=args.fit_method,
                    make_summary_plots=not args.no_summary_plots,
                    anchor_alpha_global=args.anchor_alpha_global,
                    fvl_weight=args.fvl_weight,
                    fvl_layer_weights=fvl_layer_weights); return
    analyze_one(args.nuclide,args.barrier,output_dir,
                interactive=not args.no_interactive,
                make_plot=not args.no_plot,
                target_unc=args.target_unc,
                fit_points=args.fit_points,
                fit_min_T=args.fit_min_T,
                start_idx=args.fit_start_idx,
                end_idx=args.fit_end_idx,
                alpha_tail_n=args.alpha_tail_n,
                alpha_tol=args.alpha_tol,
                thickness_unc_mm=args.thickness_unc,
                min_nonzero_pct=args.min_nonzero_pct,
                fit_method=args.fit_method,
                anchor_alpha_global=args.anchor_alpha_global,
                fvl_weight=args.fvl_weight,
                fvl_layer_weights=fvl_layer_weights)

if __name__=="__main__":
    main()