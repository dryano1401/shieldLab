"""
gateTurbo.py  —  GATE 10 (opengate) shielding simulation  [oblique + kV edition]
══════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import opengate as gate
from opengate import g4_units

m   = g4_units.m
cm  = g4_units.cm
mm  = g4_units.mm
MeV = g4_units.MeV
keV = g4_units.keV
deg = g4_units.deg

N_PRIMARIES      = 2_000_000_000
N_PRIMARIES_TEST =    10_000_000
OUTPUT_DIR = Path("output")

# ══════════════════════════════════════════════════════════════════════════════
# Oumano et al. 2025 (JACMP 26:e70084) replication defaults
# ------------------------------------------------------------------------------
# These are the ZERO-FLAG defaults: `python shieldLabSim.py --barrier X
# --thickness Y` with nothing else specified reproduces the paper's Sec 2.1
# geometry and source. Override any of them explicitly when you need something
# else (a narrow-beam validation run, a legacy comparison, etc).
#
# DEFAULT_CONE_HALF_ANGLE_DEG = 90 deg: with cone_source=True this sets
#   theta=[180-90, 180] deg = [90,180] deg. GATE measures theta from -z, so
#   this is the full 2-pi hemisphere aimed at the barrier -- mathematically
#   identical to --no-cone, and exactly the source Oumano Sec 2.1 describes
#   ("constrained to emit in the 2-pi solid angle... oriented toward the
#   barrier"). A previous version of this constant auto-sized a narrow cone
#   just wide enough to cover the (much smaller) old detector footprint; that
#   geometry-fitting logic no longer applies now that the detector spans the
#   full tissue-block face (see below), so the constant is just 90 deg.
#
# DEFAULT_DETECTOR_DEPTH_MM = 10: Oumano scores "a depth of 1 cm into the
#   tissue block" by averaging the 2nd and 3rd 5 mm Z-planes (5-15 mm depth).
#   Paired with DEFAULT_DETECTOR_Z_MM=20 (4 planes), a slab centred at 10 mm
#   spans 0-20 mm, so planes[1:3] land exactly on 5-15 mm depth.
#
# DEFAULT_DETECTOR_XY_MM = 250: Oumano Sec 2.1's Dose Actor is attached to the
#   FULL 2 m x 2 m tissue-block face, and a previous version of this constant
#   set 2000 (800x800 voxels) to match that literally. In practice, running
#   that full footprint measurably LOWERED T relative to the same depth/plane
#   settings at a compact 250 mm footprint (e.g. observed 0.452->0.399 for NW
#   concrete at 78.6 mm, 0.454->0.397 for LW at 113 mm, 0.526->0.489 for lead
#   at 4.75 mm -- a bigger drop where buildup/scatter is larger, consistent
#   with the ROI's statistics converging on a different, spurious timeline
#   when --unc-goal's per-voxel early-stop check is evaluated over a mostly-
#   empty 2.56M-voxel array instead of a densely-hit 40K-voxel one). Since the
#   150 mm ROI mask is always centred on the array's own geometric centre
#   (_build_roi_mask in shieldLabAnalyze.py) and that centre sits on the beam
#   axis regardless of total footprint, the physically scored voxels should be
#   identical either way -- so this is treated as a live artifact of the large
#   array, not a feature worth keeping as the default. 250 mm is comfortably
#   larger than the 150 mm ROI (50 mm margin on each side) and is the
#   footprint that reproduced Oumano's Table 3 to within cross-validated
#   tolerances (LW/NW density-scaling identity to 0.5%, independent
#   narrow-beam check to 2.5%). Pass --detector-size-x/y 2000 explicitly if
#   you specifically want the literal full-block footprint despite this.
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONE_HALF_ANGLE_DEG = 90.0
DEFAULT_DETECTOR_DEPTH_MM   = 10.0
DEFAULT_DETECTOR_XY_MM      = 250.0
DEFAULT_DETECTOR_Z_MM       = 20.0

# ══════════════════════════════════════════════════════════════════════════════
# EM physics list — adjustable for cross-checking against Oumano's published
# values. build_simulation() previously hardcoded this (option4 for --source-
# type xray, option3 for everything else); it's now overridable via
# --physics-list / physics_list= so the buildup/scatter physics can be swapped
# without editing code, e.g. to test whether option4's more refined low-energy
# Compton/Rayleigh/atomic-relaxation treatment narrows the ~4-9% gap observed
# between this code (GATE 10 / opengate) and the paper's published Table 3
# broad-beam values at the F-18 HVL points (Lead/NW/LW concrete). "auto" (the
# default) preserves the original source-type-based selection.
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_PHYSICS_LIST_NUCLIDE = "G4EmStandardPhysics_option3"
DEFAULT_PHYSICS_LIST_XRAY    = "G4EmStandardPhysics_option4"
PHYSICS_LIST_CHOICES = [
    "auto",
    "G4EmStandardPhysics_option1",
    "G4EmStandardPhysics_option2",
    "G4EmStandardPhysics_option3",
    "G4EmStandardPhysics_option4",
    "G4EmLivermorePhysics",
    "G4EmPenelopePhysics",
]

_BARRIER_MFP_MM = {"Lead":5.0,"Steel":15.0,"NWConcrete":80.0,"LWConcrete":120.0,"Glass":60.0,"Gypsum":100.0,"Air":1e9}
DEFAULT_UNC_GOAL = 0.02
CVL_THRESHOLD    = 0.01
# ══════════════════════════════════════════════════════════════════════════════
# RAM_PER_JOB_GB: how much memory --auto assumes each worker SUBPROCESS needs,
# used to cap worker/job count so --auto doesn't oversubscribe RAM. Each worker
# is a brand-new `python shieldLabSim.py ...` process (see _spawn()) that
# re-imports numpy/opengate/scipy/pandas from scratch -- on Windows there is no
# fork()-based copy-on-write sharing of already-loaded libraries the way there
# is on Linux, so every worker pays the full DLL-load cost independently.
# 0.6 GB/worker (the original estimate) was far too optimistic: a real run
# with 17.7 GB available RAM sized 23 concurrent workers under that budget,
# and Windows failed with "The paging file is too small for this operation to
# complete" while the workers' scipy/pandas imports were still loading --
# i.e. the OS ran out of page-file-backed virtual memory before any workers
# even reached the simulation itself. Raised to 2.0 GB/worker, and the
# available-RAM fraction used for sizing is reduced on Windows specifically
# (see _safe_max_jobs/_auto_config) to leave headroom for exactly this.
# ══════════════════════════════════════════════════════════════════════════════
RAM_PER_JOB_GB   = 2.0
_ON_WINDOWS      = sys.platform.startswith("win")
_RAM_FRACTION    = 0.55 if _ON_WINDOWS else 0.80
ANGLE_SWEEP_DEG: list[float] = [0.0, 15.0, 30.0, 45.0, 60.0]

_NIST_AL_ENERGY_KEV = [5.0,8.0,10.0,15.0,20.0,30.0,40.0,50.0,60.0,80.0,100.0,120.0,150.0,200.0]
_NIST_AL_MU_RHO     = [74.24,18.97,26.26,5.848,3.441,1.128,0.5757,0.3681,0.2773,0.2018,0.1704,0.1524,0.1378,0.1228]
_AL_DENSITY_G_CM3   = 2.699
_NIST_CU_ENERGY_KEV = [5.0,8.0,9.0,10.0,15.0,20.0,30.0,40.0,50.0,60.0,80.0,100.0,120.0,150.0,200.0]
_NIST_CU_MU_RHO     = [295.7,73.25,216.4,155.2,64.14,37.44,8.513,3.170,1.553,0.8490,0.4247,0.2685,0.1935,0.1377,0.1065]
_CU_DENSITY_G_CM3   = 8.960
_W_CHAR_LINES = [(57.98,0.52),(59.32,1.00),(66.95,0.14),(67.24,0.27),(69.10,0.09)]
_W_K_EDGE_KEV = 69.525
_W_Z = 74

def _loglog_interp(E_query, E_table, mu_table):
    log_E=np.log(np.array(E_table,dtype=float)); log_mu=np.log(np.array(mu_table,dtype=float))
    return np.exp(np.interp(np.log(np.clip(E_query,E_table[0],E_table[-1])),log_E,log_mu))

def _filter_transmission(E_keV, material, thickness_mm):
    if thickness_mm<=0.0: return np.ones_like(E_keV,dtype=float)
    if material=="Al": mu_rho=_loglog_interp(E_keV,_NIST_AL_ENERGY_KEV,_NIST_AL_MU_RHO); density=_AL_DENSITY_G_CM3
    elif material=="Cu": mu_rho=_loglog_interp(E_keV,_NIST_CU_ENERGY_KEV,_NIST_CU_MU_RHO); density=_CU_DENSITY_G_CM3
    else: raise ValueError(f"Unknown filter material: {material!r}")
    return np.exp(-mu_rho*density*thickness_mm/10.0)

def generate_kv_spectrum(kVp,al_filter_mm=2.5,cu_filter_mm=0.0,n_bins=128):
    E_max=float(kVp); E_keV=np.linspace(1.0,E_max-0.5,n_bins)
    I=_W_Z*np.maximum(E_max-E_keV,0.0)*_filter_transmission(E_keV,"Al",al_filter_mm)*_filter_transmission(E_keV,"Cu",cu_filter_mm)
    spectrum=[(float(E)/1000.0,float(w)) for E,w in zip(E_keV,I) if w>0.0]
    if kVp>_W_K_EDGE_KEV:
        U=kVp/_W_K_EDGE_KEV; overv=((U-1.0)**1.63)*math.log(U)
        brem_Ka1=_W_Z*max(E_max-59.32,0.0); I_Ka1=0.25*brem_Ka1*overv
        for E_l,ri in _W_CHAR_LINES:
            if E_l>_W_K_EDGE_KEV and kVp<E_l+5.0: continue
            T_l=float(_filter_transmission(np.array([E_l]),"Al",al_filter_mm)[0]*_filter_transmission(np.array([E_l]),"Cu",cu_filter_mm)[0])
            Il=I_Ka1*ri*T_l
            if Il>0.0: spectrum.append((E_l/1000.0,Il))
    total=sum(w for _,w in spectrum)
    if total<=0.0: raise ValueError("kV spectrum empty")
    return [(E,w/total) for E,w in spectrum]

def _kv_spectrum_summary(spectrum,kVp,al_filter_mm,cu_filter_mm):
    E_arr=np.array([e for e,_ in spectrum]); w_arr=np.array([w for _,w in spectrum])
    E_mean=np.sum(E_arr*w_arr)*1000.0
    depths=np.linspace(0,30.0,3000)
    T_t=np.array([float(np.sum(w_arr*_filter_transmission(E_arr*1000.0,"Al",d))) for d in depths])
    hvl=float(np.interp(0.5,T_t[::-1],depths[::-1]))
    cu_str=f"  Cu={cu_filter_mm:.1f} mm" if cu_filter_mm>0 else ""
    return f"kVp={kVp:.0f}  Al={al_filter_mm:.1f} mm{cu_str}  E_mean={E_mean:.1f} keV  HVL(Al)≈{hvl:.1f} mm  bins={len(spectrum)}"

def plot_xray_spectrum(kVp,al_filter_mm,cu_filter_mm,output_dir,n_bins=128,save=True):
    import matplotlib.pyplot as plt; import matplotlib.ticker as ticker
    E_max=float(kVp); E_keV=np.linspace(1.0,E_max-0.5,n_bins)
    phi_raw=_W_Z*np.maximum(E_max-E_keV,0.0).astype(float); phi_raw_n=phi_raw/max(phi_raw.max(),1e-30)
    phi_al=phi_raw*_filter_transmission(E_keV,"Al",al_filter_mm); phi_al_n=phi_al/max(phi_al.max(),1e-30)
    phi_f=phi_al*_filter_transmission(E_keV,"Cu",cu_filter_mm); phi_f_n=phi_f/max(phi_f.max(),1e-30)
    char_lines=[]
    if kVp>_W_K_EDGE_KEV:
        U=kVp/_W_K_EDGE_KEV; overv=((U-1.0)**1.63)*math.log(U); brem=_W_Z*max(E_max-59.32,0.0); I_Ka1=0.25*brem*overv
        for E_l,ri in _W_CHAR_LINES:
            if E_l>_W_K_EDGE_KEV and kVp<E_l+5.0: continue
            T_l=float(_filter_transmission(np.array([E_l]),"Al",al_filter_mm)[0]*_filter_transmission(np.array([E_l]),"Cu",cu_filter_mm)[0])
            Il=I_Ka1*ri*T_l
            if Il>0.0: char_lines.append((E_l,Il/max(phi_f.max(),1e-30)))
    full=generate_kv_spectrum(kVp,al_filter_mm,cu_filter_mm,n_bins)
    E_a=np.array([e*1000.0 for e,_ in full]); w_a=np.array([w for _,w in full])
    mean_E=float(np.sum(E_a*w_a)/np.sum(w_a))
    depths=np.linspace(0,30.0,3000); T_h=np.array([float(np.sum(w_a*_filter_transmission(E_a,"Al",d))) for d in depths])
    hvl=float(np.interp(0.5,T_h[::-1],depths[::-1]))
    fig,ax=plt.subplots(figsize=(10,6))
    ax.fill_between(E_keV,phi_raw_n,alpha=0.12,color='#555555'); ax.plot(E_keV,phi_raw_n,color='#888888',lw=1.0,ls='--',label='Unfiltered Kramers')
    if cu_filter_mm>0: ax.plot(E_keV,phi_al_n,color='#E07B39',lw=1.4,ls='-.',alpha=0.8,label=f'After Al {al_filter_mm:.1f} mm')
    fd=f"Al {al_filter_mm:.1f} mm"+(f" + Cu {cu_filter_mm:.1f} mm" if cu_filter_mm>0 else "")
    ax.fill_between(E_keV,phi_f_n,alpha=0.30,color='#2176AE'); ax.plot(E_keV,phi_f_n,color='#2176AE',lw=2.2,label=f'Filtered ({fd})')
    cn=['Kα₂','Kα₁','Kβ₃','Kβ₁','Kβ₂']
    for i,(ec,hc) in enumerate(char_lines):
        ax.annotate('',xy=(ec,hc),xytext=(ec,0),arrowprops=dict(arrowstyle='-',color='#C1121F',lw=2.5))
        ax.text(ec+0.8,hc+0.02,cn[i] if i<len(cn) else f'K{i}',color='#C1121F',fontsize=8,va='bottom')
    if char_lines: ax.axvline(x=-999,color='#C1121F',lw=2.5,label='W char lines')
    ax.axvline(mean_E,color='#2D6A4F',lw=1.5,ls=':',label=f'Mean E = {mean_E:.1f} keV')
    ax.set_xlabel("Photon Energy (keV)"); ax.set_ylabel("Relative Fluence"); ax.set_title(f"X-ray Spectrum — {kVp:.0f} kVp / {fd}")
    ax.set_xlim(0,kVp*1.06); ax.set_ylim(0,1.20); ax.legend(loc='upper left',fontsize=9); ax.grid(True,alpha=0.25)
    if save:
        output_dir.mkdir(parents=True,exist_ok=True)
        cu_tag=f"_Cu{cu_filter_mm:.1f}mm" if cu_filter_mm>0 else ""
        fig.savefig(output_dir/f"spectrum_{kVp:.0f}kVp_Al{al_filter_mm:.1f}mm{cu_tag}.png",dpi=150,bbox_inches='tight')
    plt.show()

def _make_source_label(source_type,nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0):
    if source_type=="nuclide": return nuclide
    cu_str=f"_Cu{cu_filter_mm:.1f}mm" if cu_filter_mm>0 else ""
    return f"xray{kvp:.0f}kVp_Al{al_filter_mm:.1f}mm{cu_str}"

def _make_stem(source_label,barrier_name,thickness_mm,angle_deg=0.0):
    base=f"{source_label}_{barrier_name}_{thickness_mm:.3f}mm"
    if abs(angle_deg)>0.01: base+=f"_a{angle_deg:.0f}"
    return base

def _rotation_matrix_x(angle_deg):
    t=math.radians(angle_deg); c,s=math.cos(t),math.sin(t)
    return [[1,0,0],[0,c,-s],[0,s,c]]

CUSTOM_MATERIALS = {
    "LWConcrete":{"density":1.60,"elements":{"H":0.010,"C":0.001,"O":0.529107,"Na":0.016,"Mg":0.002,"Al":0.033872,"Si":0.337021,"K":0.013,"Ca":0.044,"Fe":0.014}},
    "NWConcrete":{"density":2.30,"elements":{"H":0.010,"C":0.001,"O":0.529107,"Na":0.016,"Mg":0.002,"Al":0.033872,"Si":0.337021,"K":0.013,"Ca":0.044,"Fe":0.014}},
    "GlassNM":{"density":2.50,"elements":{"Na":0.1020,"Ca":0.0510,"Si":0.2480,"O":0.5990}},
    # Density corrected 2026-07-30: Oumano et al. 2025 Table 1 lists 2.33 g/cm3
    # for Gypsum, but this is an error in the published paper -- confirmed
    # directly with the paper's authors, who stated the value actually used in
    # their own simulations was 0.9 g/cm3 (typical of gypsum wallboard, not
    # solid/cast gypsum). This single value fully explained the ~2.7-3.8x
    # excess Gypsum attenuation vs. Oumano's published HVL/TVL/CVL/MVL seen in
    # shieldLabSim runs (2.33/0.9 = 2.589, in the same range as the measured
    # excess-attenuation ratio) -- material registration, composition mass
    # fractions, and EM physics list were all independently ruled out first.
    # All Gypsum sweeps run before this fix (any run predating 2026-07-30) used
    # the wrong 2.33 density and must be re-run to be comparable to Oumano.
    "Gypsum":{"density":0.9,"elements":{"H":0.0234,"O":0.55757,"S":0.186218,"Ca":0.23279}},
    "A514Steel":{"density":7.85,"elements":{"Fe":0.97000,"Mn":0.00950,"Cr":0.00650,"Si":0.00600,"Mo":0.00230,"C":0.004675,"Zr":0.00100,"B":0.000025}},
    # Oumano et al. 2025 (JACMP 26:e70084) Table 1 "muscle" tissue-phantom
    # composition/density -- digit-for-digit from the paper, same pattern as the
    # concrete barriers above. NOT the same as Geant4's stock
    # G4_MUSCLE_SKELETAL_ICRP (a different, NIST-derived composition) -- this is
    # the paper's own exact numbers, for use via --phantom-material OumanoMuscle
    # to close out the "is the phantom material itself close enough to matter"
    # question raised in the shieldLabSim-vs-Oumano investigation (mass fractions
    # sum to exactly 1.000: 0.102+0.143+0.034+0.71+0.001+0.002+0.003+0.001+0.004).
    "OumanoMuscle":{"density":1.05,"elements":{"H":0.102,"C":0.143,"N":0.034,"O":0.71,"Na":0.001,"P":0.002,"S":0.003,"Cl":0.001,"K":0.004}},
}
BARRIER_MATERIAL_MAP = {"Lead":"G4_Pb","LWConcrete":"LWConcrete","NWConcrete":"NWConcrete","Glass":"GlassNM","Gypsum":"Gypsum","Steel":"A514Steel","Air":"G4_AIR"}

PHOTON_SPECTRA = {
    "Lu177":[(0.05461,0.0157),(0.05579,0.0271),(0.06298,0.00304),(0.06324,0.00587),
          (0.06494,0.00200),(0.07164,0.00164),(0.11295,0.0623),(0.13672,0.00047),
          (0.20837,0.1041),(0.24967,0.00200),(0.32132,0.00219)],
    "Tc99m":[(0.018251,0.0215),(0.018367,0.0409),(0.020599,0.00331),(0.020619,0.0064),
          (0.021005,0.00145),(0.140511,0.8900),(0.142630,0.00022)],
    "I131":[(0.02946,0.01530),(0.02978,0.02820),(0.03356,0.00263),(0.03362,0.00509),(0.03442,0.00154),
          (0.08019,0.02620),(0.16393,0.00021),(0.17721,0.00269),(0.27250,0.00058),(0.28431,0.06120),
          (0.31809,0.00077),(0.32465,0.00021),(0.32579,0.00273),(0.35840,0.00016),(0.36449,0.81500),
          (0.40481,0.00055),(0.50300,0.00359),(0.63699,0.07160),(0.64272,0.00217),(0.72291,0.01770)],
    "F18":[(0.51100,1.93500)],
    "Zr89":[(0.016726,0.02070),(0.016738,0.04020),(0.017013,0.00770),(0.511000,0.45500),
            (0.909150,0.99040),(1.620800,0.00073),(1.657300,0.00106),(1.713000,0.00745),(1.744500,0.00123)],
    "Cu64":[(0.51100,0.3514),(1.34577,0.00473)],"Ga68":[(0.51100,1.7800),(1.07734,0.03220),(1.88316,0.00137)],
    "In111":[(0.02298,0.2410),(0.02317,0.4530),(0.02606,0.0392),(0.02610,0.0755),(0.02664,0.0194),(0.17128,0.9061),(0.24535,0.9408)],
    "I123":[(0.02720,0.24700),(0.02747,0.45600),(0.03094,0.04210),(0.03100,0.08110),(0.03170,0.02340),
          (0.15900,0.83600),(0.18262,0.00013),(0.19218,0.00018),(0.24797,0.00069),(0.28103,0.00072),
          (0.33070,0.00012),(0.34636,0.00120),(0.44002,0.00388),(0.50533,0.00288),(0.52897,0.01270),
          (0.53854,0.00310),(0.62458,0.00078),(0.68794,0.00027),(0.73587,0.00047),(0.78360,0.00053)],
    "I124":[(0.02720,0.16600),(0.02747,0.30600),(0.03094,0.02820),(0.03100,0.05440),(0.03170,0.01570),
          (0.30734,0.00021),(0.33567,0.00018),(0.35147,0.00023),(0.40280,0.00014),(0.44388,0.00038),
          (0.51100,0.45000),(0.51780,0.00024),(0.52545,0.00033),(0.54119,0.00214),(0.59234,0.00114),
          (0.60273,0.62900),(0.60992,0.00154),(0.64585,0.00996),(0.66210,0.00056),(0.70746,0.00092),
          (0.70936,0.00046),(0.71375,0.00078),(0.72278,0.10360),(0.74319,0.00013),(0.74320,0.00017),
          (0.77610,0.00012),(0.79076,0.00026),(0.79563,0.00037),(0.87697,0.00023),(0.89943,0.00022),
          (0.96184,0.00017),(0.96819,0.00444),(0.97635,0.00104),(0.98440,0.00014),(1.04511,0.00438),
          (1.05454,0.00125),(1.08640,0.00015),(1.12858,0.00046),(1.20544,0.00022),(1.31567,0.00028),
          (1.32552,0.01578),(1.35520,0.00037),(1.36818,0.00299),(1.37609,0.01790),(1.43664,0.00077),
          (1.44517,0.00039),(1.48892,0.00211),(1.50936,0.03250),(1.56053,0.00167),(1.62222,0.00051),
          (1.63743,0.00209),(1.67560,0.00113),(1.69096,0.11150),(1.72021,0.00183),(1.75251,0.00054),
          (1.85137,0.00216),(1.91856,0.00176),(2.03843,0.00359),(2.07867,0.00360),(2.09094,0.00623),
          (2.09881,0.00154),(2.14421,0.00106),(2.23203,0.00555),(2.28306,0.00530),(2.29440,0.00011),
          (2.38510,0.00013),(2.45390,0.00069),(2.68150,0.00031),(2.74690,0.00478)],
    "Rb82":[(0.511000,1.90700),(0.696850,0.00026),(0.698361,0.00143),(0.711090,0.00054),
            (0.776511,0.15100),(1.180209,0.00017),(1.395260,0.00570),(1.474895,0.00095),
            (1.703540,0.00054),(1.879610,0.00010),(2.168060,0.00040),(2.410650,0.00025),
            (2.480230,0.00036)],
    "Ac225":[(0.036700,0.00015),(0.062900,0.00430),(0.064300,0.00041),(0.071400,0.00013),(0.073500,0.00025),
              (0.073900,0.00264),(0.074600,0.00022),(0.078800,0.00011),(0.083231,0.00750),(0.086105,0.01230),
              (0.087400,0.00226),(0.094900,0.00084),(0.096700,0.00028),(0.096815,0.00150),(0.097474,0.00290),
              (0.099600,0.00700),(0.099800,0.01000),(0.100214,0.00108),(0.100800,0.00075),(0.108400,0.00216),
              (0.111500,0.00264),(0.119900,0.00066),(0.123800,0.00072),(0.124800,0.00024),(0.133600,0.00017),
              (0.134900,0.00027),(0.145200,0.00126),(0.150100,0.00600),(0.152600,0.00019),(0.153900,0.00182),
              (0.157300,0.00320),(0.169900,0.00012),(0.170700,0.00017),(0.178300,0.00014),(0.186100,0.00011),
              (0.188000,0.00450),(0.195800,0.00123),(0.197400,0.00023),(0.197900,0.00033),(0.198400,0.00017),
              (0.216900,0.00271),(0.224700,0.00098),(0.249600,0.00012),(0.253500,0.00116),(0.279300,0.00025),
              (0.452400,0.00089),(0.481100,0.00029),(0.515300,0.00019),(0.517900,0.00015),(0.526100,0.00033)],
    "Xe133":[(0.030625,0.13600),(0.030973,0.25000),(0.034920,0.02360),(0.034987,0.04560),
             (0.035818,0.01410),(0.079614,0.00440),(0.080998,0.36900),(0.160612,0.00107)],
}

THICKNESS_SWEEPS = {
    ("Lu177","Lead"):[0.5,1,1.5,2,3,4,5,6,7.5,9,11,14],("Lu177","NWConcrete"):[25,50,75,100,125,150,175,200,250,300,370],
    ("Lu177","LWConcrete"):[40,80,120,160,200,250,300,360,420,500],("Lu177","Steel"):[8,16,24,32,42,54,68,82,100],
    ("Lu177","Glass"):[35,70,105,140,180,220,270,330],("Lu177","Gypsum"):[80,160,240,320,420,530,660,850],
    ("Tc99m","Lead"):[0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5],("Tc99m","NWConcrete"):[25,50,75,100,125,150,175,210,260,330],
    ("Tc99m","LWConcrete"):[35,70,105,140,180,230,290,360,440],("Tc99m","Steel"):[5,10,15,20,26,34,42,52,62],
    ("Tc99m","Glass"):[30,60,90,120,155,195,245,300],("Tc99m","Gypsum"):[80,160,240,320,420,540,680,850],
    ("I131","Lead"):[4,8,12,16,21,27,34,42,52,64,78],("I131","NWConcrete"):[30,60,90,120,155,195,240,295,365,460],
    ("I131","LWConcrete"):[45,90,135,180,230,295,370,460,580],("I131","Steel"):[10,20,30,40,52,66,84,105,130],
    ("F18","Lead"):[0.5,1,2,4,8,12,16,20],("F18","NWConcrete"):[50,100,150,200,250,305,370,450,555],
    ("F18","LWConcrete"):[65,130,200,270,350,440,550,690,800],("F18","Steel"):[15,30,45,60,75,95,115,140,170],
    ("Zr89","Lead"):[5,10,15,20,25,31,38,46,55,66,80],("Zr89","NWConcrete"):[50,100,150,200,250,305,370,450,555],
    ("Zr89","LWConcrete"):[70,140,215,290,375,470,590,740,860],("Zr89","Steel"):[15,30,45,60,80,100,125,155,190],
    ("Cu64","Lead"):[5,10,15,20,25,31,38,46,55,66],("Cu64","NWConcrete"):[45,90,135,180,225,275,335,415,520],
    ("Cu64","LWConcrete"):[65,130,200,270,350,440,550,690,800],("Cu64","Steel"):[15,30,45,60,75,95,115,140,170],
    ("Cu64","Glass"):[50,100,150,200,255,315,390,480,590],
    ("Ga68","Lead"):[5,10,15,20,26,33,41,50,60,72,87],("Ga68","NWConcrete"):[50,100,150,200,250,305,370,450,555],
    ("Ga68","LWConcrete"):[70,140,215,290,375,470,590,740,860],("Ga68","Steel"):[15,30,45,60,80,100,125,155,190],
    ("Ga68","Glass"):[55,110,165,225,285,355,440,545,670],
    ("In111","Lead"):[1,2,3,4,6,8,10,13,17,22,28],("In111","NWConcrete"):[30,60,90,120,155,190,235,290,360,450],
    ("In111","LWConcrete"):[45,90,135,180,230,290,360,450,560,690],("In111","Steel"):[8,16,24,33,43,55,70,88,110],
    ("In111","Glass"):[35,70,105,140,180,225,280,345,430],("In111","Gypsum"):[100,200,300,420,560,720,900],
    ("I123","Lead"):[0.3,0.6,1,1.4,1.8,2.3,2.9,3.6,4.5],("I123","NWConcrete"):[25,50,80,110,145,185,235,295,370],
    ("I123","LWConcrete"):[40,80,120,165,215,270,340,430,540],("I123","Steel"):[5,10,16,22,29,37,47,59,74],
    ("I123","Glass"):[30,65,100,135,175,220,275,340],("I123","Gypsum"):[80,165,255,355,470,610,780],
    ("I124","Lead"):[6,12,18,25,32,40,50,62,76,92,110],("I124","NWConcrete"):[55,110,165,220,280,345,420,515,635],
    ("I124","LWConcrete"):[80,160,240,325,420,525,650,810,940],("I124","Steel"):[18,36,55,75,98,124,155,192,238],
    ("Rb82","Lead"):[5,10,15,20,26,33,41,50,60,72,87],("Rb82","NWConcrete"):[50,100,150,200,250,305,375,460,565],
    ("Rb82","LWConcrete"):[70,140,215,295,380,480,600,750,870],("Rb82","Steel"):[15,30,46,62,82,104,130,162,200],
    ("Rb82","Glass"):[55,110,170,230,295,370,460,570,700],
    ("Ac225","Lead"):[3,6,9,13,18,23,30,38,48,60,75],("Ac225","NWConcrete"):[40,80,120,165,210,260,320,395,490],
    ("Ac225","LWConcrete"):[55,110,170,230,300,380,475,590,730],("Ac225","Steel"):[12,24,37,52,68,87,110,138,172],
    ("Ac225","Glass"):[45,90,135,185,240,300,375,465,575],
    ("At211","Lead"):[0.2,0.4,0.6,0.9,1.2,1.6,2.1,2.8,3.6,4.7],("At211","NWConcrete"):[20,40,65,90,120,155,200,255,325],
    ("At211","LWConcrete"):[30,60,95,130,170,220,280,355,450],("At211","Steel"):[6,12,19,27,36,47,60,77,98],
    ("At211","Glass"):[25,50,80,110,145,185,235,300],("At211","Gypsum"):[60,125,195,275,370,480,615,790],
    ("Y90","Lead"):[5,10,15,20,26,33,41,50,62,76],("Y90","NWConcrete"):[50,100,155,210,270,335,415,515,640],
    ("Xe133","Lead"):[0.5,1,1.5,2,2.7,3.5,4.5,5.7,7.2,9],("Xe133","NWConcrete"):[20,45,70,100,135,175,225,285,360],
    ("Xe133","LWConcrete"):[35,70,110,150,195,250,315,400,505],("Xe133","Steel"):[6,13,20,29,39,51,65,83,105],
    ("Xe133","Glass"):[25,55,85,120,158,202,255,325],("Xe133","Gypsum"):[70,145,225,315,420,545,700,900],
}

def add_custom_materials(sim):
    g_cm3=gate.g4_units.g_cm3; db=sim.volume_manager.material_database
    for name,props in CUSTOM_MATERIALS.items():
        db.add_material_weights(name,list(props["elements"].keys()),list(props["elements"].values()),props["density"]*g_cm3)

def _available_ram_gb():
    try: import psutil; return psutil.virtual_memory().available/1024**3
    except ImportError: return 4.0

def _safe_max_jobs(requested):
    avail=_available_ram_gb(); ram_cap=max(1,int(avail*_RAM_FRACTION/RAM_PER_JOB_GB)); cpu_cap=os.cpu_count() or 1
    safe=min(requested,ram_cap,cpu_cap)
    if safe<requested: print(f"  ⚠  --jobs {requested} reduced to {safe}")
    return safe

def _auto_config(n_tasks=1):
    """
    Automatically configure parallelism based on available CPUs, RAM, and task count.

    Balancing strategy:
      - n_tasks >= ncpu  : fill all cores with parallel jobs, workers=1
      - n_tasks < ncpu   : run all tasks in parallel, give remaining cores to workers
      - n_tasks == 1     : single run, all cores become workers
      - Windows          : threads disabled, cores split between jobs/workers only

    Returns (threads, jobs, workers)
    """
    ncpu = os.cpu_count() or 1
    avail = _available_ram_gb()
    ram_cap = max(1, int(avail * _RAM_FRACTION / RAM_PER_JOB_GB))
    cores = min(ncpu, ram_cap)

    if n_tasks <= 0: n_tasks = 1

    if n_tasks >= cores:
        # Enough tasks to fill all cores — no workers needed
        jobs    = min(cores, n_tasks)
        workers = 1
        threads = 1
    elif n_tasks == 1:
        # Single run — dedicate all cores to workers
        jobs    = 1
        workers = cores
        threads = 1
    else:
        # Partial fill — run all tasks in parallel, remainder goes to workers
        jobs    = n_tasks
        workers = max(1, cores // n_tasks)
        threads = 1

    if _ON_WINDOWS: threads = 1

    print(f"  ⚙  Auto config: {ncpu} CPUs / {avail:.1f} GB RAM  →  ")
    print(f"     jobs={jobs}  workers={workers}  threads={threads}  (n_tasks={n_tasks})")
    if ram_cap < ncpu:
        print(f"     ℹ  RAM-limited to {ram_cap} concurrent worker process(es) "
              f"(budget: {RAM_PER_JOB_GB:.1f} GB/worker x {int(avail*_RAM_FRACTION)} GB usable of {avail:.1f} GB "
              f"available) rather than all {ncpu} CPUs -- each worker is a full "
              f"Python + opengate/scipy/pandas process; use --workers to override.")
    return threads, jobs, workers

def _splitting_factor(barrier_name,thickness_mm,max_factor=100):
    mfp=_BARRIER_MFP_MM.get(barrier_name,100.0)
    if mfp<=0 or thickness_mm<=0: return 1
    return max(1,min(round(1.0/max(math.exp(-thickness_mm/mfp),1e-4)),max_factor))

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_simulation(
    source_label, barrier_name, thickness_mm, n_primaries, output_dir,
    source_type="nuclide", nuclide="F18", kvp=120.0, al_filter_mm=2.5,
    cu_filter_mm=0.0, kv_bins=128, angle_deg=0.0,
    phantom_material="G4_WATER", detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
    detector_size_x_mm=DEFAULT_DETECTOR_XY_MM, detector_size_y_mm=DEFAULT_DETECTOR_XY_MM, detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
    physics_list="auto", tissue_cut_mm=0.01, barrier_cut_mm=0.1,
    verbose=False, threads=1, write_dose=False, write_uncertainty=False,
    cone_source=True, cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
    vis=False, vis_type="vrml_file_only", unc_goal=DEFAULT_UNC_GOAL,
    use_splitting=False,
    source_phantom_shape="none", source_phantom_rx=100.0, source_phantom_ry=70.0,
    source_phantom_rz=100.0, source_phantom_material="G4_WATER",
    source_phantom_ox=0.0, source_phantom_oy=0.0, source_phantom_oz=0.0,
    seed=None,
):
    output_dir.mkdir(parents=True,exist_ok=True)
    stem=_make_stem(source_label,barrier_name,thickness_mm,angle_deg)
    sim=gate.Simulation(); sim.g4_verbose=verbose; sim.visu=vis
    if vis:
        sim.visu_type=vis_type
        if vis_type=="vrml_file_only":
            sim.visu_filename=str(Path(output_dir)/"scene.wrl")
            sim.visu_commands=["/vis/open VRML2FILE","/vis/drawVolume world","/vis/viewer/set/viewpointThetaPhi 65 0","/vis/viewer/zoom 0.1","/vis/viewer/set/style surface","/vis/viewer/flush"]
    sim.random_seed=int(seed) if seed is not None else "auto"; sim.output_dir=str(output_dir)
    if threads>1 and not _ON_WINDOWS: sim.number_of_threads=threads
    add_custom_materials(sim)

    world=sim.world; world.size=[2.0*m,2.0*m,3.0*m]; world.material="G4_AIR"

    actual_thickness_mm=max(thickness_mm,0.001)
    barrier=sim.add_volume("Box","Barrier"); barrier.size=[2.0*m,2.0*m,actual_thickness_mm*mm]
    # Barrier centre fixed at z=0: source is at z=-1.0 m and TissuePhantom
    # centre at z=+1.0 m (below), so this puts source-to-barrier-centre and
    # barrier-centre-to-tissue-centre both at exactly 1.0 m, matching Oumano
    # Sec 2.1's stated distances individually. (Previously 0.77 cm -- an
    # unexplained offset that split the two segments 1007.7 mm / 992.3 mm
    # instead of 1000/1000 mm each. Their SUM was already exact at 1750 mm
    # either way, since the offsets cancelled, so direct/unscattered
    # transmission was unaffected -- but the split between "source-to-barrier"
    # and "barrier-to-detector" air path length matters a little for scatter/
    # buildup geometry, so this is now exact rather than off by ~0.77% each
    # way for no documented reason.)
    barrier.translation=[0,0,0]; barrier.material=BARRIER_MATERIAL_MAP[barrier_name]
    if abs(angle_deg)>0.01:
        from scipy.spatial.transform import Rotation as R
        barrier.rotation=R.from_euler('x',angle_deg,degrees=True).as_matrix()

    tissue=sim.add_volume("Box","TissuePhantom"); tissue.size=[2.0*m,2.0*m,0.5*m]
    tissue.translation=[0,0,1.0*m]; tissue.material=phantom_material

    if source_phantom_shape!="none":
        _sp=source_phantom_shape.lower()
        if _sp=="sphere":
            sp=sim.add_volume("Sphere","SourcePhantom"); sp.rmax=source_phantom_rx*mm; sp.rmin=0.0
        elif _sp=="ellipsoid":
            sp=sim.add_volume("Ellipsoid","SourcePhantom")
            sp.xSemiAxis=source_phantom_rx*mm; sp.ySemiAxis=source_phantom_ry*mm; sp.zSemiAxis=source_phantom_rz*mm
        else: raise ValueError(f"Unknown source_phantom_shape: {source_phantom_shape!r}")
        sp.translation=[source_phantom_ox*mm,source_phantom_oy*mm,-1.0*m+source_phantom_oz*mm]
        sp.material=source_phantom_material

    split_vol=sim.add_volume("Box","SplittingVolume"); split_vol.mother="TissuePhantom"
    split_vol.size=[2.0*m,2.0*m,1.0*mm]; split_vol.translation=[0,0,-0.249*m]; split_vol.material=phantom_material
    if use_splitting:
        sf=_splitting_factor(barrier_name,thickness_mm)
        if sf>1:
            try:
                splitter=sim.add_actor("SplittingActor","PhotonSplitter")
                splitter.attached_to="SplittingVolume"; splitter.splitting_factor=sf; splitter.particle="gamma"
            except: pass

    pm=sim.physics_manager
    if physics_list in (None,"auto"):
        pm.physics_list_name=DEFAULT_PHYSICS_LIST_XRAY if source_type=="xray" else DEFAULT_PHYSICS_LIST_NUCLIDE
    else:
        pm.physics_list_name=physics_list
    pm.enable_decay=False
    # tissue_cut_mm defaults to 0.01 mm (tightened from the original hardcoded
    # 1.0 mm) -- needed to unlock Compton secondaries for low-energy isotope
    # lines; barrier_cut_mm defaults to 0.1 mm, matching the original hardcoded
    # value (tightening it further showed no measured benefit in testing).
    # Both are CLI-tunable via --tissue-cut-mm/--barrier-cut-mm -- pass
    # --tissue-cut-mm 1.0 to restore the old, looser cut if ever needed for an
    # A/B comparison of that pathway's effect on buildup capture.
    pm.global_production_cuts.gamma=tissue_cut_mm*mm; pm.global_production_cuts.electron=tissue_cut_mm*mm; pm.global_production_cuts.positron=tissue_cut_mm*mm
    for p in ("gamma","electron","positron"): pm.set_production_cut("Barrier",p,barrier_cut_mm*mm)
    for region in ("TissuePhantom","SplittingVolume"):
        for p in ("gamma","electron","positron"): pm.set_production_cut(region,p,tissue_cut_mm*mm)
    if source_phantom_shape!="none":
        for p in ("gamma","electron","positron"): pm.set_production_cut("SourcePhantom",p,tissue_cut_mm*mm)

    _add_source(sim,source_type=source_type,nuclide=nuclide,n_primaries=n_primaries,
                cone_source=cone_source,cone_half_angle_deg=cone_half_angle_deg,
                kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,kv_bins=kv_bins,verbose=verbose)

    # ── DoseActor ─────────────────────────────────────────────────────────────
    PHANTOM_HALF_Z_MM=250.0; SPACING_XY=2.5*mm; SPACING_Z=5.0*mm
    dose=sim.add_actor("DoseActor","DoseActorTissue"); dose.attached_to="TissuePhantom"
    dose.output_filename=f"{stem}.mhd"; dose.hit_type="random"
    dose.edep.active=True; dose.edep_uncertainty.active=True   # always write edep+uncertainty for analysis
    dose.dose.active=write_dose; dose.dose_uncertainty.active=write_uncertainty

    if detector_depth_mm is None:
        # LEGACY (--detector-centered): slab centred in the 500 mm phantom, i.e.
        # ~240-260 mm depth -- NOT Oumano's 1 cm point. Kept only so old runs can
        # be reproduced for comparison; every normal entry point now passes an
        # explicit depth (default DEFAULT_DETECTOR_DEPTH_MM) so this branch is
        # only reached by deliberate opt-in.
        sx_mm=detector_size_x_mm if detector_size_x_mm is not None else DEFAULT_DETECTOR_XY_MM
        sy_mm=detector_size_y_mm if detector_size_y_mm is not None else DEFAULT_DETECTOR_XY_MM
        sz_mm=detector_size_z_mm if detector_size_z_mm is not None else DEFAULT_DETECTOR_Z_MM
        nx=max(1,round(sx_mm/(SPACING_XY/mm))); ny=max(1,round(sy_mm/(SPACING_XY/mm))); nz=max(1,round(sz_mm/(SPACING_Z/mm)))
        dose.size=[nx,ny,nz]; dose.spacing=[SPACING_XY,SPACING_XY,SPACING_Z]
        if verbose:
            print(f"  ℹ  Detector: {nx}×{ny}×{nz} voxels ({sx_mm:.1f}×{sy_mm:.1f}×{sz_mm:.1f} mm), centred in phantom  [LEGACY, NOT Oumano-matched]")
    else:
        # Oumano-matched path (the default): slab centred at detector_depth_mm
        # from the tissue face. With the module defaults (depth=10, size_z=20)
        # this spans 0-20 mm, so planes[1:3] = 5-15 mm -- the paper's "2nd and
        # 3rd 5 mm plane" / "depth of 1 cm into the tissue block".
        depth_mm=max(float(detector_depth_mm),0.5)
        sx_mm=detector_size_x_mm if detector_size_x_mm is not None else DEFAULT_DETECTOR_XY_MM
        sy_mm=detector_size_y_mm if detector_size_y_mm is not None else DEFAULT_DETECTOR_XY_MM
        # Always >=4 planes even for a single custom depth query: analysis needs
        # planes[1:3] to average, and shieldLabAnalyze.py's _is_original_actor()
        # requires nZ>=4 or it silently drops the 150 mm ROI mask and falls back
        # to "mean of every voxel" (a bare depth override with no size-z used to
        # hit exactly this trap).
        sz_mm=detector_size_z_mm if detector_size_z_mm is not None else DEFAULT_DETECTOR_Z_MM
        nx=max(1,round(sx_mm/(SPACING_XY/mm))); ny=max(1,round(sy_mm/(SPACING_XY/mm))); nz=max(1,round(sz_mm/(SPACING_Z/mm)))
        dose.size=[nx,ny,nz]; dose.spacing=[SPACING_XY,SPACING_XY,SPACING_Z]
        local_z=(-PHANTOM_HALF_Z_MM+depth_mm)*mm; dose.translation=[0,0,local_z]
        if nz<4 and verbose:
            print(f"  ⚠  detector-size-z gives only {nz} plane(s) — shieldLabAnalyze.py's ROI mask needs nZ>=4 or it will be dropped")
        if verbose:
            print(f"  ℹ  Detector: {nx}×{ny}×{nz} voxels ({sx_mm:.1f}×{sy_mm:.1f}×{sz_mm:.1f} mm) at {depth_mm:.1f} mm depth")

    if unc_goal>0.0:
        try:
            dose.edep.uncertainty_goal=unc_goal
            dose.edep.uncertainty_first_check_after_n_events=max(100_000,n_primaries//100)
            dose.edep.uncertainty_check_every_n_events=1_000_000
        except AttributeError: pass

    if vis:
        # Visual-only box showing exactly where the DoseActor scores, for the
        # .wrl export (--vis / --vis-type vrml_file_only). The DoseActor itself
        # is a scoring overlay, not a G4 volume, so it never appears in
        # /vis/drawVolume world on its own -- this box is what makes the
        # detector's position and extent actually visible in the VRML.
        # Same material as the phantom (phantom_material) and mother=
        # "TissuePhantom" so it's a same-material geometry split, identical in
        # spirit to the existing SplittingVolume: zero physics perturbation,
        # purely a rendering aid. Only created when vis=True so normal
        # production runs get no extra daughter volume/navigation overhead.
        # Size matches the actor's true voxel-rounded extent (nx/ny/nz x
        # spacing), not the raw requested mm, so the wireframe exactly bounds
        # what's actually scored. style="wireframe" + a bright, saturated
        # color keeps it visually distinct from the barrier/phantom in the
        # VRML viewer.
        # Geant4 gets unstable (crashes -- an access-violation dialog, not a
        # Python exception, since it's inside the compiled C++/Geant4 layer)
        # when a daughter volume's face sits EXACTLY on its mother's own
        # face. At the Oumano defaults (depth=10 mm, size_z=20 mm) that's
        # exactly what happens: the box's front face lands at local z =
        # -240-10 = -250 mm, precisely on TissuePhantom's own front face
        # (its half-z is 250 mm). Inset the box by a small epsilon on every
        # side so it's always strictly interior to the phantom, regardless of
        # depth/size settings -- purely cosmetic (0.1 mm is invisible at this
        # scale) but keeps the box off every mother-volume boundary.
        _VIS_EPS=0.1*mm
        _vis_sx=max(0.1*mm,nx*SPACING_XY-_VIS_EPS); _vis_sy=max(0.1*mm,ny*SPACING_XY-_VIS_EPS)
        _vis_sz=max(0.1*mm,nz*SPACING_Z-_VIS_EPS)
        detvis=sim.add_volume("Box","DetectorVis"); detvis.mother="TissuePhantom"
        detvis.material=phantom_material
        detvis.size=[_vis_sx,_vis_sy,_vis_sz]
        detvis.translation=[0,0,local_z] if detector_depth_mm is not None else [0,0,0]
        detvis.color=[1.0,0.15,0.15,1]; detvis.style="wireframe"

    stats=sim.add_actor("SimulationStatisticsActor","Stats"); stats.output_filename=f"{stem}_stats.txt"
    return sim

def _add_source(sim,source_type="nuclide",nuclide="F18",n_primaries=N_PRIMARIES,
                cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
                kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,verbose=False):
    if source_type=="xray":
        spectrum=generate_kv_spectrum(kvp,al_filter_mm,cu_filter_mm,kv_bins)
        if verbose: print(f"  ℹ  kV spectrum: {_kv_spectrum_summary(spectrum,kvp,al_filter_mm,cu_filter_mm)}")
    else: spectrum=PHOTON_SPECTRA[nuclide]
    total_intensity=sum(w for _,w in spectrum)
    src=sim.add_source("GenericSource","PointSource"); src.particle="gamma"; src.n=n_primaries
    src.position.type="point"; src.position.translation=[0,0,-1.0*m]; src.direction.type="iso"
    if cone_source: src.direction.theta=[(180.0-cone_half_angle_deg)*deg,180.0*deg]
    else: src.direction.theta=[90.0*deg,180.0*deg]
    src.direction.phi=[0,360.0*deg]
    if source_type=="nuclide" and len(spectrum)==1:
        src.energy.type="mono"; src.energy.mono=spectrum[0][0]*MeV
    else:
        src.energy.type="spectrum_discrete"
        src.energy.spectrum_energies=[e*MeV for e,_ in spectrum]
        src.energy.spectrum_weights=[w/total_intensity for _,w in spectrum]

def _print_geometry(sim,angle_deg=0.0,tissue_cut_mm=0.01,barrier_cut_mm=0.1):
    bar=sim.volume_manager.volumes["Barrier"]; tis=sim.volume_manager.volumes["TissuePhantom"]
    src=sim.source_manager.sources["PointSource"]; dose=sim.actor_manager.actors["DoseActorTissue"]
    t_phys=bar.size[2]/mm
    t_eff=t_phys/math.cos(math.radians(angle_deg)) if abs(angle_deg)>0.01 else t_phys
    eff_str=f"  (t_eff = {t_eff:.2f} mm at {angle_deg:.0f}°)" if abs(angle_deg)>0.01 else ""
    if src.energy.type=="mono": e_str=f"{src.energy.mono/MeV*1000:.1f} keV (mono)"
    else:
        ek=[round(e/MeV*1000,1) for e in src.energy.spectrum_energies]
        e_str=f"{ek} keV" if len(ek)<=6 else f"{ek[0]:.1f}–{ek[-1]:.1f} keV [{len(ek)} bins]"
    # theta[1] is always 180 deg (fixed upper bound); theta[0]=180-cone_half_angle_deg
    # is the value that actually varies, so the meaningful quantity is the span
    # theta[1]-theta[0], which equals cone_half_angle_deg exactly. (A previous
    # version of this line read theta[1] directly, which is constant, so it
    # always printed "hemisphere 180°" regardless of the true cone angle.)
    theta_extent=(src.direction.theta[1]-src.direction.theta[0])/deg
    cone_str=f"hemisphere {theta_extent:.0f}° (full, 2π toward barrier)" if theta_extent>=89.9 else f"cone {theta_extent:.1f}°"
    nx,ny,nz=dose.size; sx=dose.spacing[0]/mm*nx; sy=dose.spacing[1]/mm*ny; sz=dose.spacing[2]/mm*nz
    print(f"\n  ┌─ Geometry ──────────────────────────────────────────────────────┐")
    print(f"  │ Source   : Z={src.position.translation[2]/m:.2f} m  {cone_str}  {e_str}")
    print(f"  │ Barrier  : {bar.material}, {t_phys:.4g} mm{eff_str}, angle={angle_deg:.0f}°, Z-ctr={bar.translation[2]/cm:.2f} cm")
    print(f"  │ Tissue   : {tis.material}, {tis.size[2]*100:.0f} cm, Z-ctr={tis.translation[2]/m:.1f} m")
    if hasattr(dose,'translation') and dose.translation is not None and dose.translation!=[0,0,0]:
        depth_val=(dose.translation[2]/mm)+250.0
        print(f"  │ Detector : {nx}×{ny}×{nz} voxels ({sx:.1f}×{sy:.1f}×{sz:.1f} mm), centre at {depth_val:.1f} mm from tissue face")
    else:
        print(f"  │ Detector : {nx}×{ny}×{nz} voxels ({sx:.1f}×{sy:.1f}×{sz:.1f} mm), centred in phantom")
    print(f"  │ Physics  : {sim.physics_manager.physics_list_name}  [tissue cuts: {tissue_cut_mm:g} mm, barrier cuts: {barrier_cut_mm:g} mm]")
    vols=sim.volume_manager.volumes
    if "DetectorVis" in vols:
        print(f"  │            (detector region drawn as a red wireframe box in the .wrl)")
    if "SourcePhantom" in vols:
        sp=vols["SourcePhantom"]; sp_mat=sp.material
        if hasattr(sp,'rmax'): sp_desc=f"sphere r={sp.rmax/mm:.1f} mm"
        elif hasattr(sp,'xSemiAxis'): sp_desc=f"ellipsoid rx={sp.xSemiAxis/mm:.1f} ry={sp.ySemiAxis/mm:.1f} rz={sp.zSemiAxis/mm:.1f} mm"
        else: sp_desc="custom"
        ox=sp.translation[0]/mm; oy=sp.translation[1]/mm; oz_from_src=(sp.translation[2]-(-1.0*m))/mm
        off_str=f" offset=({ox:.1f}, {oy:.1f}, {oz_from_src:.1f}) mm" if abs(ox)>0.01 or abs(oy)>0.01 or abs(oz_from_src)>0.01 else " centred on source"
        print(f"  │ Src Phnt : {sp_mat}, {sp_desc}{off_str}")
    print(f"  │ Voxels   : {dose.size}  ({dose.spacing[0]/mm:.1f}×{dose.spacing[1]/mm:.1f}×{dose.spacing[2]/mm:.1f} mm)")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

def _reference_exists(source_label,output_dir):
    return (output_dir/f"{_make_stem(source_label,'Air',0.0,0.0)}_edep.mhd").exists()

def _read_transmission(source_label,barrier,thickness_mm,angle_deg,output_dir):
    try: import itk; _use_sitk=False
    except ImportError:
        try: import SimpleITK as sitk; _use_sitk=True
        except ImportError: return None
    def _load(path):
        if not path.exists(): return None
        try:
            if _use_sitk: return float(sitk.GetArrayFromImage(sitk.ReadImage(str(path))).mean())
            else: return float(itk.GetArrayFromImage(itk.imread(str(path))).mean())
        except: return None
    air=_load(output_dir/f"{_make_stem(source_label,'Air',0.0,0.0)}_edep.mhd")
    bar=_load(output_dir/f"{_make_stem(source_label,barrier,thickness_mm,angle_deg)}_edep.mhd")
    if air is None or bar is None or air<=0: return None
    return bar/air

# ─────────────────────────────────────────────────────────────────────────────
# RUN HELPERS — all signatures include detector_size_x/y/z_mm
# ─────────────────────────────────────────────────────────────────────────────

def run_single(source_label,barrier,thickness_mm,n_primaries,output_dir,
               source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,
               angle_deg=0.0,verbose=False,threads=1,write_dose=False,write_uncertainty=False,
               cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
               vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
               phantom_material="G4_WATER",detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
               detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
               physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
               source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
               source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0,
               seed=None):
    sim=build_simulation(source_label=source_label,barrier_name=barrier,thickness_mm=thickness_mm,n_primaries=n_primaries,
        output_dir=output_dir,source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,
        cu_filter_mm=cu_filter_mm,kv_bins=kv_bins,angle_deg=angle_deg,verbose=verbose,threads=threads,
        write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        cone_half_angle_deg=cone_half_angle_deg,vis=vis,vis_type=vis_type,unc_goal=unc_goal,use_splitting=use_splitting,
        phantom_material=phantom_material,physics_list=physics_list,tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,
        detector_depth_mm=detector_depth_mm,
        detector_size_x_mm=detector_size_x_mm,detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,source_phantom_ry=source_phantom_ry,
        source_phantom_rz=source_phantom_rz,source_phantom_material=source_phantom_material,
        source_phantom_ox=source_phantom_ox,source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz,
        seed=seed)
    _print_geometry(sim,angle_deg,tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm); sim.run()


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

def _merge_worker_outputs(worker_dirs, final_output_dir, n_workers):
    """
    Combine dose MHD files from N independent parallel worker runs.

    GATE outputs TOTAL energy deposited (not per-primary normalised), so:
        dose_combined  = sum(dose_i)       -- total edep across all N primaries
        unc_combined   = sqrt(sum((unc_i * dose_i)^2)) / |dose_combined|
                                           -- propagated absolute uncertainty / total
    This exactly matches a single run of N primaries.
    """
    import SimpleITK as sitk
    dose_arrays = []; unc_arrays = []; ref_img = None

    for wdir in worker_dirs:
        wdir = Path(wdir)
        candidates = list(wdir.glob("*_edep.mhd"))
        if not candidates:
            raise FileNotFoundError(f"No *_edep.mhd found in worker dir: {wdir}")
        dose_f = candidates[0]
        unc_candidates = list(wdir.glob("*_edep_uncertainty.mhd"))

        img = sitk.ReadImage(str(dose_f))
        dose_arrays.append(sitk.GetArrayFromImage(img).astype(np.float64))
        if ref_img is None: ref_img = img

        if unc_candidates:
            unc_arrays.append(sitk.GetArrayFromImage(sitk.ReadImage(str(unc_candidates[0]))).astype(np.float64))

    dose_stack = np.stack(dose_arrays, axis=0)

    # SUM — GATE edep is total deposited energy, not per-primary normalised
    dose_combined = np.sum(dose_stack, axis=0)

    final_output_dir = Path(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Write merged dose
    out_dose = sitk.GetImageFromArray(dose_combined.astype(np.float32))
    out_dose.CopyInformation(ref_img)
    stem_name = Path(list(Path(worker_dirs[0]).glob("*_edep.mhd"))[0]).name
    sitk.WriteImage(out_dose, str(final_output_dir / stem_name))

    # Write merged uncertainty
    # Absolute std dev propagation: sigma_combined = sqrt(sum(sigma_i^2))
    #   where sigma_i = unc_i * dose_i  (relative -> absolute)
    # Relative combined: unc_combined = sigma_combined / dose_combined
    if unc_arrays:
        unc_stack    = np.stack(unc_arrays, axis=0)
        abs_var      = np.sum((unc_stack * dose_stack) ** 2, axis=0)
        denom        = np.abs(dose_combined)
        denom        = np.where(denom < 1e-300, 1e-300, denom)
        unc_combined = np.sqrt(abs_var) / denom
        out_unc = sitk.GetImageFromArray(unc_combined.astype(np.float32))
        out_unc.CopyInformation(ref_img)
        unc_stem = stem_name.replace("_edep.mhd", "_edep_uncertainty.mhd")
        sitk.WriteImage(out_unc, str(final_output_dir / unc_stem))

    max_unc  = float(np.max(unc_combined))  if unc_arrays else float("nan")
    mean_unc = float(np.mean(unc_combined[unc_combined > 0])) if unc_arrays else float("nan")
    print(f"  v  Merged {n_workers} workers -> {final_output_dir.name}")
    print(f"     Combined uncertainty: mean={mean_unc:.4f}  max={max_unc:.4f}  (expected ~1/sqrt({n_workers}) of single-worker unc)")

    # Merge stats files — use read_n_primaries logic to correctly parse GATE format
    stats_files = []
    for wdir in worker_dirs:
        hits = list(Path(wdir).glob("*_stats.txt"))
        if hits:
            stats_files.append(hits[0])

    if stats_files:
        import re as _re, json as _json
        total_events = 0
        elapsed_times = []

        def _parse_n_from_stats(sp):
            """Parse NumberOfEvents from a GATE stats file (JSON or text)."""
            try:
                txt = sp.read_text(errors="replace")
                # Try JSON first
                try:
                    d = _json.loads(txt)
                    for k in ("events","nb_events","NumberOfEvents"):
                        if k in d:
                            v = d[k]
                            if isinstance(v, dict): v = v.get("value", v)
                            return int(v)
                except Exception:
                    pass
                # Plain text
                for line in txt.splitlines():
                    for k in ("NumberOfEvents","Number of events","Events","nb_events"):
                        if k.lower() in line.lower() and "=" in line:
                            try:
                                return int(line.split("=")[-1].strip().split()[0].replace(",",""))
                            except Exception:
                                pass
            except Exception:
                pass
            return 0

        def _parse_elapsed(sp):
            try:
                txt = sp.read_text(errors="replace")
                for line in txt.splitlines():
                    if "ElapsedTimeWall" in line and "=" in line:
                        try:
                            return float(line.split("=")[-1].strip().split()[0])
                        except Exception:
                            pass
            except Exception:
                pass
            return 0.0

        for sf in stats_files:
            n = _parse_n_from_stats(sf)
            total_events += n
            t = _parse_elapsed(sf)
            if t > 0:
                elapsed_times.append(t)

        # Copy first worker stats as template then overwrite key fields
        stats_stem = stem_name.replace("_edep.mhd", "_stats.txt")
        stats_out  = final_output_dir / stats_stem

        try:
            template = stats_files[0].read_text(errors="replace")
        except Exception:
            template = ""

        with open(stats_out, "w") as sf:
            sf.write(f"# Combined stats from {n_workers} parallel workers\n")
            sf.write(f"# Individual stats summed from worker runs\n")
            sf.write(f"NumberOfEvents  = {total_events}\n")
            if elapsed_times:
                sf.write(f"ElapsedTimeWall = {max(elapsed_times):.2f} s  "
                         f"(longest worker; sum={sum(elapsed_times):.2f} s)\n")
            sf.write(f"NumberOfWorkers = {n_workers}\n")
            # Append original template for reference
            if template:
                sf.write(f"\n# --- Worker 0 stats (for reference) ---\n")
                sf.write(template)

        print(f"     Stats saved -> {stats_out.name}  (N={total_events:,})")




def _generate_worker_seeds(n_workers, base_seed=None):
    """
    Generate n_workers fully independent seeds using NumPy SeedSequence.

    Two-output design:
      geant4_seeds  — 31-bit integers required by Geant4/CLHEP for actual simulation
      child_entropy — full 128-bit entropy tuples for use as base_seed at the next
                      hierarchy level (task -> workers), preventing entropy collapse
                      that occurs when 31-bit Geant4 seeds are fed back into SeedSequence

    Hierarchy guarantee:
      Parent SeedSequence spawns n_workers children via a hash-based mixing algorithm.
      Children are guaranteed statistically independent regardless of base_seed value.
      Passing child_entropy[i] as base_seed to the next level preserves full 128-bit
      independence across ALL workers at ALL levels — no correlation possible.
    """
    seq      = np.random.SeedSequence(base_seed)   # OS entropy if base_seed is None
    children = seq.spawn(n_workers)                # n_workers independent child sequences

    # Geant4 seeds: 31-bit integers for CLHEP RNG initialisation
    geant4_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF for c in children]
    assert len(set(geant4_seeds)) == n_workers, "Duplicate Geant4 seeds — increase entropy"

    # Full-entropy child states: tuples preserved for hierarchical spawning
    # Using 4 x uint32 = 128 bits per child — sufficient for any downstream hierarchy
    child_entropy = [tuple(int(x) for x in c.generate_state(4, dtype=np.uint32)) for c in children]

    entropy_out = int(seq.entropy) if base_seed is None else base_seed
    return geant4_seeds, child_entropy, entropy_out


def _child_seed_to_geant4(child_ent):
    """Convert a 128-bit child entropy tuple to a 31-bit Geant4 seed."""
    seq = np.random.SeedSequence(list(child_ent))
    return int(seq.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF


def _run_workers(source_label, barrier, thickness_mm, n_primaries, output_dir,
                 n_workers=1, base_seed=None, **kw):
    """
    Split n_primaries across n_workers independent GATE processes, each with a
    provably uncorrelated random seed via NumPy SeedSequence.
    Merge results into output_dir when all finish.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if n_workers <= 1:
        rc = _spawn(source_label, barrier, thickness_mm, n_primaries, Path(output_dir), **kw)
        return 0 if rc == 0 else 1

    n_per_worker = max(1, n_primaries // n_workers)

    # Generate independent seeds — full 128-bit entropy hierarchy
    worker_seeds, _, entropy = _generate_worker_seeds(n_workers, base_seed)

    print(f"  ⚡  Workers: {n_workers}  x  {n_per_worker:,} primaries  (SeedSequence entropy={entropy})")
    print(f"  🎲  Worker seeds (Geant4): {worker_seeds}")

    # Create temp directories for each worker.
    #
    # CRITICAL: this name must be unique to THIS specific invocation, not just
    # to the --output directory. Naming it purely from output_dir (the old
    # behavior) caused two confirmed real collisions on Isaac:
    #   1. Two DIFFERENT cases (90mm and 100mm) sharing one --output ran
    #      concurrently and computed the IDENTICAL tmp_root -> their worker
    #      processes raced to write the same wNNN paths at the same time.
    #   2. A run reused an --output directory name that a PRIOR, unrelated
    #      run (different thicknesses entirely) had also used at some
    #      earlier time -> orphaned leftover worker dirs from that old run
    #      (never cleaned up, since cleanup only happens after a fully
    #      successful non-partial merge) were silently picked up and mixed
    #      into the new run's worker set.
    #
    # Fix: fold in the case's own identifying stem (source_label + barrier +
    # thickness, via the existing _make_stem() helper) AND the current
    # process's PID, so that neither same-output/different-case collisions
    # nor same-case/different-time collisions with orphaned directories can
    # occur. If a stale tmp_root with this exact name somehow still exists
    # (e.g. a retry with the same PID reused by the OS, astronomically
    # unlikely but not impossible), fail loudly rather than silently reusing
    # whatever is in it.
    stem = _make_stem(source_label, barrier, thickness_mm, kw.get("angle_deg", 0.0))
    tmp_root = Path(output_dir).parent / f"_workers_{Path(output_dir).name}_{stem}_pid{os.getpid()}"
    if tmp_root.exists():
        raise RuntimeError(
            f"tmp_root already exists and would be silently reused: {tmp_root}\n"
            f"This should not happen (name includes case stem + PID) -- "
            f"remove it manually after confirming it isn't from a still-running "
            f"job before retrying."
        )
    tmp_root.mkdir(parents=True, exist_ok=True)
    worker_dirs = [tmp_root / f"w{i:03d}" for i in range(n_workers)]

    # Disable unc_goal per worker — run full count then merge for statistics
    worker_kw = dict(kw)
    worker_kw["unc_goal"] = 0.0

    def _run_one(i):
        return _spawn(source_label, barrier, thickness_mm, n_per_worker,
                      worker_dirs[i], seed=worker_seeds[i], **worker_kw)

    # Run all workers in parallel via thread pool
    t_start = time.perf_counter()
    worker_times = {}
    failed = []

    def _run_one_timed(i):
        t0 = time.perf_counter()
        rc = _spawn(source_label, barrier, thickness_mm, n_per_worker,
                    worker_dirs[i], seed=worker_seeds[i], **worker_kw)
        return rc, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_timed, i): i for i in range(n_workers)}
        for fut in as_completed(futures):
            i = futures[fut]
            rc, elapsed = fut.result()
            worker_times[i] = elapsed
            if rc != 0:
                print(f"  x  Worker {i} failed (exit {rc})  [{elapsed:.1f}s]")
                failed.append(i)
            else:
                print(f"  v  Worker {i} done  [{elapsed:.1f}s]  (seed={worker_seeds[i]})")

    t_parallel = time.perf_counter() - t_start

    # ── Merge whatever succeeded -- NEVER silently discard surviving workers ──
    # Previously: any single worker failure skipped the merge entirely (all
    # other workers' completed compute silently thrown away) and returned
    # None with no exception and no nonzero exit code all the way up through
    # main() -- so a partially-failed run looked IDENTICAL to a fully
    # successful one to SLURM (exit 0, job shows COMPLETED), even though no
    # merged .mhd/_stats.txt was ever written. This is the "workers didn't
    # recombine when done" failure mode. Fixed: merge using whatever workers
    # DID succeed (as long as at least one did), clearly flag it as a
    # PARTIAL/degraded merge (fewer effective primaries than requested), and
    # return a distinguishable nonzero status so the caller can propagate a
    # real, visible failure code instead of a silent success.
    succeeded = [i for i in range(n_workers) if i not in failed]
    if not succeeded:
        print(f"  X  ALL {n_workers} workers failed — nothing to merge. "
              f"Worker dirs kept at {tmp_root} for inspection.")
        return 1

    partial = bool(failed)
    if partial:
        print(f"  !  {len(failed)}/{n_workers} worker(s) failed (indices: {failed}) — "
              f"merging the {len(succeeded)} that succeeded. This is a "
              f"DEGRADED/PARTIAL result: effective N is "
              f"{len(succeeded)}/{n_workers} of what was requested "
              f"({n_per_worker * len(succeeded):,} of "
              f"{n_per_worker * n_workers:,} primaries). Worker dirs kept "
              f"at {tmp_root} — do not delete until you've confirmed this "
              f"result is acceptable or decided to rerun the failed indices.")

    # Merge surviving worker outputs into final directory
    surviving_dirs = [worker_dirs[i] for i in succeeded]
    print(f"  Merging {len(succeeded)} worker output(s)"
          f"{' (partial)' if partial else ''}...")
    t_merge = time.perf_counter()
    _merge_worker_outputs([str(d) for d in surviving_dirs], output_dir, len(succeeded))
    t_merge = time.perf_counter() - t_merge

    # Clean up temp worker directories -- ONLY on a full, non-degraded merge.
    # A partial merge keeps every worker dir (including the failed ones'
    # partial/empty output) so a rerun-just-the-failed-indices recovery pass
    # or manual inspection is still possible afterward.
    if not partial:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ── Runtime summary ──────────────────────────────────────────────────────
    t_total     = time.perf_counter() - t_start
    t_sequential = sum(worker_times.values())
    speedup     = t_sequential / t_parallel if t_parallel > 0 else 0

    def _fmt(s):
        h,r = divmod(int(s), 3600); m,sec = divmod(r, 60)
        return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s" if m else f"{sec:02d}s"

    print(f"\n  ┌─ Worker Runtime Summary ──────────────────────────────────┐")
    print(f"  │  Workers       : {n_workers}  x  {n_per_worker:,} primaries"
          f"{f'  ({len(failed)} FAILED)' if partial else ''}")
    print(f"  │  Parallel wall : {_fmt(t_parallel):<12}  (all workers running)")
    print(f"  │  Sequential eq : {_fmt(t_sequential):<12}  (sum of worker times)")
    print(f"  │  Merge time    : {_fmt(t_merge):<12}")
    print(f"  │  Total wall    : {_fmt(t_total):<12}")
    print(f"  │  Speedup       : {speedup:.1f}x")
    fastest = min(worker_times, key=worker_times.get)
    slowest = max(worker_times, key=worker_times.get)
    print(f"  │  Fastest worker: #{fastest}  ({_fmt(worker_times[fastest])})")
    print(f"  │  Slowest worker: #{slowest}  ({_fmt(worker_times[slowest])})")
    if partial:
        print(f"  │  STATUS        : PARTIAL -- {len(failed)}/{n_workers} worker(s) "
              f"failed, merged result uses only {len(succeeded)}/{n_workers}")
    print(f"  └───────────────────────────────────────────────────────────┘")

    return 2 if partial else 0

def _spawn(source_label,barrier,thickness_mm,n_primaries,output_dir,
           source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,
           angle_deg=0.0,threads=1,verbose=False,write_dose=False,write_uncertainty=False,
           cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
           vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
           phantom_material="G4_WATER",detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
           detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
           physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
           source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
           source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0,
           seed=None):
    cmd=[sys.executable,__file__,"--source-type",source_type,"--nuclide",nuclide,"--barrier",barrier,
         "--thickness",str(thickness_mm),"--n",str(n_primaries),"--output",str(output_dir),
         "--threads",str(threads),"--unc-goal",str(unc_goal),"--angle",str(angle_deg),
         "--kvp",str(kvp),"--al-filter",str(al_filter_mm),"--cu-filter",str(cu_filter_mm),
         "--kv-bins",str(kv_bins),"--cone-angle-deg",str(cone_half_angle_deg),
         "--phantom-material",phantom_material,"--physics-list",physics_list,
         "--tissue-cut-mm",str(tissue_cut_mm),"--barrier-cut-mm",str(barrier_cut_mm),
         "--source-phantom-shape",source_phantom_shape,"--source-phantom-rx",str(source_phantom_rx),
         "--source-phantom-ry",str(source_phantom_ry),"--source-phantom-rz",str(source_phantom_rz),
         "--source-phantom-material",source_phantom_material,
         "--source-phantom-ox",str(source_phantom_ox),"--source-phantom-oy",str(source_phantom_oy),
         "--source-phantom-oz",str(source_phantom_oz)]
    if detector_depth_mm is not None: cmd.extend(["--detector-depth",str(detector_depth_mm)])
    else: cmd.append("--detector-centered")  # None means legacy/opt-in mode - must be explicit for the
                                              # subprocess, since the child's own --detector-depth default
                                              # is now Oumano's 10 mm, not None (omitting the flag here
                                              # would silently give the child the wrong geometry)
    if detector_size_x_mm is not None: cmd.extend(["--detector-size-x",str(detector_size_x_mm)])
    if detector_size_y_mm is not None: cmd.extend(["--detector-size-y",str(detector_size_y_mm)])
    if detector_size_z_mm is not None: cmd.extend(["--detector-size-z",str(detector_size_z_mm)])
    if verbose: cmd.append("--verbose")
    if write_dose: cmd.append("--dose")
    if write_uncertainty: cmd.append("--uncertainty")
    if not cone_source: cmd.append("--no-cone")
    if use_splitting: cmd.append("--split")
    if vis: cmd.extend(["--vis","--vis-type",vis_type])
    if seed is not None: cmd.extend(["--seed",str(seed)])
    label=f"{source_label}/{barrier}/{thickness_mm} mm a={angle_deg:.0f}° N={n_primaries:,}"
    print(f"\n  ▶  {label}")
    result=subprocess.run(cmd,capture_output=(not verbose))
    if result.returncode==0: print(f"  ✓  {label}")
    else:
        print(f"  ✗  {label}  (exit {result.returncode})")
        if not verbose and result.stderr:
            for line in result.stderr.decode(errors="replace").strip().splitlines()[-8:]: print(f"     {line}")
    return result.returncode

def _run_tasks_parallel(tasks, n_primaries, output_dir, threads=1, max_jobs=1, verbose=False,
                        n_workers=1, base_seed=None,
                        source_type="nuclide", nuclide="F18", kvp=120.0, al_filter_mm=2.5, cu_filter_mm=0.0,
                        kv_bins=128, write_dose=False, write_uncertainty=False, cone_source=True,
                        unc_goal=DEFAULT_UNC_GOAL, use_splitting=False,
                        phantom_material="G4_WATER", detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
                        detector_size_x_mm=DEFAULT_DETECTOR_XY_MM, detector_size_y_mm=DEFAULT_DETECTOR_XY_MM, detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
                        physics_list="auto", tissue_cut_mm=0.01, barrier_cut_mm=0.1,
                        source_phantom_shape="none", source_phantom_rx=100.0, source_phantom_ry=70.0,
                        source_phantom_rz=100.0, source_phantom_material="G4_WATER",
                        source_phantom_ox=0.0, source_phantom_oy=0.0, source_phantom_oz=0.0):
    """
    Dispatch a list of (label, barrier, thickness, angle) tasks.

    Parallelism:
      max_jobs  = tasks running simultaneously
      n_workers = sub-processes per task (splits N primaries)
      Total cores used = max_jobs x n_workers
    """
    import time as _time
    kw = dict(source_type=source_type, nuclide=nuclide, kvp=kvp, al_filter_mm=al_filter_mm,
              cu_filter_mm=cu_filter_mm, kv_bins=kv_bins, write_dose=write_dose,
              write_uncertainty=write_uncertainty, cone_source=cone_source,
              unc_goal=unc_goal, use_splitting=use_splitting, phantom_material=phantom_material,
              physics_list=physics_list, tissue_cut_mm=tissue_cut_mm, barrier_cut_mm=barrier_cut_mm,
              detector_depth_mm=detector_depth_mm, detector_size_x_mm=detector_size_x_mm,
              detector_size_y_mm=detector_size_y_mm, detector_size_z_mm=detector_size_z_mm,
              source_phantom_shape=source_phantom_shape, source_phantom_rx=source_phantom_rx,
              source_phantom_ry=source_phantom_ry, source_phantom_rz=source_phantom_rz,
              source_phantom_material=source_phantom_material,
              source_phantom_ox=source_phantom_ox, source_phantom_oy=source_phantom_oy,
              source_phantom_oz=source_phantom_oz)

    # Per-task seeds — child_entropy preserves full 128-bit independence across levels
    if n_workers > 1 or base_seed is not None:
        task_geant4, task_child_entropy, _ = _generate_worker_seeds(len(tasks), base_seed)
    else:
        task_geant4        = [None] * len(tasks)
        task_child_entropy = [None] * len(tasks)

    def _dispatch(label, bar, t, ang, geant4_seed, child_ent):
        if n_workers > 1:
            # Pass full 128-bit child entropy as base_seed — prevents entropy collapse
            task_outdir = Path(output_dir) / f"{label}_{bar}_{t}mm"
            # Previously this hardcoded `return 0` regardless of what
            # _run_workers() actually returned -- meaning EVERY task in a
            # --sweep --workers N run was reported successful even if that
            # task's merge was partial (some workers failed) or total
            # failure (all workers failed, nothing merged at all). This is
            # the same "workers didn't recombine when done" failure mode as
            # the plain single-run path, but worse here since it explicitly
            # discarded a real return value instead of merely failing to
            # check one. Now propagates _run_workers()'s real status
            # (0=full success, 2=partial, 1=total failure) so the caller's
            # failed-task bookkeeping (see `failed.append(...)` below) is
            # accurate instead of unconditionally green.
            return _run_workers(label, bar, t, n_primaries, task_outdir,
                         n_workers=n_workers, base_seed=child_ent,
                         threads=threads, verbose=verbose, **kw)
        return _spawn(label, bar, t, n_primaries, output_dir,
                      angle_deg=ang, threads=threads, verbose=verbose,
                      seed=geant4_seed, **kw)

    def _fmt(s):
        h, r = divmod(int(s), 3600); m, sec = divmod(r, 60)
        return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s" if m else f"{sec:02d}s"

    n_tasks = len(tasks)
    print(f"\n  Sweep pool : {max_jobs} jobs × {n_workers} workers × {threads} threads")
    print(f"  Total cores: {max_jobs * n_workers}  |  Tasks: {n_tasks}")

    t_start = _time.perf_counter()
    task_times = {}
    failed = []

    def _dispatch_timed(i, label, bar, t, ang):
        t0 = _time.perf_counter()
        rc = _dispatch(label, bar, t, ang, task_geant4[i], task_child_entropy[i])
        elapsed = _time.perf_counter() - t0
        return rc, elapsed

    if max_jobs == 1:
        for i, (label, bar, t, ang) in enumerate(tasks):
            rc, elapsed = _dispatch_timed(i, label, bar, t, ang)
            task_times[f"{bar}/{t}mm"] = elapsed
            if rc != 0: failed.append((label, bar, t, ang, rc))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=max_jobs) as pool:
            for i, (label, bar, t, ang) in enumerate(tasks):
                fut = pool.submit(_dispatch_timed, i, label, bar, t, ang)
                futures[fut] = (label, bar, t, ang)  
            for fut in as_completed(futures):
                label, bar, t, ang = futures[fut]
                try:
                    rc, elapsed = fut.result()
                    task_times[f"{bar}/{t}mm"] = elapsed
                    status = "✓" if rc == 0 else "✗"
                    print(f"  {status}  {label}/{bar}/{t}mm  [{_fmt(elapsed)}]")
                    if rc is not None and rc != 0:
                        failed.append((label, bar, t, ang, rc))
                except Exception as exc:
                    print(f"  ✗  {label}/{bar}/{t}mm raised {exc}")
                    failed.append((label, bar, t, ang, -1))

    t_total = _time.perf_counter() - t_start
    t_sequential = sum(task_times.values())
    speedup = t_sequential / t_total if t_total > 0 and max_jobs > 1 else 1.0

    print(f"\n  ┌─ Sweep Runtime Summary ────────────────────────────────────┐")
    print(f"  │  Tasks         : {n_tasks}  ({max_jobs} parallel × {n_workers} workers each)")
    print(f"  │  Wall time     : {_fmt(t_total):<12}")
    print(f"  │  Sequential eq : {_fmt(t_sequential):<12}  (sum of all task times)")
    if max_jobs > 1:
        print(f"  │  Speedup       : {speedup:.1f}x")
    if task_times:
        slowest_k = max(task_times, key=task_times.get)
        fastest_k = min(task_times, key=task_times.get)
        print(f"  │  Slowest task  : {slowest_k}  [{_fmt(task_times[slowest_k])}]")
        print(f"  │  Fastest task  : {fastest_k}  [{_fmt(task_times[fastest_k])}]")
    print(f"  └───────────────────────────────────────────────────────────┘")

    return failed



def _common_kw(args):
    return dict(source_type=args.source_type,nuclide=args.nuclide,kvp=args.kvp,al_filter_mm=args.al_filter,
                cu_filter_mm=args.cu_filter,kv_bins=args.kv_bins,write_dose=args.dose,write_uncertainty=args.uncertainty,
                cone_source=not args.no_cone,cone_half_angle_deg=args.cone_angle_deg,vis=args.vis,vis_type=args.vis_type,
                unc_goal=args.unc_goal,use_splitting=args.split,phantom_material=args.phantom_material,
                physics_list=args.physics_list,tissue_cut_mm=args.tissue_cut_mm,barrier_cut_mm=args.barrier_cut_mm,
                detector_depth_mm=args.detector_depth,
                detector_size_x_mm=args.detector_size_x,detector_size_y_mm=args.detector_size_y,
                detector_size_z_mm=args.detector_size_z,
                source_phantom_shape=args.source_phantom_shape,source_phantom_rx=args.source_phantom_rx,
                source_phantom_ry=args.source_phantom_ry,source_phantom_rz=args.source_phantom_rz,
                source_phantom_material=args.source_phantom_material,
                source_phantom_ox=args.source_phantom_ox,source_phantom_oy=args.source_phantom_oy,
                source_phantom_oz=args.source_phantom_oz)

def run_sweep(source_label,barrier,n_primaries,output_dir,angle_deg=0.0,verbose=False,threads=1,max_jobs=1,
              n_workers=1,base_seed=None,
              source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,
              write_dose=False,write_uncertainty=False,cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
              vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
              phantom_material="G4_WATER",detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
              detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
              physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
              source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
              source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    lookup=(nuclide,barrier)
    if source_type=="nuclide" and lookup not in THICKNESS_SWEEPS:
        print(f"No sweep table for {nuclide}/{barrier}."); sys.exit(1)
    thicknesses=THICKNESS_SWEEPS[lookup] if source_type=="nuclide" else _kv_default_thicknesses(barrier)
    tasks=[(source_label,barrier,float(t),angle_deg) for t in thicknesses]
    if _reference_exists(source_label,output_dir): print(f"  ℹ  Air reference exists — skipping.")
    else: tasks.append((source_label,"Air",0.0,0.0))
    print(f"\nSweep: {source_label}/{barrier} angle={angle_deg:.0f}° ({len(thicknesses)} points)")
    failed=_run_tasks_parallel(tasks,n_primaries,output_dir,threads,max_jobs,verbose,
        source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,physics_list=physics_list,
        tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,
        detector_depth_mm=detector_depth_mm,detector_size_x_mm=detector_size_x_mm,
        detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,
        source_phantom_ry=source_phantom_ry,source_phantom_rz=source_phantom_rz,
        source_phantom_material=source_phantom_material,source_phantom_ox=source_phantom_ox,
        source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if failed: print(f"\n  ✗ {len(failed)} failed"); sys.exit(1)
    print(f"\n  ✓ Sweep complete: {source_label}/{barrier}")

def run_angle_sweep(source_label,barrier,thickness_mm,n_primaries,output_dir,angles=None,verbose=False,threads=1,max_jobs=1,
                    source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,
                    write_dose=False,write_uncertainty=False,cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
                    vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
                    phantom_material="G4_WATER",detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
                    detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
                    physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
                    source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
                    source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    if angles is None: angles=ANGLE_SWEEP_DEG
    tasks=[(source_label,barrier,float(thickness_mm),float(a)) for a in angles]
    if not _reference_exists(source_label,output_dir): tasks.append((source_label,"Air",0.0,0.0))
    failed=_run_tasks_parallel(tasks,n_primaries,output_dir,threads,max_jobs,verbose,
        source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,physics_list=physics_list,
        tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,
        detector_depth_mm=detector_depth_mm,detector_size_x_mm=detector_size_x_mm,
        detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,
        source_phantom_ry=source_phantom_ry,source_phantom_rz=source_phantom_rz,
        source_phantom_material=source_phantom_material,source_phantom_ox=source_phantom_ox,
        source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if failed: print(f"\n  ✗ {len(failed)} failed"); sys.exit(1)
    print(f"\n  ✓ Angle sweep complete")

def run_nuclide_sweep(source_label,n_primaries,output_dir,nuclide="F18",sweep_to_cvl=False,angle_deg=0.0,
                      verbose=False,threads=1,max_jobs=1,source_type="nuclide",kvp=120.0,al_filter_mm=2.5,
                      cu_filter_mm=0.0,kv_bins=128,write_dose=False,write_uncertainty=False,
                      cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
                      vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
                      phantom_material="G4_WATER",detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,
                      detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
                      physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
                      source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
                      source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    spawn_kw=dict(source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,angle_deg=angle_deg,threads=threads,verbose=verbose,write_dose=write_dose,
        write_uncertainty=write_uncertainty,cone_source=cone_source,cone_half_angle_deg=cone_half_angle_deg,
        vis=vis,vis_type=vis_type,unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
        physics_list=physics_list,tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,
        detector_depth_mm=detector_depth_mm,detector_size_x_mm=detector_size_x_mm,
        detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,
        source_phantom_ry=source_phantom_ry,source_phantom_rz=source_phantom_rz,
        source_phantom_material=source_phantom_material,source_phantom_ox=source_phantom_ox,
        source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if source_type=="nuclide":
        barriers=sorted({bar for nuc,bar in THICKNESS_SWEEPS if nuc==nuclide})
        if not barriers: print(f"No sweep tables for {nuclide}."); sys.exit(1)
        if sweep_to_cvl:
            if not _reference_exists(source_label,output_dir):
                rc=_spawn(source_label,"Air",0.0,n_primaries,output_dir,**spawn_kw)
                if rc!=0: sys.exit(1)
            all_failed=[]; total_ran=0; cvl_reached={}
            for barrier in barriers:
                thicknesses=THICKNESS_SWEEPS[(nuclide,barrier)]
                for t in thicknesses:
                    rc=_spawn(source_label,barrier,float(t),n_primaries,output_dir,**spawn_kw); total_ran+=1
                    if rc!=0: all_failed.append((source_label,barrier,t,angle_deg,rc)); continue
                    T=_read_transmission(source_label,barrier,float(t),angle_deg,output_dir)
                    if T is not None and T<=CVL_THRESHOLD: cvl_reached[barrier]=t; break
            if all_failed: sys.exit(1)
            return
        tasks=[(source_label,barrier,float(t),angle_deg) for barrier in barriers for t in THICKNESS_SWEEPS[(nuclide,barrier)]]
    else:
        barriers=[b for b in BARRIER_MATERIAL_MAP if b!="Air"]
        tasks=[(source_label,barrier,float(t),angle_deg) for barrier in barriers for t in _kv_default_thicknesses(barrier)]
    if not _reference_exists(source_label,output_dir): tasks.append((source_label,"Air",0.0,0.0))
    failed=_run_tasks_parallel(tasks,n_primaries,output_dir,threads,max_jobs,verbose,
        source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,physics_list=physics_list,
        tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,
        detector_depth_mm=detector_depth_mm,detector_size_x_mm=detector_size_x_mm,
        detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,
        source_phantom_ry=source_phantom_ry,source_phantom_rz=source_phantom_rz,
        source_phantom_material=source_phantom_material,source_phantom_ox=source_phantom_ox,
        source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if failed: sys.exit(1)

def run_reference(source_label,n_primaries,output_dir,source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,
                  cu_filter_mm=0.0,kv_bins=128,verbose=False,threads=1,write_dose=False,write_uncertainty=False,
                  cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,vis=False,vis_type="vrml_file_only",
                  unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,phantom_material="G4_WATER",
                  detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
                  physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
                  source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
                  source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    rc=_spawn(source_label,"Air",0.0,n_primaries,output_dir,source_type=source_type,nuclide=nuclide,kvp=kvp,
              al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,kv_bins=kv_bins,angle_deg=0.0,threads=threads,
              verbose=verbose,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
              cone_half_angle_deg=cone_half_angle_deg,vis=vis,vis_type=vis_type,unc_goal=unc_goal,
              use_splitting=use_splitting,phantom_material=phantom_material,physics_list=physics_list,
              tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,detector_depth_mm=detector_depth_mm,
              detector_size_x_mm=detector_size_x_mm,detector_size_y_mm=detector_size_y_mm,
              detector_size_z_mm=detector_size_z_mm,source_phantom_shape=source_phantom_shape,
              source_phantom_rx=source_phantom_rx,source_phantom_ry=source_phantom_ry,
              source_phantom_rz=source_phantom_rz,source_phantom_material=source_phantom_material,
              source_phantom_ox=source_phantom_ox,source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if rc!=0: sys.exit(rc)

def run_all(n_primaries,output_dir,verbose=False,threads=1,max_jobs=1,source_type="nuclide",nuclide="F18",
            kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,write_dose=False,write_uncertainty=False,
            cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,vis=False,vis_type="vrml_file_only",
            unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,phantom_material="G4_WATER",
            detector_depth_mm=DEFAULT_DETECTOR_DEPTH_MM,detector_size_x_mm=DEFAULT_DETECTOR_XY_MM,detector_size_y_mm=DEFAULT_DETECTOR_XY_MM,detector_size_z_mm=DEFAULT_DETECTOR_Z_MM,
            physics_list="auto",tissue_cut_mm=0.01,barrier_cut_mm=0.1,
            source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
            source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    tasks=[(nuc,bar,float(t),0.0) for (nuc,bar),thicks in THICKNESS_SWEEPS.items() for t in thicks]
    for nuc in PHOTON_SPECTRA: tasks.append((nuc,"Air",0.0,0.0))
    failed=_run_tasks_parallel(tasks,n_primaries,output_dir,threads,max_jobs,verbose,
        source_type="nuclide",nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,physics_list=physics_list,
        tissue_cut_mm=tissue_cut_mm,barrier_cut_mm=barrier_cut_mm,
        detector_depth_mm=detector_depth_mm,detector_size_x_mm=detector_size_x_mm,
        detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,
        source_phantom_ry=source_phantom_ry,source_phantom_rz=source_phantom_rz,
        source_phantom_material=source_phantom_material,source_phantom_ox=source_phantom_ox,
        source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if failed: sys.exit(1)
    print("\n  ✓ All tasks complete.")

def _kv_default_thicknesses(barrier):
    return {"Lead":[0.5,1,1.5,2,2.5,3,3.5,4,5,6],"NWConcrete":[25,50,75,100,125,150,175,200,250,300],
            "LWConcrete":[35,70,105,140,175,210,260,310,380],"Steel":[3,6,9,12,16,20,25,30,38],
            "Glass":[20,40,60,80,100,125,155,190],"Gypsum":[50,100,150,200,260,330,410,510]}.get(barrier,[25,50,100,150,200,250])

def parse_args():
    ncpu=os.cpu_count() or 1
    p=argparse.ArgumentParser(description="GATE 10 shielding simulation",formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-type",default="nuclide",choices=["nuclide","xray"],dest="source_type")
    p.add_argument("--nuclide",default="F18",choices=sorted(PHOTON_SPECTRA.keys()))
    p.add_argument("--kvp",type=float,default=120.0); p.add_argument("--al-filter",type=float,default=2.5,dest="al_filter")
    p.add_argument("--cu-filter",type=float,default=0.0,dest="cu_filter"); p.add_argument("--kv-bins",type=int,default=128,dest="kv_bins")
    p.add_argument("--barrier",default="Lead",choices=list(BARRIER_MATERIAL_MAP.keys()))
    p.add_argument("--thickness",type=float,default=5.0); p.add_argument("--angle",type=float,default=0.0)
    p.add_argument("--angle-sweep",action="store_true",dest="angle_sweep")
    p.add_argument("--angles",type=float,nargs="+",default=None)
    p.add_argument("--phantom-material",default="G4_WATER",dest="phantom_material")
    p.add_argument("--physics-list",default="auto",choices=PHYSICS_LIST_CHOICES,dest="physics_list",
                   help=f"Geant4 EM physics list ('auto' = {DEFAULT_PHYSICS_LIST_XRAY} for xray sources, "
                        f"{DEFAULT_PHYSICS_LIST_NUCLIDE} otherwise).")
    p.add_argument("--tissue-cut-mm",type=float,default=0.01,dest="tissue_cut_mm",
                   help="Geant4 production cut (range threshold) for gamma/e-/e+ in TissuePhantom/"
                        "SplittingVolume/SourcePhantom and globally, mm (default 0.01 -- tightened from "
                        "the original 1.0mm hardcoded behavior; this forces more complete "
                        "secondary-electron and resultant bremsstrahlung-photon tracking, needed to "
                        "unlock Compton secondaries for low-energy isotope lines. Pass --tissue-cut-mm "
                        "1.0 to restore the old, looser cut if ever needed for an A/B comparison.)")
    p.add_argument("--barrier-cut-mm",type=float,default=0.1,dest="barrier_cut_mm",
                   help="Geant4 production cut for gamma/e-/e+ in the Barrier region, mm (default 0.1, "
                        "matches prior hardcoded behavior).")
    p.add_argument("--source-phantom-shape",default="none",choices=["none","sphere","ellipsoid"],dest="source_phantom_shape")
    p.add_argument("--source-phantom-rx",type=float,default=100.0,dest="source_phantom_rx")
    p.add_argument("--source-phantom-ry",type=float,default=70.0,dest="source_phantom_ry")
    p.add_argument("--source-phantom-rz",type=float,default=100.0,dest="source_phantom_rz")
    p.add_argument("--source-phantom-material",default="G4_WATER",dest="source_phantom_material")
    p.add_argument("--source-phantom-ox",type=float,default=0.0,dest="source_phantom_ox")
    p.add_argument("--source-phantom-oy",type=float,default=0.0,dest="source_phantom_oy")
    p.add_argument("--source-phantom-oz",type=float,default=0.0,dest="source_phantom_oz")
    # Detector defaults reproduce Oumano Sec 2.1 with ZERO flags: a slab centred
    # at 10 mm depth (planes 5-15 mm), spanning the full 2 m x 2 m tissue-block
    # face. --detector-preset/--detector-depth override the depth only; pass
    # --detector-centered to opt into the pre-fix legacy behaviour (a slab
    # centred in the phantom, ~240-260 mm depth) for backward comparisons.
    _det=p.add_mutually_exclusive_group()
    _det.add_argument("--detector-depth",type=float,default=DEFAULT_DETECTOR_DEPTH_MM,dest="detector_depth",metavar="MM",
                      help=f"Detector centre depth from tissue face, mm (default {DEFAULT_DETECTOR_DEPTH_MM:g} = Oumano's 1 cm point).")
    _det.add_argument("--detector-preset",default=None,dest="detector_preset",choices=["face","1cm","2cm","5cm","10cm"])
    _det.add_argument("--detector-centered",action="store_true",dest="detector_centered",
                      help="LEGACY: slab centred in the 500 mm phantom (~240-260 mm depth), NOT Oumano-matched. Pre-fix behaviour, kept for comparison.")
    p.add_argument("--detector-size-x",type=float,default=DEFAULT_DETECTOR_XY_MM,dest="detector_size_x",metavar="MM",
                   help=f"Detector X dimension mm (default {DEFAULT_DETECTOR_XY_MM:g} = full tissue-block face). Voxels = size / 2.5 mm.")
    p.add_argument("--detector-size-y",type=float,default=DEFAULT_DETECTOR_XY_MM,dest="detector_size_y",metavar="MM",
                   help=f"Detector Y dimension mm (default {DEFAULT_DETECTOR_XY_MM:g}).")
    p.add_argument("--detector-size-z",type=float,default=DEFAULT_DETECTOR_Z_MM,dest="detector_size_z",metavar="MM",
                   help=f"Detector Z dimension mm (default {DEFAULT_DETECTOR_Z_MM:g} -> 4 planes; arr[1:3] = Oumano's 2nd/3rd 5 mm plane). Voxels = size / 5.0 mm.")
    p.add_argument("--n",type=int,default=N_PRIMARIES); p.add_argument("--test",action="store_true")
    p.add_argument("--sweep",action="store_true"); p.add_argument("--nuclide-sweep",action="store_true",dest="nuclide_sweep")
    p.add_argument("--sweep-to-cvl",action="store_true",dest="sweep_to_cvl")
    p.add_argument("--reference",action="store_true"); p.add_argument("--all",action="store_true")
    p.add_argument("--threads",type=int,default=1); p.add_argument("--jobs",type=int,default=1)
    p.add_argument("--workers",type=int,default=1,help="Parallel worker processes splitting N primaries")
    p.add_argument("--seed",type=int,default=None,help="Base random seed for reproducibility")
    p.add_argument("--auto",action="store_true")
    p.add_argument("--output",default="output"); p.add_argument("--verbose",action="store_true")
    p.add_argument("--dose",action="store_true"); p.add_argument("--uncertainty",action="store_true")
    p.add_argument("--vis",action="store_true"); p.add_argument("--vis-type",default="vrml_file_only",choices=["vrml_file_only","vrml","qt"],dest="vis_type")
    p.add_argument("--no-cone",action="store_true",dest="no_cone")
    p.add_argument("--cone-angle-deg",type=float,default=DEFAULT_CONE_HALF_ANGLE_DEG,dest="cone_angle_deg")
    p.add_argument("--unc-goal",type=float,default=DEFAULT_UNC_GOAL,dest="unc_goal")
    p.add_argument("--split",action="store_true")
    p.add_argument("--show-spectrum",action="store_true",dest="show_spectrum")
    return p.parse_args()

def main():
    args=parse_args(); output_dir=Path(args.output); n=N_PRIMARIES_TEST if args.test else args.n
    if args.show_spectrum:
        if args.source_type!="xray": print("  ⚠  --show-spectrum requires --source-type xray"); sys.exit(1)
        plot_xray_spectrum(args.kvp,args.al_filter,args.cu_filter,output_dir,args.kv_bins); sys.exit(0)
    threads=args.threads; jobs=args.jobs; workers=args.workers

    if args.auto:
        # Compute n_tasks so auto config can balance jobs vs workers
        if args.all:
            n_tasks = sum(len(v) for v in THICKNESS_SWEEPS.values())
        elif args.sweep:
            n_tasks = len(THICKNESS_SWEEPS.get((args.nuclide if hasattr(args,"nuclide") else "F18", args.barrier), []))
            n_tasks = max(n_tasks, 1)
        elif args.angle_sweep:
            n_tasks = len(args.angles or ANGLE_SWEEP_DEG)
        elif args.nuclide_sweep:
            n_tasks = len(PHOTON_SPECTRA)
        elif args.reference:
            n_tasks = 1
        else:
            n_tasks = 1   # single run — all cores become workers
        threads, jobs, workers = _auto_config(n_tasks=n_tasks)

    if _ON_WINDOWS and threads>1: threads=1
    jobs=_safe_max_jobs(jobs)
    _PRESET_DEPTHS={"face":2.5,"1cm":10.0,"2cm":20.0,"5cm":50.0,"10cm":100.0}
    if args.detector_centered:
        args.detector_depth=None    # opt-in sentinel -> build_simulation's legacy centred-in-phantom branch
    elif args.detector_preset is not None:
        args.detector_depth=_PRESET_DEPTHS[args.detector_preset]
    # else: args.detector_depth already holds its default (Oumano's 10 mm) or an explicit --detector-depth value
    # Detector info
    det_parts = []
    if args.detector_depth is not None:
        oumano_tag = "  [Oumano default]" if args.detector_depth==DEFAULT_DETECTOR_DEPTH_MM and args.detector_preset is None else ""
        det_parts.append(f"depth={args.detector_depth:.1f} mm{oumano_tag}")
    else:
        det_parts.append("LEGACY: 4-voxel slab centred in phantom (~240-260 mm depth, NOT Oumano-matched)")
    det_parts.append(f"footprint {args.detector_size_x:.0f}×{args.detector_size_y:.0f}×{args.detector_size_z:.0f} mm")
    print(f"  ℹ  Detector: {', '.join(det_parts)}")

    source_label=_make_source_label(args.source_type,args.nuclide,args.kvp,args.al_filter,args.cu_filter)
    kw=_common_kw(args)
    opts=[]
    if not args.no_cone: opts.append(f"cone {args.cone_angle_deg:.1f}°")
    if args.unc_goal>0: opts.append(f"unc-stop {args.unc_goal:.1%}")
    if args.split: opts.append("splitting")
    if args.physics_list!="auto": opts.append(f"physics={args.physics_list}")
    if args.tissue_cut_mm!=0.01 or args.barrier_cut_mm!=0.1: opts.append(f"cuts=[tissue={args.tissue_cut_mm:g}mm, barrier={args.barrier_cut_mm:g}mm]")
    if opts: print(f"  ℹ  Active: {', '.join(opts)}")

    if args.all: run_all(n,output_dir,args.verbose,threads,jobs,**kw)
    elif args.angle_sweep:
        run_angle_sweep(source_label,args.barrier,args.thickness,n,output_dir,
                        angles=args.angles or ANGLE_SWEEP_DEG,verbose=args.verbose,threads=threads,max_jobs=jobs,**kw)
    elif args.nuclide_sweep:
        run_nuclide_sweep(source_label,n,output_dir,sweep_to_cvl=args.sweep_to_cvl,
                          angle_deg=args.angle,verbose=args.verbose,threads=threads,max_jobs=jobs,**kw)
    elif args.reference:
        if workers > 1:
            # NOTE: write_dose is NOT passed explicitly here - **kw (from _common_kw)
            # already carries write_dose=args.dose, i.e. whatever --dose actually
            # requested. Passing write_dose=True here too used to raise
            # "got multiple values for keyword argument 'write_dose'" any time
            # --workers > 1 was combined with --reference (e.g. via the GUI's
            # Auto config, which sizes workers from CPU count for any single-task
            # run) - and even before that, it silently forced dose-map output
            # regardless of --dose. Both are fixed by just not re-specifying it.
            wrc = _run_workers(source_label,"Air",0.0,n,output_dir,
                         n_workers=workers,base_seed=args.seed,
                         threads=threads,verbose=args.verbose,**kw)
            # _run_workers() returns 0=full success, 2=partial (some workers
            # failed but a degraded merge was still produced), 1=total
            # failure (nothing merged). Both nonzero cases must propagate to
            # the process exit code -- previously this call's return value
            # was silently discarded and main() always exited 0 here, which
            # is exactly why a partially-failed --workers run showed as
            # SLURM-COMPLETED with no merged output and no visible error.
            if wrc: sys.exit(wrc)
        else:
            run_reference(source_label,n,output_dir,verbose=args.verbose,threads=threads,**kw)
    elif args.sweep:
        # For sweeps, workers applies per-thickness simulation
        run_sweep(source_label,args.barrier,n,output_dir,angle_deg=args.angle,verbose=args.verbose,
                  threads=threads,max_jobs=jobs,n_workers=workers,base_seed=args.seed,**kw)
    else:
        if workers > 1:
            wrc = _run_workers(source_label,args.barrier,args.thickness,n,output_dir,
                         n_workers=workers,base_seed=args.seed,
                         threads=threads,verbose=args.verbose,**kw)
            # see the --reference branch above for why this check matters --
            # without it, a partial/failed merge silently exits 0
            if wrc: sys.exit(wrc)
        else:
            run_single(source_label,args.barrier,args.thickness,n,output_dir,angle_deg=args.angle,
                       verbose=args.verbose,threads=threads,seed=args.seed,**kw)

if __name__=="__main__":
    main()