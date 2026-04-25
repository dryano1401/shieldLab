"""
gateTurbo.py  —  GATE 10 (opengate) shielding simulation  [oblique + kV edition]
══════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import math
import os
import subprocess
import sys
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

_DOSE_ACTOR_HALF_DIAG_M = (0.250 / 2.0) * math.sqrt(2)
_SOURCE_TO_ACTOR_M      = 0.75 - (-1.0)
DEFAULT_CONE_HALF_ANGLE_DEG = math.degrees(
    math.atan(_DOSE_ACTOR_HALF_DIAG_M / _SOURCE_TO_ACTOR_M)
) * 1.20 * 4

_BARRIER_MFP_MM = {"Lead":5.0,"Steel":15.0,"NWConcrete":80.0,"LWConcrete":120.0,"Glass":60.0,"Gypsum":100.0,"Air":1e9}
DEFAULT_UNC_GOAL = 0.02
CVL_THRESHOLD    = 0.01
RAM_PER_JOB_GB   = 0.6
_ON_WINDOWS      = sys.platform.startswith("win")
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
    "Gypsum":{"density":2.33,"elements":{"H":0.0234,"O":0.55757,"S":0.186218,"Ca":0.23279}},
    "A514Steel":{"density":7.85,"elements":{"Fe":0.97000,"Mn":0.00950,"Cr":0.00650,"Si":0.00600,"Mo":0.00230,"C":0.004675,"Zr":0.00100,"B":0.000025}},
}
BARRIER_MATERIAL_MAP = {"Lead":"G4_Pb","LWConcrete":"LWConcrete","NWConcrete":"NWConcrete","Glass":"GlassNM","Gypsum":"Gypsum","Steel":"A514Steel","Air":"G4_AIR"}

PHOTON_SPECTRA = {
    "Lu177":[(0.05579,0.1105),(0.05765,0.1937),(0.06513,0.0472),(0.06701,0.0133),(0.11295,0.0623),(0.20837,0.1105)],
    "Tc99m":[(0.14051,0.8907)],"I131":[(0.08020,0.02620),(0.28430,0.06120),(0.36453,0.81200),(0.63699,0.07260),(0.72290,0.01770)],
    "F18":[(0.51100,1.93500)],"Zr89":[(0.51100,0.4550),(0.90915,0.9904),(1.65700,0.0010),(1.71300,0.0077),(1.74400,0.0013)],
    "Cu64":[(0.51100,0.3514),(1.34577,0.00473)],"Ga68":[(0.51100,1.7800),(1.07734,0.03220),(1.88316,0.00137)],
    "In111":[(0.02298,0.2410),(0.02317,0.4530),(0.02606,0.0392),(0.02610,0.0755),(0.02664,0.0194),(0.17128,0.9061),(0.24535,0.9408)],
    "I123":[(0.02720,0.2470),(0.02747,0.4560),(0.03094,0.0421),(0.03099,0.0811),(0.03170,0.0234),(0.15900,0.8326),(0.34436,0.0012),(0.44002,0.0039),(0.50533,0.0029),(0.52897,0.0127),(0.53854,0.3100)],
    "I124":[(0.02720,0.1660),(0.02747,0.3060),(0.03100,0.0544),(0.03170,0.0157),(0.30944,0.0282),(0.51100,0.4580),(0.59234,0.0011),(0.60273,0.6285),(0.60992,0.0015),(0.64585,0.0100),(0.72278,0.1013),(0.96819,0.0044),(0.97635,0.0010),(1.04511,0.0044),(1.05454,0.0012),(1.32552,0.0158),(1.36818,0.0030),(1.37609,0.0179),(1.48892,0.0021),(1.50950,0.0325),(1.63743,0.0021),(1.67560,0.0011),(1.69100,0.1115),(1.72000,0.0018),(1.85137,0.0022),(1.91856,0.0018),(2.03843,0.0036),(2.07867,0.0036),(2.09094,0.0062),(2.09881,0.0015),(2.14421,0.0011),(2.23203,0.0056),(2.28306,0.0053),(2.74690,0.0048)],
    "Rb82":[(0.51100,1.9040),(0.77652,0.1531),(1.39540,0.0014)],"Ac225":[(0.09979,0.0101),(0.21800,0.1164),(0.44045,0.2641)],
    "At211":[(0.07748,0.1168),(0.07929,0.1959),(0.08967,0.0572),(0.68700,0.00263)],"Y90":[(1.76070,0.0000159)],
    "Xe133":[(0.03086,0.0529),(0.03491,0.0145),(0.08099,0.3691)],
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
    ("F18","Lead"):[5,10,15,20,26,33,41,50,60,72,87],("F18","NWConcrete"):[50,100,150,200,250,305,370,450,555],
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
    avail=_available_ram_gb(); ram_cap=max(1,int(avail*0.80/RAM_PER_JOB_GB)); cpu_cap=os.cpu_count() or 1
    safe=min(requested,ram_cap,cpu_cap)
    if safe<requested: print(f"  ⚠  --jobs {requested} reduced to {safe}")
    return safe

def _auto_config():
    ncpu=os.cpu_count() or 1; avail=_available_ram_gb(); ram_cap=max(1,int(avail*0.80/RAM_PER_JOB_GB))
    if _ON_WINDOWS: return 1,min(ram_cap,ncpu)
    t=min(4,ncpu); j=min(ram_cap,max(1,ncpu//t)); return t,j

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
    phantom_material="G4_MUSCLE_SKELETAL_ICRP", detector_depth_mm=None,
    detector_size_x_mm=None, detector_size_y_mm=None, detector_size_z_mm=None,
    verbose=False, threads=1, write_dose=False, write_uncertainty=False,
    cone_source=True, cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
    vis=False, vis_type="vrml_file_only", unc_goal=DEFAULT_UNC_GOAL,
    use_splitting=False,
    source_phantom_shape="none", source_phantom_rx=100.0, source_phantom_ry=70.0,
    source_phantom_rz=100.0, source_phantom_material="G4_WATER",
    source_phantom_ox=0.0, source_phantom_oy=0.0, source_phantom_oz=0.0,
):
    output_dir.mkdir(parents=True,exist_ok=True)
    stem=_make_stem(source_label,barrier_name,thickness_mm,angle_deg)
    sim=gate.Simulation(); sim.g4_verbose=verbose; sim.visu=vis
    if vis:
        sim.visu_type=vis_type
        if vis_type=="vrml_file_only":
            sim.visu_filename=str(Path(output_dir)/"scene.wrl")
            sim.visu_commands=["/vis/open VRML2FILE","/vis/drawVolume world","/vis/viewer/set/viewpointThetaPhi 65 0","/vis/viewer/zoom 0.1","/vis/viewer/set/style surface","/vis/viewer/flush"]
    sim.random_seed="auto"; sim.output_dir=str(output_dir)
    if threads>1 and not _ON_WINDOWS: sim.number_of_threads=threads
    add_custom_materials(sim)

    world=sim.world; world.size=[2.0*m,2.0*m,3.0*m]; world.material="G4_AIR"

    actual_thickness_mm=max(thickness_mm,0.001)
    barrier=sim.add_volume("Box","Barrier"); barrier.size=[2.0*m,2.0*m,actual_thickness_mm*mm]
    barrier.translation=[0,0,0.77*cm]; barrier.material=BARRIER_MATERIAL_MAP[barrier_name]
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
    pm.physics_list_name="G4EmStandardPhysics_option4" if source_type=="xray" else "G4EmStandardPhysics_option3"
    pm.enable_decay=False
    pm.global_production_cuts.gamma=1.0*mm; pm.global_production_cuts.electron=1.0*mm; pm.global_production_cuts.positron=1.0*mm
    for p in ("gamma","electron","positron"): pm.set_production_cut("Barrier",p,0.1*mm)
    for region in ("TissuePhantom","SplittingVolume"):
        for p in ("gamma","electron","positron"): pm.set_production_cut(region,p,1.0*mm)
    if source_phantom_shape!="none":
        for p in ("gamma","electron","positron"): pm.set_production_cut("SourcePhantom",p,1.0*mm)

    _add_source(sim,source_type=source_type,nuclide=nuclide,n_primaries=n_primaries,
                cone_source=cone_source,cone_half_angle_deg=cone_half_angle_deg,
                kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,kv_bins=kv_bins,verbose=verbose)

    # ── DoseActor ─────────────────────────────────────────────────────────────
    PHANTOM_HALF_Z_MM=250.0; SPACING_XY=2.5*mm; SPACING_Z=5.0*mm
    dose=sim.add_actor("DoseActor","DoseActorTissue"); dose.attached_to="TissuePhantom"
    dose.output_filename=f"{stem}.mhd"; dose.hit_type="random"
    dose.edep.active=True; dose.edep_uncertainty.active=False
    dose.dose.active=write_dose; dose.dose_uncertainty.active=write_uncertainty

    if detector_depth_mm is None:
        sx_mm=detector_size_x_mm if detector_size_x_mm is not None else 250.0
        sy_mm=detector_size_y_mm if detector_size_y_mm is not None else 250.0
        sz_mm=detector_size_z_mm if detector_size_z_mm is not None else 20.0
        nx=max(1,round(sx_mm/(SPACING_XY/mm))); ny=max(1,round(sy_mm/(SPACING_XY/mm))); nz=max(1,round(sz_mm/(SPACING_Z/mm)))
        dose.size=[nx,ny,nz]; dose.spacing=[SPACING_XY,SPACING_XY,SPACING_Z]
        if verbose and (nx!=100 or ny!=100 or nz!=4):
            print(f"  ℹ  Detector: {nx}×{ny}×{nz} voxels ({sx_mm:.1f}×{sy_mm:.1f}×{sz_mm:.1f} mm), centred in phantom")
    else:
        depth_mm=max(float(detector_depth_mm),0.5)
        sx_mm=detector_size_x_mm if detector_size_x_mm is not None else 250.0
        sy_mm=detector_size_y_mm if detector_size_y_mm is not None else 250.0
        sz_mm=detector_size_z_mm if detector_size_z_mm is not None else 5.0
        nx=max(1,round(sx_mm/(SPACING_XY/mm))); ny=max(1,round(sy_mm/(SPACING_XY/mm))); nz=max(1,round(sz_mm/(SPACING_Z/mm)))
        dose.size=[nx,ny,nz]; dose.spacing=[SPACING_XY,SPACING_XY,SPACING_Z]
        local_z=(-PHANTOM_HALF_Z_MM+depth_mm)*mm; dose.translation=[0,0,local_z]
        if verbose:
            print(f"  ℹ  Detector: {nx}×{ny}×{nz} voxels ({sx_mm:.1f}×{sy_mm:.1f}×{sz_mm:.1f} mm) at {depth_mm:.1f} mm depth")

    if unc_goal>0.0:
        try:
            dose.edep.uncertainty_goal=unc_goal
            dose.edep.uncertainty_first_check_after_n_events=max(100_000,n_primaries//100)
            dose.edep.uncertainty_check_every_n_events=1_000_000
        except AttributeError: pass

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

def _print_geometry(sim,angle_deg=0.0):
    bar=sim.volume_manager.volumes["Barrier"]; tis=sim.volume_manager.volumes["TissuePhantom"]
    src=sim.source_manager.sources["PointSource"]; dose=sim.actor_manager.actors["DoseActorTissue"]
    t_phys=bar.size[2]/mm
    t_eff=t_phys/math.cos(math.radians(angle_deg)) if abs(angle_deg)>0.01 else t_phys
    eff_str=f"  (t_eff = {t_eff:.2f} mm at {angle_deg:.0f}°)" if abs(angle_deg)>0.01 else ""
    if src.energy.type=="mono": e_str=f"{src.energy.mono/MeV*1000:.1f} keV (mono)"
    else:
        ek=[round(e/MeV*1000,1) for e in src.energy.spectrum_energies]
        e_str=f"{ek} keV" if len(ek)<=6 else f"{ek[0]:.1f}–{ek[-1]:.1f} keV [{len(ek)} bins]"
    theta_max=src.direction.theta[1]/deg
    cone_str=f"cone {theta_max:.1f}°" if theta_max<45 else f"hemisphere {theta_max:.0f}°"
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
    print(f"  │ Physics  : {sim.physics_manager.physics_list_name}  [tissue cuts: 1.0 mm, barrier cuts: 0.1 mm]")
    vols=sim.volume_manager.volumes
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
               phantom_material="G4_MUSCLE_SKELETAL_ICRP",detector_depth_mm=None,
               detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
               source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
               source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    sim=build_simulation(source_label=source_label,barrier_name=barrier,thickness_mm=thickness_mm,n_primaries=n_primaries,
        output_dir=output_dir,source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,
        cu_filter_mm=cu_filter_mm,kv_bins=kv_bins,angle_deg=angle_deg,verbose=verbose,threads=threads,
        write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        cone_half_angle_deg=cone_half_angle_deg,vis=vis,vis_type=vis_type,unc_goal=unc_goal,use_splitting=use_splitting,
        phantom_material=phantom_material,detector_depth_mm=detector_depth_mm,
        detector_size_x_mm=detector_size_x_mm,detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
        source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,source_phantom_ry=source_phantom_ry,
        source_phantom_rz=source_phantom_rz,source_phantom_material=source_phantom_material,
        source_phantom_ox=source_phantom_ox,source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    _print_geometry(sim,angle_deg); sim.run()

def _spawn(source_label,barrier,thickness_mm,n_primaries,output_dir,
           source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,
           angle_deg=0.0,threads=1,verbose=False,write_dose=False,write_uncertainty=False,
           cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
           vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
           phantom_material="G4_MUSCLE_SKELETAL_ICRP",detector_depth_mm=None,
           detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
           source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
           source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    cmd=[sys.executable,__file__,"--source-type",source_type,"--nuclide",nuclide,"--barrier",barrier,
         "--thickness",str(thickness_mm),"--n",str(n_primaries),"--output",str(output_dir),
         "--threads",str(threads),"--unc-goal",str(unc_goal),"--angle",str(angle_deg),
         "--kvp",str(kvp),"--al-filter",str(al_filter_mm),"--cu-filter",str(cu_filter_mm),
         "--kv-bins",str(kv_bins),"--cone-angle-deg",str(cone_half_angle_deg),
         "--phantom-material",phantom_material,
         "--source-phantom-shape",source_phantom_shape,"--source-phantom-rx",str(source_phantom_rx),
         "--source-phantom-ry",str(source_phantom_ry),"--source-phantom-rz",str(source_phantom_rz),
         "--source-phantom-material",source_phantom_material,
         "--source-phantom-ox",str(source_phantom_ox),"--source-phantom-oy",str(source_phantom_oy),
         "--source-phantom-oz",str(source_phantom_oz)]
    if detector_depth_mm is not None: cmd.extend(["--detector-depth",str(detector_depth_mm)])
    if detector_size_x_mm is not None: cmd.extend(["--detector-size-x",str(detector_size_x_mm)])
    if detector_size_y_mm is not None: cmd.extend(["--detector-size-y",str(detector_size_y_mm)])
    if detector_size_z_mm is not None: cmd.extend(["--detector-size-z",str(detector_size_z_mm)])
    if verbose: cmd.append("--verbose")
    if write_dose: cmd.append("--dose")
    if write_uncertainty: cmd.append("--uncertainty")
    if not cone_source: cmd.append("--no-cone")
    if use_splitting: cmd.append("--split")
    if vis: cmd.extend(["--vis","--vis-type",vis_type])
    label=f"{source_label}/{barrier}/{thickness_mm} mm a={angle_deg:.0f}° N={n_primaries:,}"
    print(f"\n  ▶  {label}")
    result=subprocess.run(cmd,capture_output=(not verbose))
    if result.returncode==0: print(f"  ✓  {label}")
    else:
        print(f"  ✗  {label}  (exit {result.returncode})")
        if not verbose and result.stderr:
            for line in result.stderr.decode(errors="replace").strip().splitlines()[-8:]: print(f"     {line}")
    return result.returncode

def _run_tasks_parallel(tasks,n_primaries,output_dir,threads=1,max_jobs=1,verbose=False,
                        source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,
                        kv_bins=128,write_dose=False,write_uncertainty=False,cone_source=True,
                        unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
                        phantom_material="G4_MUSCLE_SKELETAL_ICRP",detector_depth_mm=None,
                        detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
                        source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,
                        source_phantom_rz=100.0,source_phantom_material="G4_WATER",
                        source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    kw=dict(source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
            kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
            unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
            detector_depth_mm=detector_depth_mm,detector_size_x_mm=detector_size_x_mm,
            detector_size_y_mm=detector_size_y_mm,detector_size_z_mm=detector_size_z_mm,
            source_phantom_shape=source_phantom_shape,source_phantom_rx=source_phantom_rx,
            source_phantom_ry=source_phantom_ry,source_phantom_rz=source_phantom_rz,
            source_phantom_material=source_phantom_material,
            source_phantom_ox=source_phantom_ox,source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if max_jobs==1:
        failed=[]
        for label,bar,t,ang in tasks:
            rc=_spawn(label,bar,t,n_primaries,output_dir,angle_deg=ang,threads=threads,verbose=verbose,**kw)
            if rc!=0: failed.append((label,bar,t,ang,rc))
        return failed
    print(f"\n  Parallel pool: {max_jobs} jobs × {threads} threads")
    futures={}; failed=[]
    with ThreadPoolExecutor(max_workers=max_jobs) as pool:
        for label,bar,t,ang in tasks:
            fut=pool.submit(_spawn,label,bar,t,n_primaries,output_dir,angle_deg=ang,threads=threads,verbose=verbose,**kw)
            futures[fut]=(label,bar,t,ang)
        for fut in as_completed(futures):
            label,bar,t,ang=futures[fut]
            try:
                rc=fut.result()
                if rc!=0: failed.append((label,bar,t,ang,rc))
            except Exception as exc: print(f"  ✗  {label}/{bar}/{t}mm raised {exc}"); failed.append((label,bar,t,ang,-1))
    return failed

def _common_kw(args):
    return dict(source_type=args.source_type,nuclide=args.nuclide,kvp=args.kvp,al_filter_mm=args.al_filter,
                cu_filter_mm=args.cu_filter,kv_bins=args.kv_bins,write_dose=args.dose,write_uncertainty=args.uncertainty,
                cone_source=not args.no_cone,cone_half_angle_deg=args.cone_angle_deg,vis=args.vis,vis_type=args.vis_type,
                unc_goal=args.unc_goal,use_splitting=args.split,phantom_material=args.phantom_material,
                detector_depth_mm=args.detector_depth,
                detector_size_x_mm=args.detector_size_x,detector_size_y_mm=args.detector_size_y,
                detector_size_z_mm=args.detector_size_z,
                source_phantom_shape=args.source_phantom_shape,source_phantom_rx=args.source_phantom_rx,
                source_phantom_ry=args.source_phantom_ry,source_phantom_rz=args.source_phantom_rz,
                source_phantom_material=args.source_phantom_material,
                source_phantom_ox=args.source_phantom_ox,source_phantom_oy=args.source_phantom_oy,
                source_phantom_oz=args.source_phantom_oz)

def run_sweep(source_label,barrier,n_primaries,output_dir,angle_deg=0.0,verbose=False,threads=1,max_jobs=1,
              source_type="nuclide",nuclide="F18",kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,
              write_dose=False,write_uncertainty=False,cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,
              vis=False,vis_type="vrml_file_only",unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,
              phantom_material="G4_MUSCLE_SKELETAL_ICRP",detector_depth_mm=None,
              detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
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
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
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
                    phantom_material="G4_MUSCLE_SKELETAL_ICRP",detector_depth_mm=None,
                    detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
                    source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
                    source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    if angles is None: angles=ANGLE_SWEEP_DEG
    tasks=[(source_label,barrier,float(thickness_mm),float(a)) for a in angles]
    if not _reference_exists(source_label,output_dir): tasks.append((source_label,"Air",0.0,0.0))
    failed=_run_tasks_parallel(tasks,n_primaries,output_dir,threads,max_jobs,verbose,
        source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
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
                      phantom_material="G4_MUSCLE_SKELETAL_ICRP",detector_depth_mm=None,
                      detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
                      source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
                      source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    spawn_kw=dict(source_type=source_type,nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,angle_deg=angle_deg,threads=threads,verbose=verbose,write_dose=write_dose,
        write_uncertainty=write_uncertainty,cone_source=cone_source,cone_half_angle_deg=cone_half_angle_deg,
        vis=vis,vis_type=vis_type,unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
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
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
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
                  unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,phantom_material="G4_MUSCLE_SKELETAL_ICRP",
                  detector_depth_mm=None,detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
                  source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
                  source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    rc=_spawn(source_label,"Air",0.0,n_primaries,output_dir,source_type=source_type,nuclide=nuclide,kvp=kvp,
              al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,kv_bins=kv_bins,angle_deg=0.0,threads=threads,
              verbose=verbose,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
              cone_half_angle_deg=cone_half_angle_deg,vis=vis,vis_type=vis_type,unc_goal=unc_goal,
              use_splitting=use_splitting,phantom_material=phantom_material,detector_depth_mm=detector_depth_mm,
              detector_size_x_mm=detector_size_x_mm,detector_size_y_mm=detector_size_y_mm,
              detector_size_z_mm=detector_size_z_mm,source_phantom_shape=source_phantom_shape,
              source_phantom_rx=source_phantom_rx,source_phantom_ry=source_phantom_ry,
              source_phantom_rz=source_phantom_rz,source_phantom_material=source_phantom_material,
              source_phantom_ox=source_phantom_ox,source_phantom_oy=source_phantom_oy,source_phantom_oz=source_phantom_oz)
    if rc!=0: sys.exit(rc)

def run_all(n_primaries,output_dir,verbose=False,threads=1,max_jobs=1,source_type="nuclide",nuclide="F18",
            kvp=120.0,al_filter_mm=2.5,cu_filter_mm=0.0,kv_bins=128,write_dose=False,write_uncertainty=False,
            cone_source=True,cone_half_angle_deg=DEFAULT_CONE_HALF_ANGLE_DEG,vis=False,vis_type="vrml_file_only",
            unc_goal=DEFAULT_UNC_GOAL,use_splitting=False,phantom_material="G4_MUSCLE_SKELETAL_ICRP",
            detector_depth_mm=None,detector_size_x_mm=None,detector_size_y_mm=None,detector_size_z_mm=None,
            source_phantom_shape="none",source_phantom_rx=100.0,source_phantom_ry=70.0,source_phantom_rz=100.0,
            source_phantom_material="G4_WATER",source_phantom_ox=0.0,source_phantom_oy=0.0,source_phantom_oz=0.0):
    tasks=[(nuc,bar,float(t),0.0) for (nuc,bar),thicks in THICKNESS_SWEEPS.items() for t in thicks]
    for nuc in PHOTON_SPECTRA: tasks.append((nuc,"Air",0.0,0.0))
    failed=_run_tasks_parallel(tasks,n_primaries,output_dir,threads,max_jobs,verbose,
        source_type="nuclide",nuclide=nuclide,kvp=kvp,al_filter_mm=al_filter_mm,cu_filter_mm=cu_filter_mm,
        kv_bins=kv_bins,write_dose=write_dose,write_uncertainty=write_uncertainty,cone_source=cone_source,
        unc_goal=unc_goal,use_splitting=use_splitting,phantom_material=phantom_material,
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
    ncpu=os.cpu_count() or 1; auto_t,auto_j=_auto_config()
    p=argparse.ArgumentParser(description="GATE 10 shielding simulation",formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-type",default="nuclide",choices=["nuclide","xray"],dest="source_type")
    p.add_argument("--nuclide",default="F18",choices=sorted(PHOTON_SPECTRA.keys()))
    p.add_argument("--kvp",type=float,default=120.0); p.add_argument("--al-filter",type=float,default=2.5,dest="al_filter")
    p.add_argument("--cu-filter",type=float,default=0.0,dest="cu_filter"); p.add_argument("--kv-bins",type=int,default=128,dest="kv_bins")
    p.add_argument("--barrier",default="Lead",choices=list(BARRIER_MATERIAL_MAP.keys()))
    p.add_argument("--thickness",type=float,default=5.0); p.add_argument("--angle",type=float,default=0.0)
    p.add_argument("--angle-sweep",action="store_true",dest="angle_sweep")
    p.add_argument("--angles",type=float,nargs="+",default=None)
    p.add_argument("--phantom-material",default="G4_MUSCLE_SKELETAL_ICRP",dest="phantom_material")
    p.add_argument("--source-phantom-shape",default="none",choices=["none","sphere","ellipsoid"],dest="source_phantom_shape")
    p.add_argument("--source-phantom-rx",type=float,default=100.0,dest="source_phantom_rx")
    p.add_argument("--source-phantom-ry",type=float,default=70.0,dest="source_phantom_ry")
    p.add_argument("--source-phantom-rz",type=float,default=100.0,dest="source_phantom_rz")
    p.add_argument("--source-phantom-material",default="G4_WATER",dest="source_phantom_material")
    p.add_argument("--source-phantom-ox",type=float,default=0.0,dest="source_phantom_ox")
    p.add_argument("--source-phantom-oy",type=float,default=0.0,dest="source_phantom_oy")
    p.add_argument("--source-phantom-oz",type=float,default=0.0,dest="source_phantom_oz")
    _det=p.add_mutually_exclusive_group()
    _det.add_argument("--detector-depth",type=float,default=None,dest="detector_depth",metavar="MM")
    _det.add_argument("--detector-preset",default=None,dest="detector_preset",choices=["face","1cm","2cm","5cm","10cm"])
    p.add_argument("--detector-size-x",type=float,default=None,dest="detector_size_x",metavar="MM",
                   help="Detector X dimension mm (default 250). Voxels = size / 2.5 mm.")
    p.add_argument("--detector-size-y",type=float,default=None,dest="detector_size_y",metavar="MM",
                   help="Detector Y dimension mm (default 250).")
    p.add_argument("--detector-size-z",type=float,default=None,dest="detector_size_z",metavar="MM",
                   help="Detector Z dimension mm (default 20 or 5). Voxels = size / 5.0 mm.")
    p.add_argument("--n",type=int,default=N_PRIMARIES); p.add_argument("--test",action="store_true")
    p.add_argument("--sweep",action="store_true"); p.add_argument("--nuclide-sweep",action="store_true",dest="nuclide_sweep")
    p.add_argument("--sweep-to-cvl",action="store_true",dest="sweep_to_cvl")
    p.add_argument("--reference",action="store_true"); p.add_argument("--all",action="store_true")
    p.add_argument("--threads",type=int,default=1); p.add_argument("--jobs",type=int,default=1)
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
    threads=args.threads; jobs=args.jobs
    if args.auto: threads,jobs=_auto_config()
    if _ON_WINDOWS and threads>1: threads=1
    jobs=_safe_max_jobs(jobs)
    _PRESET_DEPTHS={"face":2.5,"1cm":10.0,"2cm":20.0,"5cm":50.0,"10cm":100.0}
    if args.detector_preset is not None: args.detector_depth=_PRESET_DEPTHS[args.detector_preset]
    # Detector info
    det_parts = []
    if args.detector_depth is not None: det_parts.append(f"depth={args.detector_depth:.1f} mm")
    else: det_parts.append("4-voxel slab centred in phantom")
    if args.detector_size_x is not None: det_parts.append(f"X={args.detector_size_x:.0f} mm")
    if args.detector_size_y is not None: det_parts.append(f"Y={args.detector_size_y:.0f} mm")
    if args.detector_size_z is not None: det_parts.append(f"Z={args.detector_size_z:.0f} mm")
    print(f"  ℹ  Detector: {', '.join(det_parts)}")

    source_label=_make_source_label(args.source_type,args.nuclide,args.kvp,args.al_filter,args.cu_filter)
    kw=_common_kw(args)
    opts=[]
    if not args.no_cone: opts.append(f"cone {args.cone_angle_deg:.1f}°")
    if args.unc_goal>0: opts.append(f"unc-stop {args.unc_goal:.1%}")
    if args.split: opts.append("splitting")
    if opts: print(f"  ℹ  Active: {', '.join(opts)}")

    if args.all: run_all(n,output_dir,args.verbose,threads,jobs,**kw)
    elif args.angle_sweep:
        run_angle_sweep(source_label,args.barrier,args.thickness,n,output_dir,
                        angles=args.angles or ANGLE_SWEEP_DEG,verbose=args.verbose,threads=threads,max_jobs=jobs,**kw)
    elif args.nuclide_sweep:
        run_nuclide_sweep(source_label,n,output_dir,sweep_to_cvl=args.sweep_to_cvl,
                          angle_deg=args.angle,verbose=args.verbose,threads=threads,max_jobs=jobs,**kw)
    elif args.reference: run_reference(source_label,n,output_dir,verbose=args.verbose,threads=threads,**kw)
    elif args.sweep:
        run_sweep(source_label,args.barrier,n,output_dir,angle_deg=args.angle,verbose=args.verbose,
                  threads=threads,max_jobs=jobs,**kw)
    else:
        run_single(source_label,args.barrier,args.thickness,n,output_dir,angle_deg=args.angle,
                   verbose=args.verbose,threads=threads,**kw)

if __name__=="__main__":
    main()