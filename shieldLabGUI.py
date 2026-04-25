"""
shieldLabGUI.py  —  GUI launcher for shieldLabSim.py
"""

import os
import sys
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path

try:
    import vtk
    import vtkmodules.vtkRenderingOpenGL2
    import vtkmodules.vtkInteractionStyle
    from PIL import Image, ImageTk
    VTK_OK = True
except Exception:
    VTK_OK = False

try:
    import SimpleITK as sitk
    import numpy as np
    SITK_OK = True
except ImportError:
    SITK_OK = False

try:
    import shieldLabAnalyze as ai
    AI_OK = True
except ImportError:
    AI_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MPL_OK = True
except ImportError:
    MPL_OK = False

SCRIPT_DIR = Path(__file__).parent
SIM_SCRIPT = SCRIPT_DIR / "shieldLabSim.py"

NUCLIDES = ["F18","Tc99m","I131","Lu177","Zr89","Cu64","Ga68","In111","I123","I124","Rb82","Ac225","At211","Y90","Xe133"]
BARRIERS = ["Lead","Steel","NWConcrete","LWConcrete","Glass","Gypsum"]
PHANTOM_DISPLAY = [
    "G4_MUSCLE_SKELETAL_ICRP  (skeletal muscle)","G4_WATER                 (liquid water)",
    "G4_TISSUE_SOFT_ICRP      (soft tissue)","G4_BONE_CORTICAL_ICRP    (cortical bone)",
    "G4_BONE_COMPACT_ICRU     (compact bone)","G4_LUNG_ICRP             (lung)",
    "G4_BRAIN_ICRP            (brain)","G4_ADIPOSE_TISSUE_ICRP   (adipose/fat)","G4_SKIN_ICRP             (skin)",
]
PHANTOM_VALUES = [
    "G4_MUSCLE_SKELETAL_ICRP","G4_WATER","G4_TISSUE_SOFT_ICRP","G4_BONE_CORTICAL_ICRP",
    "G4_BONE_COMPACT_ICRU","G4_LUNG_ICRP","G4_BRAIN_ICRP","G4_ADIPOSE_TISSUE_ICRP","G4_SKIN_ICRP",
]
DETECTOR_OPTIONS = [
    ("Original — 4-voxel slab centred in phantom (~250 mm)","original"),
    ("Entrance face  (2.5 mm from front surface)","face"),("1 cm depth","1cm"),("2 cm depth","2cm"),
    ("5 cm depth","5cm"),("10 cm depth","10cm"),("Custom depth (specify below)","custom"),
]
RUN_MODES = [
    ("Single point","single"),("Thickness sweep","sweep"),("Angle sweep","angle_sweep"),
    ("Nuclide / source sweep","nuclide_sweep"),("Air reference only","reference"),("Run ALL","all"),
]
VIS_TYPES = ["vrml_file_only","vrml","qt"]
MODES_NO_BARRIER = {"nuclide_sweep","reference"}

BG="#1c2128"; BG2="#22272e"; BG3="#2d333b"; BORDER="#444c56"; ACCENT="#58a6ff"
FG="#cdd9e5"; FG2="#8b949e"; GREEN="#3fb950"; RED="#f85149"; YELLOW="#d29922"
_W = sys.platform.startswith("win")
F_UI=("Segoe UI",10) if _W else ("Helvetica",10)
F_BOLD=("Segoe UI",10,"bold") if _W else ("Helvetica",10,"bold")
F_TITLE=("Segoe UI",13,"bold") if _W else ("Helvetica",13,"bold")
F_MONO=("Consolas",10) if _W else ("Menlo",10)
F_SMALL=("Segoe UI",9) if _W else ("Helvetica",9)

def mk_entry(parent,var,w=12):
    return tk.Entry(parent,textvariable=var,width=w,bg=BG3,fg=FG,insertbackground=FG,relief="flat",font=F_UI,bd=4,highlightthickness=1,highlightbackground=BORDER,highlightcolor=ACCENT)
def mk_combo(parent,var,values,w=28):
    return ttk.Combobox(parent,textvariable=var,values=values,state="readonly",style="G.TCombobox",width=w)
def mk_lbl(parent,text,fg=FG2,font=None,**kw):
    return tk.Label(parent,text=text,bg=BG2,fg=fg,font=font or F_UI,**kw)
def mk_check(parent,var,text,cmd=None):
    return tk.Checkbutton(parent,text=text,variable=var,bg=BG2,fg=FG,selectcolor=BG3,activebackground=BG2,activeforeground=FG,font=F_UI,bd=0,command=cmd)
def mk_radio(parent,text,val,var,cmd=None):
    return tk.Radiobutton(parent,text=text,value=val,variable=var,bg=BG2,fg=FG,selectcolor=BG3,activebackground=BG2,activeforeground=FG,font=F_UI,bd=0,command=cmd)
def mk_btn(parent,text,cmd,bg=BG3,fg=FG,font=None,**kw):
    return tk.Button(parent,text=text,command=cmd,bg=bg,fg=fg,relief="flat",activebackground=BORDER,activeforeground=FG,font=font or F_UI,cursor="hand2",bd=0,**kw)
def grid_row(parent,r,label_text,widget,hint=""):
    mk_lbl(parent,label_text,anchor="w").grid(row=r,column=0,sticky="w",padx=(8,4),pady=3)
    widget.grid(row=r,column=1,sticky="ew",padx=(0,4),pady=3)
    if hint: mk_lbl(parent,hint,fg=FG2,font=F_SMALL).grid(row=r,column=2,sticky="w",padx=(0,8),pady=3)


class VRMLViewerWindow(tk.Toplevel):
    _BG_DARK=(0.13,0.14,0.17); _BG_LIGHT=(0.93,0.93,0.93); _W=1024; _H=720
    def __init__(self,parent,vrml_path):
        super().__init__(parent); self._path=Path(vrml_path); self._bg=self._BG_DARK; self._wire=False
        self._ren=None; self._rw=None; self._iren=None; self._tk_img=None
        self.title(f"VRML Viewer — {self._path.name}"); self.geometry(f"{self._W}x{self._H+44}")
        self.configure(bg=BG); self.resizable(False,False); self.protocol("WM_DELETE_WINDOW",self._on_close)
        self._build_toolbar(); self._build_canvas(); self._setup_vtk(); self._load_vrml()
    def _build_toolbar(self):
        bar=tk.Frame(self,bg="#161b22",pady=4); bar.pack(fill="x",side="top")
        tk.Label(bar,text=f"⚛  {self._path.name}",font=F_BOLD,bg="#161b22",fg=ACCENT,padx=12).pack(side="left")
        for t,c in [("⟳ Reset",self._reset_camera),("⬡ Wire",self._toggle_wire),("◑ BG",self._toggle_bg),("📷 Save",self._screenshot)]:
            mk_btn(bar,t,c,fg=FG2,padx=8,pady=3).pack(side="right",padx=(0,4))
    def _build_canvas(self):
        self._canvas=tk.Canvas(self,width=self._W,height=self._H,bg="black",highlightthickness=0,cursor="fleur"); self._canvas.pack(fill="both",expand=True)
        for ev,fn in [("<ButtonPress-1>",self._on_lbp),("<B1-Motion>",self._on_lbd),("<ButtonRelease-1>",self._on_lbr),
                      ("<ButtonPress-2>",self._on_mbp),("<B2-Motion>",self._on_mbd),("<ButtonRelease-2>",self._on_mbr),
                      ("<ButtonPress-3>",self._on_rbp),("<B3-Motion>",self._on_rbd),("<ButtonRelease-3>",self._on_rbr),
                      ("<Shift-B1-Motion>",self._on_slbd),("<MouseWheel>",self._on_wh),("<Button-4>",self._on_whu),
                      ("<Button-5>",self._on_whd),("<KeyPress>",self._on_key)]:
            self._canvas.bind(ev,fn)
        self._canvas.focus_set()
    def _setup_vtk(self):
        self._rw=vtk.vtkRenderWindow(); self._rw.SetOffScreenRendering(1); self._rw.SetSize(self._W,self._H); self._rw.SetMultiSamples(0)
        self._ren=vtk.vtkRenderer(); self._ren.SetBackground(*self._bg); self._rw.AddRenderer(self._ren)
        self._iren=vtk.vtkGenericRenderWindowInteractor(); self._iren.SetRenderWindow(self._rw); self._iren.SetSize(self._W,self._H)
        self._iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self._w2i=vtk.vtkWindowToImageFilter(); self._w2i.SetInput(self._rw); self._w2i.SetInputBufferTypeToRGB(); self._w2i.ReadFrontBufferOff()
        self._blitting=False; self._canvas_img_id=None; self._iren_init=False
    def _load_vrml(self):
        imp=vtk.vtkVRMLImporter(); imp.SetFileName(str(self._path)); imp.SetRenderWindow(self._rw); imp.Update()
        self._iren.GetInteractorStyle().SetDefaultRenderer(self._ren); self._ren.SetBackground(*self._bg)
        self._ren.ResetCamera(); c=self._ren.GetActiveCamera(); c.Elevation(20); c.Azimuth(30); self._ren.ResetCameraClippingRange()
        self.after(50,self._rab)
    def _rab(self):
        if self._rw is None or self._blitting: return
        self._blitting=True
        try:
            self._rw.Render()
            if not self._iren_init: self._iren.Initialize(); self._iren_init=True
            self._blit()
        except: import traceback; self._show_error(traceback.format_exc())
        finally: self._blitting=False
    def _blit(self):
        from vtkmodules.util.numpy_support import vtk_to_numpy
        self._w2i.Modified(); self._w2i.Update(); d=self._w2i.GetOutput(); w,h,_=d.GetDimensions()
        s=d.GetPointData().GetScalars()
        if s is None or s.GetNumberOfTuples()==0: raise RuntimeError("Empty")
        arr=vtk_to_numpy(s).reshape(h,w,3)[::-1]; self._tk_img=ImageTk.PhotoImage(Image.fromarray(arr.astype("uint8"),"RGB"))
        if self._canvas_img_id is None: self._canvas_img_id=self._canvas.create_image(0,0,anchor="nw",image=self._tk_img)
        else: self._canvas.itemconfig(self._canvas_img_id,image=self._tk_img)
    def _show_error(self,msg): self._canvas.delete("all"); self._canvas_img_id=None; self._canvas.create_text(self._W//2,self._H//2,text=msg,fill=RED,font=F_MONO,justify="left",width=self._W-40)
    def _ev(self,e,ctrl=0,shift=0): self._iren.SetEventInformationFlipY(e.x,e.y,ctrl,shift,chr(0),0,None)
    def _ia(self): self._iren.MouseMoveEvent(); self._rab()
    def _on_lbp(self,e): self._ev(e); self._iren.LeftButtonPressEvent()
    def _on_lbd(self,e): self._ev(e); self._ia()
    def _on_lbr(self,e): self._ev(e); self._iren.LeftButtonReleaseEvent()
    def _on_slbd(self,e): self._ev(e,shift=1); self._ia()
    def _on_mbp(self,e): self._ev(e); self._iren.MiddleButtonPressEvent()
    def _on_mbd(self,e): self._ev(e); self._ia()
    def _on_mbr(self,e): self._ev(e); self._iren.MiddleButtonReleaseEvent()
    def _on_rbp(self,e): self._ev(e); self._iren.RightButtonPressEvent()
    def _on_rbd(self,e): self._ev(e); self._ia()
    def _on_rbr(self,e): self._ev(e); self._iren.RightButtonReleaseEvent()
    def _on_wh(self,e):
        self._ev(e)
        if e.delta>0: self._iren.MouseWheelForwardEvent()
        else: self._iren.MouseWheelBackwardEvent()
        self._rab()
    def _on_whu(self,e): self._ev(e); self._iren.MouseWheelForwardEvent(); self._rab()
    def _on_whd(self,e): self._ev(e); self._iren.MouseWheelBackwardEvent(); self._rab()
    def _on_key(self,e):
        k=e.char if e.char else chr(0); self._iren.SetEventInformationFlipY(e.x,e.y,0,0,k,0,e.keysym)
        self._iren.KeyPressEvent(); self._iren.CharEvent(); self._rab()
    def _reset_camera(self):
        if self._ren: self._ren.ResetCamera(); self._ren.ResetCameraClippingRange(); self._rab()
    def _toggle_wire(self):
        if not self._ren: return
        self._wire=not self._wire; actors=self._ren.GetActors(); actors.InitTraversal(); a=actors.GetNextItem()
        while a:
            if self._wire: a.GetProperty().SetRepresentationToWireframe()
            else: a.GetProperty().SetRepresentationToSurface()
            a=actors.GetNextItem()
        self._rab()
    def _toggle_bg(self):
        if not self._ren: return
        self._bg=self._BG_LIGHT if self._bg==self._BG_DARK else self._BG_DARK; self._ren.SetBackground(*self._bg); self._rab()
    def _screenshot(self):
        if not self._rw: return
        self._w2i.Modified(); self._w2i.Update(); d=self._w2i.GetOutput(); w,h,_=d.GetDimensions()
        from vtkmodules.util.numpy_support import vtk_to_numpy
        s=d.GetPointData().GetScalars()
        if s is None: return
        arr=vtk_to_numpy(s).reshape(h,w,3)[::-1]; p=Image.fromarray(arr.astype("uint8"),"RGB").resize((w*2,h*2),Image.LANCZOS)
        out=self._path.with_suffix(".png"); p.save(str(out)); messagebox.showinfo("Saved",f"{out}",parent=self)
    def _on_close(self):
        try:
            if self._rw: self._rw.Finalize()
        except: pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# MHD Viewer Window — visualise dose/edep/uncertainty MHD files
# ═══════════════════════════════════════════════════════════════════════════════

class MHDViewerWindow(tk.Toplevel):
    """
    Standalone window to load, visualise and quantify MHD image files.
    Requires SimpleITK + matplotlib.
    """
    CMAPS = ["hot","viridis","inferno","plasma","magma","jet","gray","coolwarm"]
    PROJECTIONS = ["Single slice","Max projection","Sum projection","Mean projection"]

    def __init__(self, parent):
        super().__init__(parent)
        self.title("MHD Viewer — Dose / Edep / Uncertainty")
        self.geometry("900x720")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._arr = None
        self._spacing = None
        self._origin = None
        self._filepath = None

        self._build_ui()

        # Force window to front
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))

    def _build_ui(self):
        # ── Top toolbar ───────────────────────────────────────────────────────
        bar = tk.Frame(self, bg="#161b22", pady=4)
        bar.pack(fill="x", side="top")
        tk.Label(bar, text="🔬  MHD Viewer", font=F_BOLD, bg="#161b22", fg=ACCENT, padx=12).pack(side="left")
        mk_btn(bar, "📂  Load MHD", self._load_file, fg=ACCENT, padx=10, pady=3).pack(side="left", padx=(8,4))
        mk_btn(bar, "📷  Save Image", self._save_image, fg=FG2, padx=8, pady=3).pack(side="left", padx=(0,4))

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG2, pady=4)
        ctrl.pack(fill="x", padx=8, pady=(4,0))

        mk_lbl(ctrl, "Colormap:", fg=FG).pack(side="left", padx=(8,4))
        self.v_cmap = tk.StringVar(value="hot")
        mk_combo(ctrl, self.v_cmap, self.CMAPS, 10).pack(side="left", padx=(0,8))
        self.v_cmap.trace_add("write", lambda *_: self._redraw())

        mk_lbl(ctrl, "View:", fg=FG).pack(side="left", padx=(8,4))
        self.v_proj = tk.StringVar(value="Single slice")
        mk_combo(ctrl, self.v_proj, self.PROJECTIONS, 16).pack(side="left", padx=(0,8))
        self.v_proj.trace_add("write", lambda *_: self._on_proj_change())

        mk_lbl(ctrl, "Slice:", fg=FG).pack(side="left", padx=(8,4))
        self.v_slice = tk.IntVar(value=0)
        self._slice_scale = tk.Scale(ctrl, from_=0, to=0, orient="horizontal",
                                      variable=self.v_slice, bg=BG2, fg=FG,
                                      troughcolor=BG3, highlightthickness=0,
                                      font=F_SMALL, length=150,
                                      command=lambda _: self._redraw())
        self._slice_scale.pack(side="left", padx=(0,8))

        self._slice_label = mk_lbl(ctrl, "0 / 0", fg=FG2, font=F_SMALL)
        self._slice_label.pack(side="left", padx=(0,8))

        self.v_log = tk.BooleanVar(value=False)
        mk_check(ctrl, self.v_log, "Log scale", lambda: self._redraw()).pack(side="left", padx=(8,4))

        # ── Matplotlib canvas ─────────────────────────────────────────────────
        self._fig = Figure(figsize=(8,5), dpi=100, facecolor="#0d1117")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#0d1117")
        self._cbar = None  # track colorbar object
        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(4,0))

        # ── Statistics box ────────────────────────────────────────────────────
        stats_fr = tk.Frame(self, bg=BG2)
        stats_fr.pack(fill="x", padx=8, pady=(4,8))
        mk_lbl(stats_fr, "STATISTICS", fg=ACCENT, font=F_BOLD).pack(anchor="w", padx=8, pady=(6,2))
        self._stats_box = scrolledtext.ScrolledText(stats_fr, height=8, bg="#0d1117", fg=FG,
                                                     font=F_MONO, relief="flat", wrap="word",
                                                     state="disabled", padx=10, pady=8,
                                                     highlightthickness=0)
        self._stats_box.pack(fill="x", padx=8, pady=(0,8))
        for tag, col in [("header",ACCENT),("value",GREEN),("label",FG2),("warn",YELLOW)]:
            self._stats_box.tag_config(tag, foreground=col)

    def _stats_emit(self, text, tag=None):
        self._stats_box.configure(state="normal")
        self._stats_box.insert("end", text + "\n", tag or "")
        self._stats_box.see("end")
        self._stats_box.configure(state="disabled")

    def _stats_clear(self):
        self._stats_box.configure(state="normal")
        self._stats_box.delete("1.0", "end")
        self._stats_box.configure(state="disabled")

    def _load_file(self):
        if not SITK_OK:
            messagebox.showerror("Missing", "SimpleITK required: pip install SimpleITK")
            return
        path = filedialog.askopenfilename(
            title="Select MHD file",
            filetypes=[("MetaImage", "*.mhd"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = sitk.ReadImage(str(path))
            self._arr = sitk.GetArrayFromImage(img).astype(np.float64)
            self._spacing = img.GetSpacing()
            self._origin = img.GetOrigin()
            self._filepath = Path(path)

            nz = self._arr.shape[0]
            self._slice_scale.configure(to=max(0, nz - 1))
            self.v_slice.set(nz // 2)
            self.title(f"MHD Viewer — {self._filepath.name}")

            self._compute_stats()
            self._redraw()
        except Exception as exc:
            import traceback
            messagebox.showerror("Load Error", f"{exc}\n\n{traceback.format_exc()}")

    def _compute_stats(self):
        if self._arr is None:
            return
        self._stats_clear()
        arr = self._arr
        flat = arr.ravel()
        nz, ny, nx = arr.shape
        sx, sy, sz = self._spacing

        sep = "─" * 60
        self._stats_emit(f"FILE: {self._filepath.name}", "header")
        self._stats_emit(sep)
        self._stats_emit(f"  Shape          : {nx} × {ny} × {nz}  (X × Y × Z voxels)", "label")
        self._stats_emit(f"  Spacing        : {sx:.3f} × {sy:.3f} × {sz:.3f} mm", "label")
        self._stats_emit(f"  Physical size  : {nx*sx:.1f} × {ny*sy:.1f} × {nz*sz:.1f} mm", "label")
        self._stats_emit(f"  Origin         : ({self._origin[0]:.1f}, {self._origin[1]:.1f}, {self._origin[2]:.1f}) mm", "label")
        self._stats_emit(sep)
        self._stats_emit(f"  Total voxels   : {flat.size:,}", "label")
        self._stats_emit(f"  Non-zero voxels: {int(np.count_nonzero(flat)):,}  ({np.count_nonzero(flat)/flat.size*100:.1f}%)", "label")
        self._stats_emit(sep)
        self._stats_emit(f"  Sum            : {float(np.sum(flat)):.6e}", "value")
        self._stats_emit(f"  Mean           : {float(np.mean(flat)):.6e}", "value")
        self._stats_emit(f"  Std            : {float(np.std(flat)):.6e}", "value")
        self._stats_emit(f"  Min            : {float(np.min(flat)):.6e}", "value")
        self._stats_emit(f"  Max            : {float(np.max(flat)):.6e}", "value")
        self._stats_emit(f"  Median         : {float(np.median(flat)):.6e}", "value")

        nz_vals = flat[flat != 0]
        if nz_vals.size > 0:
            self._stats_emit(sep)
            self._stats_emit(f"  Non-zero sum   : {float(np.sum(nz_vals)):.6e}", "value")
            self._stats_emit(f"  Non-zero mean  : {float(np.mean(nz_vals)):.6e}", "value")
            self._stats_emit(f"  Non-zero std   : {float(np.std(nz_vals)):.6e}", "value")
            self._stats_emit(f"  Non-zero min   : {float(np.min(nz_vals)):.6e}", "value")
            self._stats_emit(f"  Non-zero max   : {float(np.max(nz_vals)):.6e}", "value")

        if arr.shape[0] > 1:
            self._stats_emit(sep)
            self._stats_emit(f"  Per-slice summary (Z axis):", "header")
            self._stats_emit(f"  {'Slice':>5}  {'Sum':>14}  {'Mean':>14}  {'Max':>14}  {'NonZero':>8}", "label")
            for iz in range(arr.shape[0]):
                sl = arr[iz]
                self._stats_emit(
                    f"  {iz:5d}  {float(np.sum(sl)):14.6e}  {float(np.mean(sl)):14.6e}  "
                    f"{float(np.max(sl)):14.6e}  {int(np.count_nonzero(sl)):8d}", "label")

    def _on_proj_change(self):
        proj = self.v_proj.get()
        self._slice_scale.configure(state="normal" if proj == "Single slice" else "disabled")
        self._redraw()

    def _get_display_slice(self):
        if self._arr is None:
            return None
        proj = self.v_proj.get()
        if proj == "Max projection":
            return np.max(self._arr, axis=0)
        elif proj == "Sum projection":
            return np.sum(self._arr, axis=0)
        elif proj == "Mean projection":
            return np.mean(self._arr, axis=0)
        else:
            iz = max(0, min(self.v_slice.get(), self._arr.shape[0] - 1))
            return self._arr[iz]

    def _redraw(self):
        if self._arr is None:
            return
        sl = self._get_display_slice()
        if sl is None:
            return

        nz = self._arr.shape[0]
        iz = self.v_slice.get()
        self._slice_label.configure(text=f"{iz} / {nz-1}")

        # Remove old colorbar properly before clearing axes
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self._ax.clear()

        cmap = self.v_cmap.get()
        use_log = self.v_log.get()

        display = sl.copy()
        if use_log:
            display = np.where(display > 0, np.log10(display), np.nan)

        sx, sy = self._spacing[0], self._spacing[1]
        extent = [0, sl.shape[1] * sx, 0, sl.shape[0] * sy]

        im = self._ax.imshow(display, cmap=cmap, origin="lower", extent=extent, aspect="equal")
        self._ax.set_xlabel("X (mm)", color=FG2, fontsize=9)
        self._ax.set_ylabel("Y (mm)", color=FG2, fontsize=9)
        self._ax.tick_params(colors=FG2, labelsize=8)

        proj = self.v_proj.get()
        title = f"{self._filepath.name}"
        if proj == "Single slice":
            title += f"  —  slice {iz}/{nz-1}"
        else:
            title += f"  —  {proj}"
        if use_log:
            title += "  (log₁₀)"
        self._ax.set_title(title, color=FG, fontsize=10)

        self._cbar = self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)
        self._cbar.ax.tick_params(colors=FG2, labelsize=8)
        self._cbar.set_label("log₁₀(value)" if use_log else "Value", color=FG2, fontsize=9)

        self._fig.tight_layout()
        self._canvas_widget.draw_idle()

    def _save_image(self):
        if self._arr is None:
            messagebox.showinfo("No data", "Load an MHD file first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save image", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
            messagebox.showinfo("Saved", f"Saved to:\n{path}", parent=self)


# ═══════════════════════════════════════════════════════════════════════════════
# Main launcher GUI
# ═══════════════════════════════════════════════════════════════════════════════

class GateArcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shield Lab — GATE 10 Shielding Simulation"); self.configure(bg=BG)
        self.geometry("1320x860"); self.minsize(1060,700)
        self._proc=None; self._q=queue.Queue()
        self._vars(); self._ttk_style(); self._layout(); self._traces()
        self._on_src_type(); self._on_det_mode(); self._refresh(); self._vrml_refresh()
        self.after(100,self._poll)

    def _vars(self):
        self.v_src=tk.StringVar(value="nuclide"); self.v_nuclide=tk.StringVar(value="F18")
        self.v_kvp=tk.StringVar(value="120"); self.v_al=tk.StringVar(value="2.5")
        self.v_cu=tk.StringVar(value="0.0"); self.v_kbins=tk.StringVar(value="128")
        self.v_barrier=tk.StringVar(value="Lead"); self.v_thick=tk.StringVar(value="5.0")
        self.v_phantom=tk.StringVar(value=PHANTOM_DISPLAY[0])
        self.v_det=tk.StringVar(value="original"); self.v_depth=tk.StringVar(value="10.0")
        self.v_det_sx=tk.StringVar(value=""); self.v_det_sy=tk.StringVar(value=""); self.v_det_sz=tk.StringVar(value="")
        self.v_angle=tk.StringVar(value="0.0"); self.v_angles=tk.StringVar(value="0 15 30 45 60")
        self.v_mode=tk.StringVar(value="single"); self.v_n=tk.StringVar(value="2000000000")
        self.v_test=tk.BooleanVar(value=False); self.v_threads=tk.StringVar(value="1")
        self.v_jobs=tk.StringVar(value="1"); self.v_auto=tk.BooleanVar(value=False)
        self.v_unc=tk.StringVar(value="0.02"); self.v_split=tk.BooleanVar(value=False)
        self.v_nocone=tk.BooleanVar(value=False); self.v_coneang=tk.StringVar(value="")
        self.v_verbose=tk.BooleanVar(value=False); self.v_outdir=tk.StringVar(value="output")
        self.v_dose=tk.BooleanVar(value=False); self.v_uncout=tk.BooleanVar(value=False)
        self.v_vis=tk.BooleanVar(value=False); self.v_vistype=tk.StringVar(value="vrml_file_only")
        self.v_countphot=tk.BooleanVar(value=False); self.v_sweep_to_cvl=tk.BooleanVar(value=False)
        self.v_sp_shape=tk.StringVar(value="none"); self.v_sp_rx=tk.StringVar(value="100.0")
        self.v_sp_ry=tk.StringVar(value="70.0"); self.v_sp_rz=tk.StringVar(value="100.0")
        self.v_sp_mat=tk.StringVar(value=PHANTOM_DISPLAY[1])
        self.v_sp_ox=tk.StringVar(value="0.0"); self.v_sp_oy=tk.StringVar(value="0.0"); self.v_sp_oz=tk.StringVar(value="0.0")
        self.v_vrml_viewer=tk.StringVar(value="")
        self.v_air_edep=tk.StringVar(value=""); self.v_air_unc=tk.StringVar(value="")
        self.v_mat_edep=tk.StringVar(value=""); self.v_mat_unc=tk.StringVar(value="")
        self.v_af_nuclide=tk.StringVar(value="F18"); self.v_af_barrier=tk.StringVar(value="Lead")
        self.v_af_outdir=tk.StringVar(value="output"); self.v_af_target_unc=tk.StringVar(value="0.01")
        self.v_af_tail_n=tk.StringVar(value="3"); self.v_af_alpha_tol=tk.StringVar(value="0.15")
        self.v_af_thick_unc=tk.StringVar(value="0.5"); self.v_af_fit_min_T=tk.StringVar(value="")
        self.v_af_fit_points=tk.StringVar(value=""); self.v_af_interactive=tk.BooleanVar(value=True)

    def _phantom_val(self):
        try: return PHANTOM_VALUES[PHANTOM_DISPLAY.index(self.v_phantom.get())]
        except: return "G4_MUSCLE_SKELETAL_ICRP"

    def _ttk_style(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("Dark.TNotebook",background=BG,borderwidth=0,tabmargins=[0,0,0,0])
        s.configure("Dark.TNotebook.Tab",background=BG3,foreground=FG2,padding=[14,6],font=F_UI,borderwidth=0)
        s.map("Dark.TNotebook.Tab",background=[("selected",BG2),("active",BORDER)],foreground=[("selected",ACCENT),("active",FG)])
        s.configure("G.TCombobox",fieldbackground=BG3,background=BG3,foreground=FG,arrowcolor=FG2,selectbackground=BG3,selectforeground=FG,bordercolor=BORDER,lightcolor=BG3,darkcolor=BG3)
        s.map("G.TCombobox",fieldbackground=[("readonly",BG3)],selectbackground=[("readonly",BG3)],foreground=[("readonly",FG)])

    def _layout(self):
        hdr=tk.Frame(self,bg="#161b22"); hdr.pack(fill="x")
        tk.Label(hdr,text="⚛  Shield Lab",font=F_TITLE,bg="#161b22",fg=ACCENT,padx=16,pady=10).pack(side="left")
        tk.Label(hdr,text="GATE 10 · Shielding Simulation Launcher",font=F_UI,bg="#161b22",fg=FG2).pack(side="left")
        # MHD Viewer button in header
        mk_btn(hdr, "🔬 MHD Viewer", self._open_mhd_viewer, fg=ACCENT, padx=10, pady=4).pack(side="left", padx=(16,0))
        vtk_col=GREEN if VTK_OK else YELLOW; tk.Label(hdr,text="VTK ✓" if VTK_OK else "VTK ✗",font=F_SMALL,bg="#161b22",fg=vtk_col,padx=8).pack(side="right")
        ai_col=GREEN if AI_OK else YELLOW; tk.Label(hdr,text="Archer ✓" if AI_OK else "Archer ✗",font=F_SMALL,bg="#161b22",fg=ai_col,padx=8).pack(side="right")
        mpl_col=GREEN if MPL_OK else YELLOW; tk.Label(hdr,text="MPL ✓" if MPL_OK else "MPL ✗",font=F_SMALL,bg="#161b22",fg=mpl_col,padx=8).pack(side="right")
        tk.Label(hdr,text=SIM_SCRIPT.name,font=F_SMALL,bg="#161b22",fg=FG2,padx=8).pack(side="right")
        body=tk.PanedWindow(self,orient="horizontal",bg=BG,sashwidth=5,sashrelief="flat"); body.pack(fill="both",expand=True)
        left=tk.Frame(body,bg=BG,width=520); left.pack_propagate(False); body.add(left,minsize=460)
        nb=ttk.Notebook(left,style="Dark.TNotebook"); nb.pack(fill="both",expand=True)
        def tab(title):
            f=tk.Frame(nb,bg=BG2); f.columnconfigure(1,weight=1); nb.add(f,text=f"  {title}  "); return f
        self._build_source(tab("Source")); self._build_barrier(tab("Barrier & Phantom"))
        self._build_detector(tab("Detector")); self._build_angle_mode(tab("Angle & Mode"))
        self._build_options(tab("Options")); self._build_analysis(tab("Analysis ⚗"))
        self._build_archer_fit(tab("Archer Fit 📈"))
        right=tk.Frame(body,bg=BG); body.add(right,minsize=500); self._build_right(right)

    def _open_mhd_viewer(self):
        if not SITK_OK or not MPL_OK:
            missing = []
            if not SITK_OK: missing.append("SimpleITK")
            if not MPL_OK: missing.append("matplotlib")
            messagebox.showerror("Missing packages",
                                 f"MHD Viewer requires: {', '.join(missing)}\n\n"
                                 f"Install with: pip install {' '.join(missing)}")
            return
        MHDViewerWindow(self)

    def _build_source(self,f):
        rb=tk.Frame(f,bg=BG2); rb.grid(row=0,column=0,columnspan=3,sticky="w",padx=8,pady=(10,4))
        mk_lbl(rb,"Source type").pack(side="left",padx=(0,12))
        mk_radio(rb,"Nuclide","nuclide",self.v_src,self._on_src_type).pack(side="left",padx=(0,8))
        mk_radio(rb,"X-ray (kV)","xray",self.v_src,self._on_src_type).pack(side="left")
        self._nuc_row=tk.Frame(f,bg=BG2); self._nuc_row.columnconfigure(1,weight=1)
        self._nuc_row.grid(row=1,column=0,columnspan=3,sticky="ew")
        grid_row(self._nuc_row,0,"Nuclide",mk_combo(self._nuc_row,self.v_nuclide,NUCLIDES,22))
        self._xray_rows=tk.Frame(f,bg=BG2); self._xray_rows.columnconfigure(1,weight=1)
        self._xray_rows.grid(row=2,column=0,columnspan=3,sticky="ew")
        grid_row(self._xray_rows,0,"kVp",mk_entry(self._xray_rows,self.v_kvp,10),"kV")
        grid_row(self._xray_rows,1,"Al filtration",mk_entry(self._xray_rows,self.v_al,10),"mm")
        grid_row(self._xray_rows,2,"Cu filtration",mk_entry(self._xray_rows,self.v_cu,10),"mm")
        grid_row(self._xray_rows,3,"Spectrum bins",mk_entry(self._xray_rows,self.v_kbins,10))
        mk_btn(self._xray_rows,"📊  Preview kV Spectrum",self._show_spectrum,fg=ACCENT,padx=10,pady=5).grid(row=4,column=0,columnspan=3,sticky="w",padx=8,pady=(6,2))

    def _on_src_type(self,*_):
        if self.v_src.get()=="nuclide": self._nuc_row.grid(); self._xray_rows.grid_remove()
        else: self._nuc_row.grid_remove(); self._xray_rows.grid()
        self._refresh()

    def _on_sp_shape(self,*_):
        shape=self.v_sp_shape.get()
        if shape=="sphere": self._sp_sphere_fr.grid(); self._sp_ellip_fr.grid_remove()
        elif shape=="ellipsoid": self._sp_sphere_fr.grid_remove(); self._sp_ellip_fr.grid()
        else: self._sp_sphere_fr.grid_remove(); self._sp_ellip_fr.grid_remove()
        if hasattr(self,'_sp_offset_fr'):
            if shape!="none": self._sp_offset_fr.grid()
            else: self._sp_offset_fr.grid_remove()
        if hasattr(self,"cmd_box"): self._refresh()

    def _build_barrier(self,f):
        mk_lbl(f,"Barrier",fg=ACCENT,font=F_BOLD).grid(row=0,column=0,columnspan=3,sticky="w",padx=8,pady=(10,2))
        grid_row(f,1,"Material",mk_combo(f,self.v_barrier,BARRIERS,22))
        grid_row(f,2,"Thickness",mk_entry(f,self.v_thick,10),"mm")
        tk.Frame(f,bg=BORDER,height=1).grid(row=3,column=0,columnspan=3,sticky="ew",padx=8,pady=(12,4))
        mk_lbl(f,"Tissue Phantom",fg=ACCENT,font=F_BOLD).grid(row=4,column=0,columnspan=3,sticky="w",padx=8,pady=(4,2))
        grid_row(f,5,"Material",mk_combo(f,self.v_phantom,PHANTOM_DISPLAY,36))
        tk.Frame(f,bg=BORDER,height=1).grid(row=6,column=0,columnspan=3,sticky="ew",padx=8,pady=(12,4))
        mk_lbl(f,"Source Phantom",fg=ACCENT,font=F_BOLD).grid(row=7,column=0,columnspan=3,sticky="w",padx=8,pady=(4,2))
        mk_lbl(f,"  Surrounds the point source",fg=FG2,font=F_SMALL).grid(row=8,column=0,columnspan=3,sticky="w",padx=8,pady=(0,4))
        sp_rb=tk.Frame(f,bg=BG2); sp_rb.grid(row=9,column=0,columnspan=3,sticky="w",padx=8,pady=(0,4))
        mk_lbl(sp_rb,"Shape").pack(side="left",padx=(0,8))
        for t,v in [("None","none"),("Sphere","sphere"),("Ellipsoid","ellipsoid")]:
            mk_radio(sp_rb,t,v,self.v_sp_shape,self._on_sp_shape).pack(side="left",padx=(0,10))
        self._sp_sphere_fr=tk.Frame(f,bg=BG2); self._sp_sphere_fr.columnconfigure(1,weight=1)
        self._sp_sphere_fr.grid(row=10,column=0,columnspan=3,sticky="ew")
        grid_row(self._sp_sphere_fr,0,"    Radius",mk_entry(self._sp_sphere_fr,self.v_sp_rx,10),"mm")
        self._sp_ellip_fr=tk.Frame(f,bg=BG2); self._sp_ellip_fr.columnconfigure(1,weight=1)
        self._sp_ellip_fr.grid(row=11,column=0,columnspan=3,sticky="ew")
        grid_row(self._sp_ellip_fr,0,"    X semi-axis",mk_entry(self._sp_ellip_fr,self.v_sp_rx,10),"mm")
        grid_row(self._sp_ellip_fr,1,"    Y semi-axis",mk_entry(self._sp_ellip_fr,self.v_sp_ry,10),"mm")
        grid_row(self._sp_ellip_fr,2,"    Z semi-axis",mk_entry(self._sp_ellip_fr,self.v_sp_rz,10),"mm")
        self._sp_offset_fr=tk.Frame(f,bg=BG2); self._sp_offset_fr.columnconfigure(1,weight=1)
        self._sp_offset_fr.grid(row=12,column=0,columnspan=3,sticky="ew")
        grid_row(self._sp_offset_fr,0,"    Offset X",mk_entry(self._sp_offset_fr,self.v_sp_ox,10),"mm")
        grid_row(self._sp_offset_fr,1,"    Offset Y",mk_entry(self._sp_offset_fr,self.v_sp_oy,10),"mm")
        grid_row(self._sp_offset_fr,2,"    Offset Z",mk_entry(self._sp_offset_fr,self.v_sp_oz,10),"mm (+Z→barrier)")
        grid_row(f,13,"    Material",mk_combo(f,self.v_sp_mat,PHANTOM_DISPLAY,36))
        self._on_sp_shape()

    def _build_detector(self,f):
        mk_lbl(f,"Detector position:",fg=FG,font=F_BOLD).grid(row=0,column=0,columnspan=2,sticky="w",padx=8,pady=(10,4))
        for i,(text,val) in enumerate(DETECTOR_OPTIONS):
            mk_radio(f,text,val,self.v_det,self._on_det_mode).grid(row=i+1,column=0,columnspan=2,sticky="w",padx=14,pady=2)
        self._det_custom=tk.Frame(f,bg=BG2); self._det_custom.columnconfigure(1,weight=1)
        self._det_custom.grid(row=len(DETECTOR_OPTIONS)+1,column=0,columnspan=2,sticky="ew")
        grid_row(self._det_custom,0,"    Custom depth",mk_entry(self._det_custom,self.v_depth,10),"mm from tissue face")
        tk.Frame(f,bg=BORDER,height=1).grid(row=len(DETECTOR_OPTIONS)+2,column=0,columnspan=2,sticky="ew",padx=8,pady=(8,4))
        mk_lbl(f,"Detector size (blank = default):",fg=FG,font=F_BOLD).grid(row=len(DETECTOR_OPTIONS)+3,column=0,columnspan=2,sticky="w",padx=8,pady=(4,2))
        sz_fr=tk.Frame(f,bg=BG2); sz_fr.columnconfigure(1,weight=1)
        sz_fr.grid(row=len(DETECTOR_OPTIONS)+4,column=0,columnspan=2,sticky="ew")
        grid_row(sz_fr,0,"    X size",mk_entry(sz_fr,self.v_det_sx,10),"mm (default 250)")
        grid_row(sz_fr,1,"    Y size",mk_entry(sz_fr,self.v_det_sy,10),"mm (default 250)")
        grid_row(sz_fr,2,"    Z size",mk_entry(sz_fr,self.v_det_sz,10),"mm (default 20 or 5)")
        mk_lbl(f,"  Spacing: 2.5×2.5×5.0 mm. Voxels = size ÷ spacing.",fg=FG2,font=F_SMALL).grid(row=len(DETECTOR_OPTIONS)+5,column=0,columnspan=2,sticky="w",padx=8,pady=(0,6))
        f.columnconfigure(1,weight=1)

    def _on_det_mode(self,*_):
        if self.v_det.get()=="custom": self._det_custom.grid()
        else: self._det_custom.grid_remove()
        self._refresh()

    def _build_angle_mode(self,f):
        mk_lbl(f,"Oblique Angle",fg=ACCENT,font=F_BOLD).grid(row=0,column=0,columnspan=3,sticky="w",padx=8,pady=(10,2))
        grid_row(f,1,"Barrier angle",mk_entry(f,self.v_angle,10),"degrees from normal")
        grid_row(f,2,"Sweep angle list",mk_entry(f,self.v_angles,22),"space-separated °")
        tk.Frame(f,bg=BORDER,height=1).grid(row=4,column=0,columnspan=3,sticky="ew",padx=8,pady=(10,4))
        mk_lbl(f,"Run Mode",fg=ACCENT,font=F_BOLD).grid(row=5,column=0,columnspan=3,sticky="w",padx=8,pady=(4,2))
        for i,(text,val) in enumerate(RUN_MODES):
            mk_radio(f,text,val,self.v_mode,self._on_mode_change).grid(row=6+i,column=0,columnspan=2,sticky="w",padx=14,pady=2)
        f.columnconfigure(1,weight=1)

    def _on_mode_change(self,*_):
        mode=self.v_mode.get(); state="disabled" if mode in MODES_NO_BARRIER else "normal"
        for w in self._barrier_widgets: w.configure(state=state)
        if hasattr(self,"_cvl_cb"): self._cvl_cb.configure(state="normal" if mode=="nuclide_sweep" else "disabled")
        if mode!="nuclide_sweep": self.v_sweep_to_cvl.set(False)
        self._refresh()

    def _build_options(self,f):
        mk_lbl(f,"Run Control",fg=ACCENT,font=F_BOLD).grid(row=0,column=0,columnspan=3,sticky="w",padx=8,pady=(10,2))
        grid_row(f,1,"N primaries",mk_entry(f,self.v_n,16)); grid_row(f,2,"Threads",mk_entry(f,self.v_threads,6))
        grid_row(f,3,"Parallel jobs",mk_entry(f,self.v_jobs,6)); grid_row(f,4,"UNC goal",mk_entry(f,self.v_unc,8))
        grid_row(f,5,"Cone half-angle",mk_entry(f,self.v_coneang,8),"° (blank = default)")
        cb1=tk.Frame(f,bg=BG2); cb1.grid(row=6,column=0,columnspan=3,sticky="w",padx=8,pady=4)
        for var,text in [(self.v_test,"Test mode"),(self.v_auto,"Auto threads/jobs"),(self.v_split,"Photon splitting"),
                         (self.v_nocone,"Disable cone source"),(self.v_countphot,"Count photons"),(self.v_verbose,"Verbose")]:
            mk_check(cb1,var,text,self._refresh).pack(side="left",padx=(0,10))
        cb2=tk.Frame(f,bg=BG2); cb2.grid(row=7,column=0,columnspan=3,sticky="w",padx=8,pady=(0,4))
        self._cvl_cb=mk_check(cb2,self.v_sweep_to_cvl,"Stop at CVL — nuclide sweep only",self._refresh)
        self._cvl_cb.pack(side="left",padx=(0,10))
        tk.Frame(f,bg=BORDER,height=1).grid(row=9,column=0,columnspan=3,sticky="ew",padx=8,pady=(6,4))
        mk_lbl(f,"Output",fg=ACCENT,font=F_BOLD).grid(row=10,column=0,columnspan=3,sticky="w",padx=8,pady=(4,2))
        dir_fr=tk.Frame(f,bg=BG2); dir_fr.columnconfigure(1,weight=1); dir_fr.grid(row=10,column=0,columnspan=3,sticky="ew")
        mk_lbl(dir_fr,"Output directory",anchor="w").grid(row=0,column=0,sticky="w",padx=(8,4),pady=3)
        mk_entry(dir_fr,self.v_outdir,22).grid(row=0,column=1,sticky="ew",padx=(0,4),pady=3)
        mk_btn(dir_fr,"…",self._browse,padx=6,pady=2).grid(row=0,column=2,padx=(0,8),pady=3)
        grid_row(f,11,"Vis type",mk_combo(f,self.v_vistype,VIS_TYPES,18))
        cb3=tk.Frame(f,bg=BG2); cb3.grid(row=12,column=0,columnspan=3,sticky="w",padx=8,pady=4)
        for var,text in [(self.v_dose,"Dose maps"),(self.v_uncout,"Uncertainty maps"),(self.v_vis,"Visualisation")]:
            mk_check(cb3,var,text,self._refresh).pack(side="left",padx=(0,10))
        tk.Frame(f,bg=BORDER,height=1).grid(row=13,column=0,columnspan=3,sticky="ew",padx=8,pady=(8,4))
        vtk_col=GREEN if VTK_OK else YELLOW
        mk_lbl(f,"VRML Viewer",fg=ACCENT,font=F_BOLD).grid(row=14,column=0,columnspan=2,sticky="w",padx=8,pady=(2,0))
        mk_lbl(f,"VTK ✓" if VTK_OK else "VTK ✗",fg=vtk_col,font=F_SMALL).grid(row=14,column=2,sticky="e",padx=8,pady=(2,0))
        vwr_fr=tk.Frame(f,bg=BG2); vwr_fr.columnconfigure(1,weight=1); vwr_fr.grid(row=15,column=0,columnspan=3,sticky="ew")
        mk_lbl(vwr_fr,"Fallback viewer",anchor="w").grid(row=0,column=0,sticky="w",padx=(8,4),pady=3)
        mk_entry(vwr_fr,self.v_vrml_viewer,22).grid(row=0,column=1,sticky="ew",padx=(0,4),pady=3)
        mk_btn(vwr_fr,"…",self._browse_viewer,padx=6,pady=2).grid(row=0,column=2,padx=(0,8),pady=3)
        f.columnconfigure(1,weight=1)

    def _build_analysis(self,f):
        f.columnconfigure(1,weight=1)
        sitk_col=GREEN if SITK_OK else RED
        mk_lbl(f,"Transmission Factor Calculator",fg=ACCENT,font=F_BOLD).grid(row=0,column=0,columnspan=3,sticky="w",padx=8,pady=(10,0))
        mk_lbl(f,"SimpleITK ✓" if SITK_OK else "SimpleITK not found",fg=sitk_col,font=F_SMALL).grid(row=1,column=0,columnspan=3,sticky="w",padx=8,pady=(0,4))
        tk.Frame(f,bg=BORDER,height=1).grid(row=2,column=0,columnspan=3,sticky="ew",padx=8,pady=(4,8))
        mk_lbl(f,"Air Reference",fg=ACCENT,font=F_BOLD).grid(row=3,column=0,columnspan=3,sticky="w",padx=8,pady=(0,2))
        def _fr(parent,row,label,var,cmd):
            mk_lbl(parent,label,anchor="w").grid(row=row,column=0,sticky="w",padx=(8,4),pady=2)
            tk.Entry(parent,textvariable=var,width=32,bg=BG3,fg=FG,insertbackground=FG,relief="flat",font=F_SMALL,bd=3,highlightthickness=1,highlightbackground=BORDER,highlightcolor=ACCENT).grid(row=row,column=1,sticky="ew",padx=(0,4),pady=2)
            mk_btn(parent,"…",cmd,padx=5,pady=2).grid(row=row,column=2,padx=(0,8),pady=2)
        _fr(f,4,"edep MHD",self.v_air_edep,self._browse_air_edep); _fr(f,6,"unc MHD",self.v_air_unc,self._browse_air_unc)
        tk.Frame(f,bg=BORDER,height=1).grid(row=9,column=0,columnspan=3,sticky="ew",padx=8,pady=(2,8))
        mk_lbl(f,"Material (with barrier)",fg=ACCENT,font=F_BOLD).grid(row=10,column=0,columnspan=3,sticky="w",padx=8,pady=(0,2))
        _fr(f,11,"edep MHD",self.v_mat_edep,self._browse_mat_edep); _fr(f,13,"unc MHD",self.v_mat_unc,self._browse_mat_unc)
        tk.Frame(f,bg=BORDER,height=1).grid(row=16,column=0,columnspan=3,sticky="ew",padx=8,pady=(2,8))
        mk_lbl(f,"Averaging",fg=ACCENT,font=F_BOLD).grid(row=17,column=0,columnspan=3,sticky="w",padx=8,pady=(0,2))
        self.v_avg_mode=tk.StringVar(value="auto")
        avg_fr=tk.Frame(f,bg=BG2); avg_fr.grid(row=18,column=0,columnspan=3,sticky="w",padx=8,pady=(0,2))
        for t,v in [("Auto-detect","auto"),("Original 4-slice","original"),("Mean all","mean")]:
            mk_radio(avg_fr,t,v,self.v_avg_mode).pack(side="left",padx=(0,12))
        tk.Frame(f,bg=BORDER,height=1).grid(row=20,column=0,columnspan=3,sticky="ew",padx=8,pady=(2,6))
        calc_row=tk.Frame(f,bg=BG2); calc_row.grid(row=21,column=0,columnspan=3,sticky="w",padx=8,pady=(4,6))
        mk_btn(calc_row,"⚗  Calculate T",self._calc_transmission,bg=ACCENT,fg="#0d1117",font=("Segoe UI",10,"bold") if _W else ("Helvetica",10,"bold"),padx=14,pady=7).pack(side="left")
        mk_btn(calc_row,"✕  Clear",self._clear_analysis,fg=FG2,padx=10,pady=7).pack(side="left",padx=(8,0))
        res_fr=tk.Frame(f,bg=BG2); res_fr.grid(row=22,column=0,columnspan=3,sticky="nsew",padx=8,pady=(0,8)); f.rowconfigure(22,weight=1)
        mk_lbl(res_fr,"RESULTS",fg=ACCENT,font=F_BOLD).pack(anchor="w",padx=8,pady=(6,2))
        self.analysis_box=scrolledtext.ScrolledText(res_fr,height=10,bg="#0d1117",fg=FG,font=F_MONO,relief="flat",wrap="word",state="disabled",padx=10,pady=8,highlightthickness=0)
        self.analysis_box.pack(fill="both",expand=True,padx=8,pady=(0,8))
        for tag,col in [("header",ACCENT),("value",GREEN),("warn",YELLOW),("error",RED),("label",FG2)]:
            self.analysis_box.tag_config(tag,foreground=col)

    def _analysis_emit(self,text,tag=None):
        self.analysis_box.configure(state="normal"); self.analysis_box.insert("end",text+"\n",tag or ""); self.analysis_box.see("end"); self.analysis_box.configure(state="disabled")
    def _clear_analysis(self):
        self.analysis_box.configure(state="normal"); self.analysis_box.delete("1.0","end"); self.analysis_box.configure(state="disabled")
    def _browse_air_edep(self):
        p=filedialog.askopenfilename(title="Air — edep MHD",filetypes=[("MetaImage","*.mhd"),("All","*.*")])
        if p:
            self.v_air_edep.set(p); unc=Path(p).with_name(Path(p).stem+"_uncertainty.mhd")
            if unc.exists() and not self.v_air_unc.get(): self.v_air_unc.set(str(unc))
    def _browse_air_unc(self):
        p=filedialog.askopenfilename(title="Air — unc MHD",filetypes=[("MetaImage","*.mhd"),("All","*.*")])
        if p: self.v_air_unc.set(p)
    def _browse_mat_edep(self):
        p=filedialog.askopenfilename(title="Material — edep MHD",filetypes=[("MetaImage","*.mhd"),("All","*.*")])
        if p:
            self.v_mat_edep.set(p); unc=Path(p).with_name(Path(p).stem+"_uncertainty.mhd")
            if unc.exists() and not self.v_mat_unc.get(): self.v_mat_unc.set(str(unc))
    def _browse_mat_unc(self):
        p=filedialog.askopenfilename(title="Material — unc MHD",filetypes=[("MetaImage","*.mhd"),("All","*.*")])
        if p: self.v_mat_unc.set(p)
    def _read_mhd(self,path):
        if not SITK_OK: raise RuntimeError("SimpleITK not available")
        if not os.path.isfile(str(path)): raise FileNotFoundError(f"Not found: {path}")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float64)
    def _detector_mean(self,arr,mode):
        flat=arr.ravel(); n=flat.size
        if mode=="auto": mode="original" if n==4 else "mean"
        if mode=="original" and n!=4: raise ValueError(f"Original 4-slice but {n} voxel(s).")
        return float(np.mean(flat)),n
    def _calc_transmission(self):
        if not SITK_OK: messagebox.showerror("Missing","pip install SimpleITK"); return
        air_path=self.v_air_edep.get().strip(); mat_path=self.v_mat_edep.get().strip()
        if not air_path or not mat_path: messagebox.showwarning("Missing","Select both files."); return
        self._clear_analysis(); avg_mode=self.v_avg_mode.get()
        try:
            air_arr=self._read_mhd(air_path); mat_arr=self._read_mhd(mat_path)
            air_mean,air_n=self._detector_mean(air_arr,avg_mode); mat_mean,mat_n=self._detector_mean(mat_arr,avg_mode)
            if air_mean<=0: raise ValueError("Air edep ≤ 0")
            T=mat_mean/air_mean; rel_unc_T=None
            air_unc_path=self.v_air_unc.get().strip(); mat_unc_path=self.v_mat_unc.get().strip()
            if air_unc_path and mat_unc_path:
                au=self._read_mhd(air_unc_path).ravel(); mu=self._read_mhd(mat_unc_path).ravel()
                aw=air_arr.ravel(); mw=mat_arr.ravel()
                if avg_mode=="original" or (avg_mode=="auto" and air_n==4):
                    ar=float(np.sqrt(np.sum((aw*au)**2)))/float(np.sum(aw)) if np.sum(aw)>0 else None
                    mr=float(np.sqrt(np.sum((mw*mu)**2)))/float(np.sum(mw)) if np.sum(mw)>0 else None
                else: ar=float(np.sqrt(np.mean(au**2))); mr=float(np.sqrt(np.mean(mu**2)))
                if ar is not None and mr is not None: rel_unc_T=float(np.sqrt(ar**2+mr**2))
            sep="─"*56; self._analysis_emit("TRANSMISSION FACTOR RESULT","header"); self._analysis_emit(sep)
            self._analysis_emit(f"  Air:      {Path(air_path).name}","label"); self._analysis_emit(f"  Material: {Path(mat_path).name}","label"); self._analysis_emit(sep)
            if rel_unc_T: self._analysis_emit(f"  T = {T:.6f}  ±{T*rel_unc_T:.6f}  ({rel_unc_T*100:.2f}%)","value")
            else: self._analysis_emit(f"  T = {T:.6f}","value")
            if T>1.0: self._analysis_emit("  ⚠  T > 1","warn")
            self._emit(f"[Analysis] T = {T:.6f}","ok")
        except Exception as exc:
            import traceback; self._analysis_emit(f"ERROR: {exc}","error"); self._analysis_emit(traceback.format_exc(),"error")

    def _build_archer_fit(self,f):
        f.columnconfigure(1,weight=1)
        ai_col=GREEN if AI_OK else RED
        mk_lbl(f,"Archer Equation Fitting",fg=ACCENT,font=F_BOLD).grid(row=0,column=0,columnspan=3,sticky="w",padx=8,pady=(10,0))
        mk_lbl(f,"✓" if AI_OK else "shieldLabAnalyze not found",fg=ai_col,font=F_SMALL).grid(row=1,column=0,columnspan=3,sticky="w",padx=8,pady=(0,4))
        tk.Frame(f,bg=BORDER,height=1).grid(row=2,column=0,columnspan=3,sticky="ew",padx=8,pady=(2,6))
        AF_NUC=NUCLIDES
        grid_row(f,4,"Nuclide",mk_combo(f,self.v_af_nuclide,AF_NUC,18)); grid_row(f,5,"Barrier",mk_combo(f,self.v_af_barrier,BARRIERS,18))
        dir_fr=tk.Frame(f,bg=BG2); dir_fr.columnconfigure(1,weight=1); dir_fr.grid(row=6,column=0,columnspan=3,sticky="ew")
        mk_lbl(dir_fr,"Output directory",anchor="w").grid(row=0,column=0,sticky="w",padx=(8,4),pady=3)
        mk_entry(dir_fr,self.v_af_outdir,22).grid(row=0,column=1,sticky="ew",padx=(0,4),pady=3)
        bdf=tk.Frame(dir_fr,bg=BG2); bdf.grid(row=0,column=2,padx=(0,8),pady=3)
        mk_btn(bdf,"…",self._browse_af_outdir,padx=4,pady=2).pack(side="left",padx=(0,2))
        mk_btn(bdf,"↻ Sim",self._sync_af_outdir,fg=FG2,padx=4,pady=2).pack(side="left")
        scan_fr=tk.Frame(f,bg=BG2); scan_fr.grid(row=7,column=0,columnspan=3,sticky="w",padx=8,pady=(2,2))
        mk_btn(scan_fr,"🔍 Scan",self._scan_af_data,fg=ACCENT,padx=10,pady=4).pack(side="left")
        self._af_scan_label=mk_lbl(scan_fr,"",fg=FG2,font=F_SMALL); self._af_scan_label.pack(side="left",padx=(8,0))
        tk.Frame(f,bg=BORDER,height=1).grid(row=8,column=0,columnspan=3,sticky="ew",padx=8,pady=(8,6))
        grid_row(f,10,"Alpha tail pts",mk_entry(f,self.v_af_tail_n,8)); grid_row(f,11,"Alpha tolerance",mk_entry(f,self.v_af_alpha_tol,8))
        grid_row(f,12,"Thickness unc",mk_entry(f,self.v_af_thick_unc,8),"mm"); grid_row(f,13,"Target unc",mk_entry(f,self.v_af_target_unc,8))
        grid_row(f,14,"Fit min T",mk_entry(f,self.v_af_fit_min_T,8)); grid_row(f,15,"Fit max points",mk_entry(f,self.v_af_fit_points,8))
        mk_check(f,self.v_af_interactive,"Interactive tuner").grid(row=16,column=0,columnspan=3,sticky="w",padx=8,pady=(4,2))
        tk.Frame(f,bg=BORDER,height=1).grid(row=17,column=0,columnspan=3,sticky="ew",padx=8,pady=(6,6))
        btn_fr=tk.Frame(f,bg=BG2); btn_fr.grid(row=18,column=0,columnspan=3,sticky="w",padx=8,pady=(2,4))
        mk_btn(btn_fr,"📈 Fit Single",self._run_archer_single,bg=ACCENT,fg="#0d1117",font=("Segoe UI",10,"bold") if _W else ("Helvetica",10,"bold"),padx=12,pady=7).pack(side="left",padx=(0,6))
        mk_btn(btn_fr,"📊 Fit All",self._run_archer_all,bg="#8b5cf6",fg="white",padx=12,pady=7).pack(side="left",padx=(0,6))
        mk_btn(btn_fr,"🔢 Estimate N",self._run_archer_estimate_n,fg=ACCENT,padx=10,pady=7).pack(side="left",padx=(0,6))
        mk_btn(btn_fr,"📐 Lu-177",self._run_lu177_example,fg=FG2,padx=10,pady=7).pack(side="left")
        res_fr=tk.Frame(f,bg=BG2); res_fr.grid(row=19,column=0,columnspan=3,sticky="nsew",padx=8,pady=(4,8)); f.rowconfigure(19,weight=1)
        mk_lbl(res_fr,"ARCHER FIT RESULTS",fg=ACCENT,font=F_BOLD).pack(anchor="w",padx=8,pady=(6,2))
        self.archer_box=scrolledtext.ScrolledText(res_fr,height=10,bg="#0d1117",fg=FG,font=F_MONO,relief="flat",wrap="word",state="disabled",padx=10,pady=8,highlightthickness=0)
        self.archer_box.pack(fill="both",expand=True,padx=8,pady=(0,8))
        for tag,col in [("header",ACCENT),("value",GREEN),("warn",YELLOW),("error",RED),("label",FG2),("ok",GREEN)]:
            self.archer_box.tag_config(tag,foreground=col)
        if not AI_OK: self._archer_emit("shieldLabAnalyze.py not available.","warn")

    def _archer_emit(self,text,tag=None):
        self.archer_box.configure(state="normal"); self.archer_box.insert("end",text+"\n",tag or ""); self.archer_box.see("end"); self.archer_box.configure(state="disabled")
    def _clear_archer(self):
        self.archer_box.configure(state="normal"); self.archer_box.delete("1.0","end"); self.archer_box.configure(state="disabled")
    def _browse_af_outdir(self):
        d=filedialog.askdirectory(title="Select output directory")
        if d: self.v_af_outdir.set(d)
    def _sync_af_outdir(self): self.v_af_outdir.set(self.v_outdir.get())
    def _scan_af_data(self):
        outdir=Path(self.v_af_outdir.get().strip() or "output")
        if not outdir.exists(): self._af_scan_label.configure(text=f"Not found",fg=RED); return
        dose_mhd=sorted([p for p in outdir.glob("*_*mm_*.mhd") if p.name.endswith(("_dose.mhd","_edep.mhd")) and "_uncertainty" not in p.name])
        if not dose_mhd: self._af_scan_label.configure(text="No files",fg=YELLOW); return
        combos={}; air_refs=set()
        for p in dose_mhd:
            parts=p.stem.split("_")
            if len(parts)>=3:
                if parts[1]=="Air": air_refs.add(parts[0])
                else: combos[(parts[0],parts[1])]=combos.get((parts[0],parts[1]),0)+1
        total=sum(1 for (n,_) in combos if n in air_refs)
        self._af_scan_label.configure(text=f"{total} fittable, {len(dose_mhd)} files",fg=GREEN if total>0 else YELLOW)
    def _get_archer_kwargs(self):
        kw={}; v=self.v_af_fit_min_T.get().strip()
        if v: kw["fit_min_T"]=float(v)
        v=self.v_af_fit_points.get().strip()
        if v: kw["fit_points"]=int(v)
        kw["alpha_tail_n"]=int(self.v_af_tail_n.get().strip() or "3")
        kw["alpha_tol"]=float(self.v_af_alpha_tol.get().strip() or "0.15")
        kw["thickness_unc_mm"]=float(self.v_af_thick_unc.get().strip() or "0.5")
        kw["target_unc"]=float(self.v_af_target_unc.get().strip() or "0.01")
        return kw
    def _run_archer_single(self):
        if not AI_OK: messagebox.showerror("Missing","shieldLabAnalyze.py"); return
        nuc=self.v_af_nuclide.get(); bar=self.v_af_barrier.get()
        outdir=Path(self.v_af_outdir.get().strip() or "output").resolve(); interact=self.v_af_interactive.get()
        if not outdir.exists(): messagebox.showerror("Not found",str(outdir)); return
        self._clear_archer(); self._archer_emit(f"Fitting {nuc}/{bar}","header")
        def _w():
            import io,contextlib
            try:
                kw=self._get_archer_kwargs(); buf=io.StringIO()
                with contextlib.redirect_stdout(buf): a,b,g,fvl=ai.analyze_one(nuc,bar,outdir,interactive=interact,make_plot=not interact,**kw)
                for line in buf.getvalue().splitlines(): self.after(0,lambda l=line: self._archer_emit(l))
                self.after(0,lambda: self._archer_emit(f"  RESULT: a={a:.7f} b={b:.7f} g={g:.7f}","value"))
            except Exception as exc:
                import traceback; self.after(0,lambda: self._archer_emit(f"ERROR: {exc}","error"))
        threading.Thread(target=_w,daemon=True).start()
    def _run_archer_all(self):
        if not AI_OK: return
        outdir=Path(self.v_af_outdir.get().strip() or "output").resolve(); self._clear_archer()
        def _w():
            import io,contextlib
            try:
                kw=self._get_archer_kwargs(); buf=io.StringIO()
                with contextlib.redirect_stdout(buf): ai.analyze_all(outdir,target_unc=kw.get("target_unc",0.01),alpha_tail_n=kw.get("alpha_tail_n",3),alpha_tol=kw.get("alpha_tol",0.15))
                for line in buf.getvalue().splitlines(): self.after(0,lambda l=line: self._archer_emit(l))
                self.after(0,lambda: self._archer_emit("All fits complete.","ok"))
            except Exception as exc:
                import traceback; self.after(0,lambda: self._archer_emit(traceback.format_exc(),"error"))
        threading.Thread(target=_w,daemon=True).start()
    def _run_archer_estimate_n(self):
        if not AI_OK: return
        nuc=self.v_af_nuclide.get(); bar=self.v_af_barrier.get()
        outdir=Path(self.v_af_outdir.get().strip() or "output").resolve(); target=float(self.v_af_target_unc.get().strip() or "0.01")
        self._clear_archer()
        def _w():
            import io,contextlib
            try:
                buf=io.StringIO()
                with contextlib.redirect_stdout(buf): ai.run_estimate_n(nuc,bar,outdir,target)
                for line in buf.getvalue().splitlines(): self.after(0,lambda l=line: self._archer_emit(l))
            except Exception as exc:
                import traceback; self.after(0,lambda: self._archer_emit(traceback.format_exc(),"error"))
        threading.Thread(target=_w,daemon=True).start()
    def _run_lu177_example(self):
        if not AI_OK: return
        self._clear_archer()
        import io,contextlib; buf=io.StringIO()
        with contextlib.redirect_stdout(buf): ai.lu177_room_example()
        for line in buf.getvalue().splitlines(): self._archer_emit(line)

    def _build_right(self,parent):
        cmd_fr=tk.Frame(parent,bg=BG2); cmd_fr.pack(fill="x",padx=8,pady=(8,4))
        hdr=tk.Frame(cmd_fr,bg=BG2); hdr.pack(fill="x",padx=8,pady=(6,2))
        tk.Label(hdr,text="COMMAND PREVIEW",font=F_BOLD,bg=BG2,fg=ACCENT).pack(side="left")
        mk_btn(hdr,"Copy",self._copy,fg=FG2,padx=8,pady=2).pack(side="right")
        self.cmd_box=tk.Text(cmd_fr,height=5,state="disabled",bg="#0d1117",fg=ACCENT,font=F_MONO,relief="flat",wrap="word",padx=10,pady=8,highlightthickness=0)
        self.cmd_box.pack(fill="x",padx=8,pady=(0,8))
        act=tk.Frame(parent,bg=BG); act.pack(fill="x",padx=8,pady=(2,6))
        self.run_btn=mk_btn(act,"▶   Run Simulation",self._run,bg=GREEN,fg="white",font=("Segoe UI",11,"bold") if _W else ("Helvetica",11,"bold"),padx=20,pady=8)
        self.run_btn.pack(side="left",padx=(0,6))
        self.stop_btn=mk_btn(act,"■  Stop",self._stop,bg=RED,fg="white",padx=14,pady=8); self.stop_btn.configure(state="disabled"); self.stop_btn.pack(side="left",padx=(0,6))
        mk_btn(act,"✕  Clear",self._clear,fg=FG2,padx=12,pady=8).pack(side="left")
        self.status_var=tk.StringVar(value="Ready"); self.status_dot=tk.Label(act,text="●",font=F_BOLD,bg=BG,fg=FG2); self.status_dot.pack(side="right",padx=(0,2))
        tk.Label(act,textvariable=self.status_var,font=F_UI,bg=BG,fg=FG2).pack(side="right")
        vrml_fr=tk.Frame(parent,bg=BG2); vrml_fr.pack(fill="x",padx=8,pady=(0,4))
        vrml_hdr=tk.Frame(vrml_fr,bg=BG2); vrml_hdr.pack(fill="x",padx=8,pady=(6,2))
        tk.Label(vrml_hdr,text="VRML OUTPUT",font=F_BOLD,bg=BG2,fg=ACCENT).pack(side="left")
        mk_btn(vrml_hdr,"⟳ Refresh",self._vrml_refresh,fg=FG2,padx=8,pady=2).pack(side="right")
        mk_btn(vrml_hdr,"📂 Folder",self._open_vrml_folder,fg=FG2,padx=8,pady=2).pack(side="right",padx=(0,4))
        mk_btn(vrml_hdr,"⬡ Open" if VTK_OK else "▶ Open",self._open_vrml,fg=ACCENT,padx=8,pady=2).pack(side="right",padx=(0,4))
        lb_fr=tk.Frame(vrml_fr,bg=BG2); lb_fr.pack(fill="x",padx=8,pady=(2,6))
        sb=tk.Scrollbar(lb_fr,orient="vertical",bg=BG3,troughcolor=BG2,relief="flat",width=10)
        self.vrml_lb=tk.Listbox(lb_fr,height=4,bg="#0d1117",fg=FG,font=F_MONO,selectbackground=BG3,selectforeground=ACCENT,relief="flat",highlightthickness=0,activestyle="none",yscrollcommand=sb.set)
        sb.config(command=self.vrml_lb.yview); self.vrml_lb.pack(side="left",fill="x",expand=True); sb.pack(side="right",fill="y")
        self.vrml_lb.bind("<Double-Button-1>",lambda _: self._open_vrml())
        log_fr=tk.Frame(parent,bg=BG2); log_fr.pack(fill="both",expand=True,padx=8,pady=(0,8))
        tk.Label(log_fr,text="SIMULATION LOG",font=F_BOLD,bg=BG2,fg=ACCENT,padx=8,pady=6).pack(anchor="w")
        self.log=scrolledtext.ScrolledText(log_fr,bg="#0d1117",fg=FG,font=F_MONO,relief="flat",wrap="word",state="disabled",padx=10,pady=8,highlightthickness=0)
        self.log.pack(fill="both",expand=True,padx=8,pady=(0,8))
        for tag,col in [("cmd",FG2),("info",ACCENT),("ok",GREEN),("warn",YELLOW),("error",RED),("photon","#e8b4f8")]:
            self.log.tag_config(tag,foreground=col)

    def _traces(self):
        self._barrier_widgets=[]
        for v in [self.v_nuclide,self.v_kvp,self.v_al,self.v_cu,self.v_kbins,self.v_barrier,self.v_thick,self.v_phantom,
                  self.v_depth,self.v_det_sx,self.v_det_sy,self.v_det_sz,
                  self.v_angle,self.v_angles,self.v_mode,self.v_n,self.v_threads,self.v_jobs,
                  self.v_unc,self.v_coneang,self.v_outdir,self.v_vistype,self.v_test,self.v_auto,self.v_split,
                  self.v_nocone,self.v_verbose,self.v_dose,self.v_uncout,self.v_vis,
                  self.v_countphot,self.v_sweep_to_cvl,self.v_vrml_viewer,
                  self.v_sp_shape,self.v_sp_rx,self.v_sp_ry,self.v_sp_rz,self.v_sp_mat,
                  self.v_sp_ox,self.v_sp_oy,self.v_sp_oz]:
            v.trace_add("write",lambda *_: self._refresh())

    def _build_cmd(self):
        a=[sys.executable,str(SIM_SCRIPT)]
        src=self.v_src.get(); a+=["--source-type",src]
        if src=="nuclide": a+=["--nuclide",self.v_nuclide.get()]
        else: a+=["--kvp",self.v_kvp.get().strip() or "120","--al-filter",self.v_al.get().strip() or "2.5","--cu-filter",self.v_cu.get().strip() or "0.0","--kv-bins",self.v_kbins.get().strip() or "128"]
        mode=self.v_mode.get()
        flags={"sweep":["--sweep"],"angle_sweep":["--angle-sweep"],"nuclide_sweep":["--nuclide-sweep"],"reference":["--reference"],"all":["--all"]}
        a+=flags.get(mode,[])
        if mode=="angle_sweep":
            ang_str=self.v_angles.get().strip()
            if ang_str and ang_str!="0 15 30 45 60": a+=["--angles"]+ang_str.split()
        if mode not in MODES_NO_BARRIER: a+=["--barrier",self.v_barrier.get(),"--thickness",self.v_thick.get().strip() or "5.0"]
        a+=["--phantom-material",self._phantom_val()]
        det=self.v_det.get()
        if det=="custom":
            d=self.v_depth.get().strip()
            if d: a+=["--detector-depth",d]
        elif det!="original": a+=["--detector-preset",det]
        for flag,var in [("--detector-size-x",self.v_det_sx),("--detector-size-y",self.v_det_sy),("--detector-size-z",self.v_det_sz)]:
            val=var.get().strip()
            if val:
                try: float(val); a+=[flag,val]
                except: pass
        try: ang_f=float(self.v_angle.get())
        except: ang_f=0.0
        if abs(ang_f)>0.001: a+=["--angle",self.v_angle.get().strip()]
        if self.v_test.get(): a.append("--test")
        else:
            n=self.v_n.get().strip()
            if n: a+=["--n",n]
        if self.v_auto.get(): a.append("--auto")
        else:
            t=self.v_threads.get().strip(); j=self.v_jobs.get().strip()
            if t and t!="1": a+=["--threads",t]
            if j and j!="1": a+=["--jobs",j]
        ug=self.v_unc.get().strip()
        if ug and ug!="0.02": a+=["--unc-goal",ug]
        if self.v_split.get(): a.append("--split")
        if self.v_nocone.get(): a.append("--no-cone")
        ca=self.v_coneang.get().strip()
        if ca: a+=["--cone-angle-deg",ca]
        if self.v_sweep_to_cvl.get() and mode=="nuclide_sweep": a.append("--sweep-to-cvl")
        if self.v_verbose.get(): a.append("--verbose")
        out=self.v_outdir.get().strip()
        if out and out!="output": a+=["--output",out]
        if self.v_dose.get(): a.append("--dose")
        if self.v_uncout.get(): a.append("--uncertainty")
        if self.v_vis.get(): a+=["--vis","--vis-type",self.v_vistype.get()]
        if self.v_countphot.get(): a.append("--count-photons")
        sp_shape=self.v_sp_shape.get()
        if sp_shape!="none":
            a+=["--source-phantom-shape",sp_shape,"--source-phantom-rx",self.v_sp_rx.get().strip() or "100.0"]
            if sp_shape=="ellipsoid": a+=["--source-phantom-ry",self.v_sp_ry.get().strip() or "70.0","--source-phantom-rz",self.v_sp_rz.get().strip() or "100.0"]
            try: sp_mat=PHANTOM_VALUES[PHANTOM_DISPLAY.index(self.v_sp_mat.get())]
            except: sp_mat="G4_WATER"
            a+=["--source-phantom-material",sp_mat]
            for flag,var in [("--source-phantom-ox",self.v_sp_ox),("--source-phantom-oy",self.v_sp_oy),("--source-phantom-oz",self.v_sp_oz)]:
                val=var.get().strip() or "0.0"
                try:
                    if abs(float(val))>0.001: a+=[flag,val]
                except: pass
        return a

    def _refresh(self,*_):
        try:
            cmd=self._build_cmd(); parts=[]; i=0
            while i<len(cmd):
                tok=cmd[i]
                if tok.startswith("--") and i+1<len(cmd) and not cmd[i+1].startswith("--"): parts.append(f"{tok} {cmd[i+1]}"); i+=2
                else: parts.append(tok); i+=1
            display=" \\\n    ".join(parts)
        except Exception as e: display=f"(error: {e})"
        self.cmd_box.configure(state="normal"); self.cmd_box.delete("1.0","end"); self.cmd_box.insert("end",display); self.cmd_box.configure(state="disabled")

    def _browse(self):
        d=filedialog.askdirectory(title="Select output directory")
        if d: self.v_outdir.set(d); self._vrml_refresh()
    def _browse_viewer(self):
        p=filedialog.askopenfilename(title="Select VRML viewer",filetypes=[("Executable","*.exe *.bat *.sh *"),("All","*.*")])
        if p: self.v_vrml_viewer.set(p)
    def _copy(self):
        try: self.clipboard_clear(); self.clipboard_append(" \\\n  ".join(self._build_cmd()))
        except: pass
    def _vrml_refresh(self,*_):
        self.vrml_lb.delete(0,"end"); out=Path(self.v_outdir.get().strip() or "output")
        if not out.exists(): return
        for f in sorted(out.rglob("*.wrl"),key=lambda p: p.stat().st_mtime,reverse=True):
            try: label=str(f.relative_to(out))
            except: label=str(f)
            self.vrml_lb.insert("end",label)
        self._vrml_base=out
    def _open_vrml(self,*_):
        sel=self.vrml_lb.curselection()
        if not sel: messagebox.showinfo("No file","Select a.wrl file."); return
        target=getattr(self,"_vrml_base",Path("."))/self.vrml_lb.get(sel[0])
        if not target.exists(): messagebox.showerror("Not found",str(target)); return
        if VTK_OK: VRMLViewerWindow(self,target); return
        viewer=self.v_vrml_viewer.get().strip()
        try:
            if viewer: subprocess.Popen([viewer,str(target)])
            elif _W: os.startfile(str(target))
            elif sys.platform=="darwin": subprocess.Popen(["open",str(target)])
            else: subprocess.Popen(["xdg-open",str(target)])
        except Exception as exc: messagebox.showerror("Error",str(exc))
    def _open_vrml_folder(self):
        out=Path(self.v_outdir.get().strip() or "output").resolve()
        if not out.exists(): return
        try:
            if _W: os.startfile(str(out))
            elif sys.platform=="darwin": subprocess.Popen(["open",str(out)])
            else: subprocess.Popen(["xdg-open",str(out)])
        except: pass
    def _show_spectrum(self):
        if self.v_src.get()!="xray": messagebox.showinfo("Source","Switch to X-ray first."); return
        cmd=[sys.executable,str(SIM_SCRIPT),"--source-type","xray","--kvp",self.v_kvp.get().strip() or "120",
             "--al-filter",self.v_al.get().strip() or "2.5","--cu-filter",self.v_cu.get().strip() or "0.0",
             "--kv-bins",self.v_kbins.get().strip() or "128","--output",self.v_outdir.get().strip() or "output","--show-spectrum"]
        self._emit("Launching spectrum preview…","info")
        def _r(): env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"; subprocess.run(cmd,env=env)
        threading.Thread(target=_r,daemon=True).start()

    def _run(self):
        if self._proc and self._proc.poll() is None: messagebox.showwarning("Busy","Already running."); return
        if not SIM_SCRIPT.exists(): messagebox.showerror("Not found",str(SIM_SCRIPT)); return
        cmd=self._build_cmd()
        self._emit("="*66,"cmd"); self._emit(" ".join(cmd),"cmd"); self._emit("="*66,"cmd")
        self.run_btn.configure(state="disabled"); self.stop_btn.configure(state="normal"); self._setstatus("Running…",ACCENT)
        def _w():
            try:
                env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"
                self._proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding="utf-8",env=env)
                for line in self._proc.stdout: self._q.put(("line",line.rstrip()))
                self._proc.wait(); self._q.put(("done",self._proc.returncode))
            except Exception as exc: self._q.put(("exc",str(exc)))
        threading.Thread(target=_w,daemon=True).start()

    def _stop(self):
        if self._proc and self._proc.poll() is None: self._proc.terminate(); self._emit("Terminated.","warn")
        self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled"); self._setstatus("Stopped",YELLOW)

    def _poll(self):
        try:
            while True:
                kind,val=self._q.get_nowait()
                if kind=="line":
                    tag=None
                    if "Photons at phantom" in val or "photon_count" in val.lower(): tag="photon"
                    elif any(x in val for x in ("complete","done"," OK")): tag="ok"
                    elif any(x in val for x in ("WARNING","warning")): tag="warn"
                    elif any(x in val for x in ("Error","Traceback","exit")): tag="error"
                    elif val.lstrip().startswith("i  ") or "INFO" in val: tag="info"
                    self._emit(val,tag)
                elif kind=="done":
                    self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
                    if val==0: self._emit("Simulation complete.","ok"); self._setstatus("Complete",GREEN); self._vrml_refresh()
                    else: self._emit(f"Exited with code {val}.","error"); self._setstatus(f"Failed (exit {val})",RED)
                elif kind=="exc":
                    self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
                    self._emit(f"Exception: {val}","error"); self._setstatus("Error",RED)
        except queue.Empty: pass
        self.after(80,self._poll)

    def _emit(self,text,tag=None):
        self.log.configure(state="normal"); self.log.insert("end",text+"\n",tag or ""); self.log.see("end"); self.log.configure(state="disabled")
    def _clear(self):
        self.log.configure(state="normal"); self.log.delete("1.0","end"); self.log.configure(state="disabled"); self._setstatus("Ready",FG2)
    def _setstatus(self,text,color=FG2):
        self.status_var.set(text); self.status_dot.configure(fg=color)


if __name__=="__main__":
    app=GateArcher(); app.mainloop()