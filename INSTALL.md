# ShieldLab Installation Guide

This guide provides detailed installation instructions for ShieldLab across different platforms and use cases.

## Table of Contents
1. [Quick Install (Conda - Recommended)](#quick-install-conda---recommended)
2. [Manual Conda Setup](#manual-conda-setup)
3. [pip Installation](#pip-installation)
4. [Platform-Specific Notes](#platform-specific-notes)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Quick Install (Conda - Recommended)

**This is the easiest and most reliable method.**

### Step 1: Install Miniconda (if not already installed)

**Windows:**
- Download from https://docs.conda.io/en/latest/miniconda.html
- Run the installer
- Check "Add Anaconda to my PATH environment variable" (optional but convenient)

**macOS:**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab
```

### Step 3: Create Environment

```bash
conda env create -f environment.yml
```

This command:
- Creates a new environment named "shieldlab"
- Installs Python 3.10
- Installs all required dependencies (numpy, scipy, matplotlib, opengate, SimpleITK)

### Step 4: Activate Environment

```bash
conda activate shieldlab
```

**That's it!** You're ready to use ShieldLab.

---

## Manual Conda Setup

If you prefer to set up the environment step-by-step:

### Create Environment

```bash
conda create -n shieldlab python=3.10
conda activate shieldlab
```

### Install Core Dependencies from Conda

```bash
conda install -c conda-forge numpy scipy matplotlib
```

### Install Physics Packages via pip

```bash
pip install opengate SimpleITK
```

### Optional: Install GUI Support

**Already included in most Python installations**, but if needed:

**Linux:**
```bash
sudo apt-get install python3-tk
```

**macOS/Windows:**
Usually included with Python. If missing, reinstall Python with tkinter support.

### Optional: Install VTK for 3D Visualization

```bash
conda install -c conda-forge vtk
pip install pillow
```

---

## pip Installation

**For advanced users who prefer pip-only installation.**

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from Repository

```bash
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab
pip install -r requirements.txt
```

### Install as Package

To make ShieldLab available system-wide:

```bash
pip install -e .
```

This creates command-line tools:
- `shieldlab-sim` (runs shieldLabSim.py)
- `shieldlab-analyze` (runs shieldLabAnalyze.py)
- `shieldlab-gui` (runs shieldLabGUI.py)

---

## Platform-Specific Notes

### Windows

**Recommended Setup:**
1. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
2. Open "Anaconda Prompt" (not regular cmd.exe)
3. Follow Quick Install steps above

**Common Issues:**
- If `conda` command not found: Add Miniconda to PATH or use Anaconda Prompt
- UTF-8 encoding: Already handled in scripts via `PYTHONIOENCODING=utf-8`
- Qt visualization: Not available in Windows opengate wheels (use matplotlib instead)

### macOS

**Intel Macs:**
- Follow Quick Install steps normally

**Apple Silicon (M1/M2):**
```bash
# Use Rosetta for compatibility
CONDA_SUBDIR=osx-64 conda env create -f environment.yml
conda activate shieldlab
conda config --env --set subdir osx-64
```

**Alternative for M1/M2:**
```bash
conda create -n shieldlab python=3.10
conda activate shieldlab
ARCHFLAGS="-arch arm64" pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

**Install system dependencies first:**

```bash
sudo apt-get update
sudo apt-get install python3-tk python3-dev build-essential
```

**Then proceed with conda installation:**

```bash
conda env create -f environment.yml
conda activate shieldlab
```

**For clusters/HPC:**
- Load conda module: `module load miniconda3` (or equivalent)
- Follow Quick Install steps
- Request adequate memory (~1 GB per job) when running simulations

---

## Verification

### Check Installation

```bash
conda activate shieldlab
python -c "import opengate; print('opengate version:', opengate.__version__)"
python -c "import numpy, scipy, matplotlib, SimpleITK; print('All dependencies OK')"
```

### Run Test Simulation

```bash
python shieldLabSim.py --test --nuclide F18 --barrier Lead --thickness 1.0
```

Expected output:
- Simulation runs for ~30 seconds
- Creates `output/F18_Lead_0deg/` directory
- Generates `.mhd` and `.raw` files

### Test GUI (Optional)

```bash
python shieldLabGUI.py
```

Should open a window with three tabs: Simulate, Analyze, Archer Fit.

### Test Analysis

```bash
python shieldLabAnalyze.py output/F18_Lead_0deg
```

Should display transmission curve and Archer fit parameters.

---

## Troubleshooting

### "conda: command not found"

**Solution:**
- Windows: Use "Anaconda Prompt" instead of cmd.exe
- Linux/macOS: Add conda to PATH or restart terminal after installation

### "No module named 'opengate'"

**Solution:**
```bash
conda activate shieldlab
pip install opengate --upgrade
```

### "No module named 'tkinter'" (GUI won't start)

**Linux:**
```bash
sudo apt-get install python3-tk
```

**macOS/Windows:**
Reinstall Python with tkinter support, or use the command-line interface instead.

### Simulation runs but produces no output

**Check:**
1. Output directory exists and is writable
2. You have enough disk space (~100 MB per run)
3. Python has write permissions in current directory

### opengate installation fails on Apple Silicon

**Solution:**
Use Rosetta compatibility mode:
```bash
conda create -n shieldlab
conda activate shieldlab
conda config --env --set subdir osx-64
conda install python=3.10
pip install opengate
```

### "ImportError: DLL load failed" (Windows)

**Solution:**
Install Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Memory errors during simulation

**Solution:**
- Reduce number of primaries: use `--test` flag
- Reduce parallel jobs if running batch simulations
- Increase system RAM or use HPC cluster

---

## Updating ShieldLab

### Update from Git

```bash
cd shieldlab
git pull origin main
```

### Update Dependencies

```bash
conda activate shieldlab
conda update --all
pip install --upgrade opengate SimpleITK
```

### Recreate Environment (if major changes)

```bash
conda deactivate
conda env remove -n shieldlab
conda env create -f environment.yml
conda activate shieldlab
```

---

## Uninstallation

### Remove Conda Environment

```bash
conda deactivate
conda env remove -n shieldlab
```

### Remove Repository

```bash
cd ..
rm -rf shieldlab
```

---

## Next Steps

After successful installation:

1. **Read USAGE.md** for command-line examples
2. **Run example simulations** to familiarize yourself
3. **Try the GUI** for interactive workflow
4. **Check CONTRIBUTING.md** if you want to contribute

For questions or issues, open an issue on GitHub or check existing issues for solutions.

---

## System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 1 GB for installation + space for output data
- OS: Windows 10, macOS 10.14+, Linux (Ubuntu 18.04+)

**Recommended:**
- CPU: 4+ cores for parallel simulations
- RAM: 8+ GB
- Disk: SSD with 10+ GB free space
- OS: Recent version of Windows 10/11, macOS, or Linux

**For Production:**
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: Fast SSD with 100+ GB
- Consider HPC cluster for large batch jobs
