# ShieldLab

Monte Carlo radiation shielding simulation toolchain for nuclear medicine and diagnostic radiology applications.

## Overview

ShieldLab provides automated broad-beam transmission equation parameter estimation using GATE 10 (opengate) Monte Carlo simulations. The toolchain implements the methodology described in Oumano et al. 2025 (JACMP 26:e70084) with empirical alpha anchoring and orthogonal distance regression (ODR) fitting.

### Key Features

- **Polyenergetic X-ray tube sources** with customizable kVp and filtration
- **Nuclear medicine radionuclide sources** (F-18, Tc-99m, Lu-177, I-131, and more)
- **Oblique incidence geometry** with configurable angles
- **Multiple barrier materials**: Lead, Steel, Concrete (normal/lightweight), Glass, Gypsum
- **Automated Archer parameter fitting** using ODR with empirical alpha anchoring
- **Interactive GUI** for simulation control and analysis
- **Uncertainty-based early stopping** for efficient computation
- **Photon splitting** for variance reduction

## Supported Radionuclides

F-18, Tc-99m, Lu-177, I-131, Zr-89, Cu-64, Ga-68, In-111, I-123, I-124, Rb-82, Ac-225, At-211, Y-90, Xe-133

## Supported Barrier Materials

Lead, Steel, Normal-weight Concrete, Lightweight Concrete, Glass, Gypsum

## Installation

**Quick Start (Recommended):**

```bash
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab
conda env create -f environment.yml
conda activate shieldlab
```

**For detailed installation instructions, platform-specific notes, and troubleshooting, see [INSTALL.md](INSTALL.md).**

### Prerequisites

- **Miniconda or Anaconda** (strongly recommended)
- Windows, Linux, or macOS
- Python 3.9+ (Python 3.10 recommended)

### Method 1: Conda Environment (Recommended)

The easiest way to set up ShieldLab with all dependencies:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate shieldlab
```

That's it! The environment file handles all dependencies including opengate.

### Method 2: Manual Conda Setup

If you prefer to create the environment manually:

```bash
# Create new environment
conda create -n shieldlab python=3.10
conda activate shieldlab

# Install conda packages
conda install numpy scipy matplotlib

# Install pip packages
pip install opengate SimpleITK
```

### Method 3: pip Only (Advanced Users)

If you're not using conda:

```bash
pip install -r requirements.txt
```

**Note:** GUI support requires tkinter:
- **Linux**: `sudo apt-get install python3-tk`
- **macOS**: Usually included with Python
- **Windows**: Usually included with Python installer

## Usage

### Command Line Interface

#### Run a basic simulation

```bash
python shieldLabSim.py --nuclide F18 --barrier Lead --thickness 0.5 1.0 2.0 4.0
```

#### X-ray tube simulation

```bash
python shieldLabSim.py --kVp 120 --al-filter 2.5 --cu-filter 0.2 --barrier Lead --thickness 0.1 0.2 0.5 1.0
```

#### Oblique incidence

```bash
python shieldLabSim.py --nuclide Tc99m --barrier NWConcrete --thickness 50 100 200 --angle 30
```

#### Sweep multiple thicknesses

```bash
python shieldLabSim.py --nuclide I131 --barrier Lead --sweep-start 0.5 --sweep-stop 5.0 --sweep-n 10
```

#### Enable photon splitting for variance reduction

```bash
python shieldLabSim.py --nuclide Lu177 --barrier Lead --thickness 1.0 2.0 4.0 --split 10
```

### Graphical User Interface

Launch the GUI:

```bash
python shieldLabGUI.py
```

The GUI provides three tabs:
1. **Simulate 🎯**: Configure and run simulations
2. **Analyze 📊**: View dose maps and statistics
3. **Archer Fit 📈**: Perform Archer curve fitting with interactive parameter tuning

### Analysis and Fitting

Analyze simulation output and fit Archer parameters:

```bash
python shieldLabAnalyze.py output/F18_Lead_0deg
```

Interactive mode with parameter slider:

```bash
python shieldLabAnalyze.py output/F18_Lead_0deg --interactive
```

Export fitted parameters to CSV:

```bash
python shieldLabAnalyze.py output/F18_Lead_0deg --csv output/archer_params.csv
```

## Project Structure

```
shieldlab/
├── shieldLabSim.py          # Main simulation engine
├── shieldLabAnalyze.py      # Post-processing and Archer fitting
├── shieldLabGUI.py          # Tkinter GUI launcher
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LICENSE                  # License information
└── output/                  # Simulation results (created automatically)
```

## Methodology

### Archer Transmission Equation

The broad-beam transmission is modeled by the three-parameter Archer equation:

```
T(x) = ((1 + β/α) * exp(-α*x) - β/α) / (1 + γ*x)
```

where:
- `α`: asymptotic attenuation coefficient
- `β`: buildup-related parameter
- `γ`: inverse characteristic buildup distance

### Fitting Procedure

1. **Alpha determination**: Empirically anchored from the log-linear tail (T ≲ 0.1)
2. **ODR fitting**: Beta and gamma fitted simultaneously using orthogonal distance regression
3. **FVL cross-check**: Local bracketing method validates HVL/TVL/CVL estimates

### Validation

Methodology validated against:
- Oumano et al. 2025 (JACMP 26:e70084)
- NCRP 147/151 reference data
- Simpkin transmission benchmarks

## Citation

If you use ShieldLab in your research, please cite:

```
Oumano N, Jansen J, Patel R, Osei E. An open-source toolkit for shielding 
calculations in diagnostic imaging and nuclear medicine. 
J Appl Clin Med Phys. 2025;26(5):e70084. doi:10.1002/acm2.70084
```

## Advanced Options

### Uncertainty-based Early Stopping

```bash
python shieldLabSim.py --nuclide F18 --barrier Lead --thickness 1.0 --unc-goal 0.02
```

Stops simulation when relative uncertainty reaches 2% (default: 5%).

### Cone-restricted Source

```bash
python shieldLabSim.py --nuclide Tc99m --barrier Steel --thickness 5.0 --cone-half-angle 15
```

Restricts source emission to a cone (default: ~28° based on detector geometry).

### Custom Output Directory

```bash
python shieldLabSim.py --nuclide I131 --barrier Lead --thickness 2.0 --output-dir my_results
```

### Test Mode (Fast)

```bash
python shieldLabSim.py --nuclide F18 --barrier Lead --thickness 1.0 --test
```

Runs with 10M primaries instead of 2B for quick testing.

## Performance Notes

- Typical run time: 30 seconds to 5 minutes per thickness (depending on barrier and splitting)
- Memory: ~600 MB per parallel job
- Photon splitting (`--split 10`) can reduce run time by 5-10× for thick barriers
- Uncertainty-based stopping prevents over-simulation of high-transmission cases

## Limitations

- Oblique incidence: Physically defensible only for primary barriers in fixed-geometry radiographic rooms
- Not validated for CT scatter barriers per NCRP 147
- Alpha emitters (Ac-225, At-211): Shielding primarily for bremsstrahlung, not alpha range

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## License

[Specify your license here - MIT, GPL, Apache 2.0, etc.]

## Authors

Dustin Osborne, MS, DABSNM, DABR  
Medical Physicist

## Contact

[Your contact information or GitHub issues page]

## Acknowledgments

- GATE 10 / opengate development team
- Oumano et al. 2025 methodology validation
- NCRP Report 147 & 151 reference data
