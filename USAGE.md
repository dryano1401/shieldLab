# ShieldLab Usage Guide

## Quick Start

### 1. Installation

**Using Conda (Recommended)**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab

# Create environment from file
conda env create -f environment.yml
conda activate shieldlab
```

**Manual Setup**

```bash
# Create conda environment
conda create -n shieldlab python=3.10
conda activate shieldlab

# Install dependencies
conda install numpy scipy matplotlib
pip install opengate SimpleITK
```

### 2. Run Your First Simulation

```bash
python shieldLabSim.py --nuclide F18 --barrier Lead --thickness 0.5 1.0 2.0 4.0
```

### 3. Analyze Results

```bash
python shieldLabAnalyze.py output/F18_Lead_0deg --interactive
```

---

## Command Line Examples

### Nuclear Medicine Sources

#### F-18 through Lead
```bash
python shieldLabSim.py --nuclide F18 --barrier Lead \
    --sweep-start 0.5 --sweep-stop 5.0 --sweep-n 10
```

#### Tc-99m through Concrete
```bash
python shieldLabSim.py --nuclide Tc99m --barrier NWConcrete \
    --thickness 50 100 150 200 250
```

#### Lu-177 with photon splitting
```bash
python shieldLabSim.py --nuclide Lu177 --barrier Lead \
    --thickness 1 2 4 8 --split 10
```

### X-ray Tube Sources

#### 120 kVp chest X-ray
```bash
python shieldLabSim.py --kVp 120 --al-filter 2.5 \
    --barrier Lead --thickness 0.1 0.2 0.5 1.0 2.0
```

#### 80 kVp with copper filtration
```bash
python shieldLabSim.py --kVp 80 --al-filter 2.5 --cu-filter 0.2 \
    --barrier NWConcrete --sweep-start 50 --sweep-stop 300 --sweep-n 8
```

#### Generate spectrum plot
```bash
python shieldLabSim.py --plot-spectrum --kVp 120 --al-filter 2.5 --cu-filter 0.1
```

### Oblique Incidence

#### 30-degree angle
```bash
python shieldLabSim.py --nuclide F18 --barrier Lead \
    --thickness 1 2 4 --angle 30
```

#### Sweep multiple angles
```bash
python shieldLabSim.py --nuclide Tc99m --barrier NWConcrete \
    --thickness 100 --run angle-sweep
```

### Performance Optimization

#### High uncertainty tolerance (faster)
```bash
python shieldLabSim.py --nuclide I131 --barrier Lead \
    --thickness 2.0 --unc-goal 0.05
```

#### Low uncertainty (more accurate, slower)
```bash
python shieldLabSim.py --nuclide F18 --barrier Lead \
    --thickness 0.5 --unc-goal 0.02
```

#### Photon splitting for thick barriers
```bash
python shieldLabSim.py --nuclide Lu177 --barrier Lead \
    --thickness 8.0 --split 20
```

---

## Analysis Examples

### Basic Analysis

```bash
# Analyze single run
python shieldLabAnalyze.py output/F18_Lead_0deg

# Interactive fitting with sliders
python shieldLabAnalyze.py output/F18_Lead_0deg --interactive

# Export to CSV
python shieldLabAnalyze.py output/F18_Lead_0deg --csv archer_params.csv
```

### Advanced Analysis Options

```bash
# Custom HVL range for alpha anchoring
python shieldLabAnalyze.py output/Tc99m_NWConcrete_0deg \
    --hvl-hard-min 0.8 --hvl-hard-max 1.2

# Adjust fitting bounds
python shieldLabAnalyze.py output/I131_Lead_0deg \
    --alpha-bounds 0.5 2.0 --beta-bounds 0.0 1.0
```

---

## GUI Walkthrough

### Launch GUI

```bash
python shieldLabGUI.py
```

### Simulate Tab

1. **Select source type**: Nuclide or kVp X-ray
2. **Choose barrier material**: Lead, Steel, Concrete, etc.
3. **Enter thicknesses**: Comma-separated list (e.g., `1, 2, 4, 8`)
4. **Configure options**:
   - Angle (0° = perpendicular)
   - Photon splitting factor
   - Uncertainty goal
5. **Click "Run Simulation"**

### Analyze Tab

1. **Click "Scan Output Directory"** to find completed runs
2. **Select a run** from the dropdown
3. **View dose maps** and statistics
4. **Export data** if needed

### Archer Fit Tab

1. **Select output directory** (or sync from Simulate tab)
2. **Click "Scan"** to detect available data
3. **Choose nuclide/barrier/angle combination**
4. **Click "Load & Fit"**
5. **Adjust HVL slider** if needed to refine alpha
6. **View fitted parameters** and quality metrics

---

## Batch Processing

### Run Multiple Configurations

Create a bash script (`run_batch.sh`):

```bash
#!/bin/bash

NUCLIDES=("F18" "Tc99m" "I131" "Lu177")
BARRIERS=("Lead" "Steel" "NWConcrete")

for nuc in "${NUCLIDES[@]}"; do
    for bar in "${BARRIERS[@]}"; do
        echo "Running $nuc through $bar..."
        python shieldLabSim.py --nuclide $nuc --barrier $bar \
            --sweep-start 0.5 --sweep-stop 10.0 --sweep-n 10
    done
done
```

Run:
```bash
chmod +x run_batch.sh
./run_batch.sh
```

### Analyze All Results

```bash
for dir in output/*/; do
    echo "Analyzing $dir..."
    python shieldLabAnalyze.py "$dir" --csv "${dir}/archer_params.csv"
done
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'opengate'"

```bash
conda activate shieldlab
pip install opengate
```

### "Qt platform plugin could not be initialized" (GUI issue)

On Linux:
```bash
export QT_QPA_PLATFORM=offscreen
```

Or install Qt dependencies:
```bash
sudo apt-get install python3-tk
```

### Simulations taking too long

1. Use `--test` flag for quick testing
2. Increase `--unc-goal` to 0.05 or 0.10
3. Enable photon splitting: `--split 10`
4. Use cone restriction (already enabled by default)

### All-NaN transmission values

This usually means the DoseActor geometry changed. Check that:
1. You're using the latest version of the scripts
2. The output directory contains valid `.mhd` files
3. The phantom material matches the barrier material

---

## Best Practices

### For Primary Barriers

- Use perpendicular incidence (`--angle 0`)
- Start with coarse thickness sweep, then refine
- Use `--unc-goal 0.02` for thin barriers, `0.05` for thick

### For Scatter Barriers

- X-ray tube simulations are more appropriate than nuclide point sources
- Consider realistic scatter angles
- Concrete barriers: use `NWConcrete` for most clinical facilities

### For Publication-Quality Data

1. Run full statistics: `--unc-goal 0.02`
2. Include multiple barrier materials for comparison
3. Validate against NCRP 147/151 where possible
4. Document all simulation parameters in your methods section

---

## Output Files

Each simulation creates a directory: `output/{source}_{barrier}_{angle}deg/`

Contains:
- `*.mhd` / `*.raw`: Dose maps (MetaImage format)
- `stats.txt`: Summary statistics
- `parameters.txt`: Simulation configuration
- `archer_fit.csv`: Fitted parameters (if analyzed)

---

## Next Steps

- Read the [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Check [CITATION.cff](CITATION.cff) for citation format
- Review Oumano et al. 2025 for methodology details
