# ShieldLab Repository Structure

```
shieldlab/
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions CI/CD workflow
│
├── output/                       # Simulation results (git-ignored except .gitkeep)
│   └── .gitkeep                  # Keeps directory in version control
│
├── .gitignore                    # Git ignore patterns
├── CITATION.cff                  # Academic citation metadata
├── CONTRIBUTING.md               # Contribution guidelines
├── environment.yml               # Conda environment specification
├── GIT_SETUP.md                  # Step-by-step Git setup instructions
├── INSTALL.md                    # Detailed installation guide
├── LICENSE                       # MIT License
├── QUICK_START.md                # Fast-track setup guide
├── README.md                     # Main documentation
├── STRUCTURE.md                  # This file - repository structure
├── USAGE.md                      # Detailed usage guide
├── requirements.txt              # Python dependencies (pip)
├── setup.py                      # Package installation configuration
│
├── shieldLabSim.py              # Main simulation engine (56 KB)
├── shieldLabAnalyze.py          # Post-processing & Archer fitting (65 KB)
└── shieldLabGUI.py              # Tkinter GUI launcher (70 KB)
```

## File Descriptions

### Core Python Scripts
- **shieldLabSim.py**: GATE 10 Monte Carlo simulation engine with polyenergetic X-ray and nuclide sources
- **shieldLabAnalyze.py**: ODR-based Archer parameter fitting with interactive visualization
- **shieldLabGUI.py**: Multi-tab Tkinter interface for simulation control and analysis

### Documentation
- **README.md**: Project overview, installation, quick start, and feature summary
- **INSTALL.md**: Comprehensive installation guide with platform-specific instructions
- **USAGE.md**: Comprehensive usage examples and CLI reference
- **CONTRIBUTING.md**: Contribution guidelines and development workflow
- **GIT_SETUP.md**: Step-by-step Git/GitHub setup instructions
- **QUICK_START.md**: Fast-track guide to get online quickly
- **CITATION.cff**: Standardized citation metadata for academic use

### Configuration
- **environment.yml**: Conda environment specification (recommended installation method)
- **requirements.txt**: Python package dependencies for pip installation
- **setup.py**: Package installation configuration with entry points
- **.gitignore**: Excludes output files, Python cache, IDE files, etc.
- **LICENSE**: MIT License for open-source distribution

### GitHub Integration
- **.github/workflows/test.yml**: Automated testing on push/PR (multi-OS, multi-Python)

### Output
- **output/**: Auto-created directory for simulation results (excluded from Git except structure)

## Total Repository Size
- **Code**: ~191 KB (3 Python scripts)
- **Documentation**: ~45 KB (7 documentation files)
- **Total committed**: ~236 KB (lightweight!)
- **Output data**: Not tracked (user-generated, can be GBs)

## Key Features of This Structure

✅ **Production-ready**: Includes all standard OSS project files  
✅ **Well-documented**: Multiple levels of documentation (README, USAGE, code comments)  
✅ **CI/CD ready**: GitHub Actions workflow included  
✅ **Pip installable**: setup.py enables `pip install .`  
✅ **Citable**: CITATION.cff for academic attribution  
✅ **Contributor-friendly**: Clear contribution guidelines  
✅ **Clean repository**: .gitignore prevents bloat from output files
