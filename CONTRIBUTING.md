# Contributing to ShieldLab

Thank you for considering contributing to ShieldLab! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Your environment (OS, Python version, opengate version)
5. Relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:

1. A clear description of the proposed feature
2. Use cases and motivation
3. Any relevant references or prior art

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Test your changes** thoroughly
4. **Update documentation** if you're adding/changing functionality
5. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Test results or validation data

## Development Setup

**Using Conda Environment File (Easiest)**

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab

# Create environment from file
conda env create -f environment.yml
conda activate shieldlab

# Run tests
python shieldLabSim.py --test --nuclide F18 --barrier Lead --thickness 1.0
```

**Manual Setup**

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/shieldlab.git
cd shieldlab

# Create conda environment
conda create -n shieldlab-dev python=3.10
conda activate shieldlab-dev

# Install dependencies
conda install numpy scipy matplotlib
pip install opengate SimpleITK

# Run tests
python shieldLabSim.py --test --nuclide F18 --barrier Lead --thickness 1.0
```

## Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and modular

## Testing

Before submitting a PR:

1. Test your changes with multiple nuclide/barrier combinations
2. Verify the GUI still launches (if GUI-related changes)
3. Ensure analysis pipeline produces valid Archer fits
4. Check for any new dependencies and update `requirements.txt`

## Validation Data

If adding new features that affect simulation physics:

- Compare against NCRP 147/151 reference data where possible
- Document any deviations from published methodologies
- Include validation plots/data in PR

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Be specific and descriptive
- Reference issues when applicable: "Fix #123: Correct oblique angle calculation"

## Questions?

Feel free to open an issue for questions or discussion!
