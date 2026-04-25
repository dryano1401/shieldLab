from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="shieldlab",
    version="0.1.0",
    author="Dustin Osborne",
    author_email="",  # Add your email
    description="Monte Carlo radiation shielding simulation toolchain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/shieldlab",  # Update with your GitHub URL
    py_modules=["shieldLabSim", "shieldLabAnalyze", "shieldLabGUI"],
    python_requires=">=3.9",
    install_requires=[
        "opengate>=10.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "SimpleITK>=2.1.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "gui": ["pillow>=8.0.0"],
        "vtk": ["vtk>=9.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "shieldlab-sim=shieldLabSim:main",
            "shieldlab-analyze=shieldLabAnalyze:main",
            "shieldlab-gui=shieldLabGUI:main",
        ],
    },
)
