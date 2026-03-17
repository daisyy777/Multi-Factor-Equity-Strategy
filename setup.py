"""
Setup script for multi-factor equity strategy package.
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="multi-factor-equity-strategy",
    version="2.0.0",
    description="Cross-sectional multi-factor long-short equity strategy research framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="daisyy777",
    url="https://github.com/daisyy777/Multi-Factor-Equity-Strategy",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.28",
        "scipy>=1.11.0",
        "seaborn>=0.12.0",
        "pyarrow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "pytest>=7.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
