# 🌊 Ocean Drift Diffusion
*A Lightweight Spatiotemporal Diffusion Model for Predicting Ocean Currents and Plastic Drift*

[![Stars](https://img.shields.io/github/stars/yashthakur2006/Ocean-Drift-Diffusion?style=social)](https://github.com/yashthakur2006/Ocean-Drift-Diffusion/stargazers)
[![Forks](https://img.shields.io/github/forks/yashthakur2006/Ocean-Drift-Diffusion?style=social)](https://github.com/yashthakur2006/Ocean-Drift-Diffusion/network/members)
[![Issues](https://img.shields.io/github/issues/yashthakur2006/Ocean-Drift-Diffusion)](https://github.com/yashthakur2006/Ocean-Drift-Diffusion/issues)
[![License](https://img.shields.io/github/license/yashthakur2006/Ocean-Drift-Diffusion)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-pending-orange)](#-citation)

> **TL;DR**: ~450 lines of PyTorch implementing a DDPM-style **spatiotemporal diffusion** model to forecast **ocean currents & plastic drift** from grid data. Minimal, hackable, research-ready.

---

## 📚 Table of Contents
- [Highlights](#-highlights)
- [Abstract](#-abstract)
- [Quickstart](#-quickstart)
- [Data Format](#-data-format)
- [Configuration](#-configuration)
- [Python API](#-python-api)
- [CLI Usage](#-cli-usage)
- [Visualization](#-visualization)
- [Results & Benchmarks](#-results--benchmarks)
- [Reproducibility](#-reproducibility)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [FAQ • Troubleshooting • Acknowledgements](#-faq--troubleshooting--acknowledgements)

---

## 🔥 Highlights
- **Compact**: ~450 LOC PyTorch core; easy to read & extend.
- **Spatiotemporal diffusion**: DDPM denoising over time-indexed grids.
- **No physics priors**: Purely data-driven baseline; plug any gridded current source.
- **Metrics included**: ADE/FDE, RMSE, plus trajectory visualization.
- **Batteries included**: train/eval scripts, config system, demo mode.

---

## 🧠 Abstract
**Ocean Drift Diffusion** is a minimal, research-grade DDPM-style model for forecasting **ocean current trajectories** and **plastic drift** using gridded inputs (e.g., zonal/meridional components on a lat–lon grid). The model learns to denoise future states conditioned on historical context, offering an efficient baseline for data-driven ocean forecasting. Despite its simplicity, it yields competitive trajectory accuracy and forms a clean foundation for extensions (physics-informed losses, equivariant backbones, multi-source fusion).

---

## 🚀 Quickstart

### 1) Install
```bash
# Clone
git clone https://github.com/yashthakur2006/Ocean-Drift-Diffusion.git
cd Ocean-Drift-Diffusion

# (Recommended) Create a virtual env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# (Optional) GPU-enabled PyTorch (adjust CUDA version as needed)
# pip install torch --index-url https://download.pytorch.org/whl/cu121
