### Ocean Drift Diffusion

A lightweight spatiotemporal diffusion model for predicting ocean currents and plastic drift






### Overview

This repository implements a spatiotemporal diffusion model (~450 lines of PyTorch) to forecast ocean current trajectories and plastic drift.

✅ Compact and lightweight

✅ Works directly on grid-based current data

✅ Implements a denoising diffusion probabilistic model (DDPM-style)

✅ Predicts drift paths with simple evaluation metrics

📂 Project Structure
Ocean-Drift-Diffusion/
│── data/              # Ocean current datasets
│── models/            # Diffusion model implementations
│── results/           # Generated drift predictions
│── utils/             # Data loading + preprocessing
│── train.py           # Training loop
│── predict.py         # Inference + visualization
│── requirements.txt   # Dependencies
│── README.md          # Project documentation

⚙️ Installation
git clone https://github.com/yashthakur2006/Ocean-Drift-Diffusion.git
cd Ocean-Drift-Diffusion
pip install -r requirements.txt

🏃 Usage
Training
python train.py --epochs 50 --lr 1e-4

Prediction
python predict.py --input data/sample_currents.npy

📊 Results

Example prediction of ocean plastic drift:

📑 Citation

If you find this work useful, please cite:

@article{thakur2025oceandrift,
  title={Ocean Drift Diffusion: A Lightweight Spatiotemporal Diffusion Model for Predicting Ocean Currents},
  author={Thakur, Yash},
  journal={arXiv preprint arXiv:pending},
  year={2025}
}

⭐ Support

If you like this project, please consider starring ⭐ the repo to support ongoing research.
