# Annotation Scheme for Object Detection & Segmentation Tasks

This repo contains the semi-automatic video annotation pipeline used in our [Maintenance Datasets] paper.


[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/abs/xxx)
[![Project](https://img.shields.io/badge/Project-GitHub-brightgreen.svg)](https://github.com/YourOrg/Annotation-Scheme)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://youtu.be/your-video)



## Installation

Tested on Python 3.9 + PyTorch 2.1, Ubuntu 22.04 / macOS 14, CUDA 11.8.

```bash
# 1 Clone *with* submodules so SAM 2 arrives too
git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
cd Annotation-Scheme

# 2 Core Python deps (GUI / YOLO / OpenCV / SAM2 helpers)
pip install -r requirements.txt

# 3 Build SAM 2 in editable mode (needed if you’ll use the CUDA ops)
cd segmentanything
pip install -e .
cd ..

# 4 (Temporary) install Meta’s original SAM package for the legacy helpers
pip install git+https://github.com/facebookresearch/segment-anything.git
