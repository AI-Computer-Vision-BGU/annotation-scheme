# Annotation Scheme for Object Detection & Segmentation Tasks

This repo contains the semi-automatic video annotation pipeline used in our [Maintenance Datasets] paper.


[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/abs/xxx)
[![Project](https://img.shields.io/badge/Project-GitHub-brightgreen.svg)](https://github.com/YourOrg/Annotation-Scheme)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://youtu.be/your-video)



## Installation

Tested on Python 3.9 + PyTorch 2.1, Ubuntu 22.04 / macOS 14, CUDA 11.8.

First, clone the repo with its submodules
```bash
git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
cd Annotation-Scheme
pip install -r requirements.txt

```

Then, install SAM requirements. More Info [Here](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
```bash
cd segmentanything
pip install -e .
cd ..
```

Make sure [segment-anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file) is installed also:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```


Feel free to tweak wording or add optional environment-creation lines (`python -m venv`, `conda`, etc.) if your audience prefers those.
::contentReference[oaicite:0]{index=0}

