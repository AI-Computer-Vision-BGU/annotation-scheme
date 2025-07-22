# Annotation Scheme for Object Detection & Segmentation Tasks

This repo contains the semi-automatic video annotation pipeline used in our [Maintenance Datasets] paper.


[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/abs/xxx)
[![Project](https://img.shields.io/badge/Project-GitHub-brightgreen.svg)](https://github.com/YourOrg/Annotation-Scheme)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://youtu.be/your-video)



## Installation


* First, clone the repo with its submodules
```bash
git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
cd Annotation-Scheme
pip install -r requirements.txt

```

* Then, install SAM requirements. More Info [Here](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
```bash
cd segmentanything
pip install -e .
.checkpoints/download_ckpts.sh    # download the weights for the model
cd ..
```

* Make sure [segment-anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file) is installed also:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```


## Getting Started
To run the script:
```bash
python AnnotationScheme/annotation_scheme.py
```
This launches an **interactive wizard** that lets you choose **one of three workflows**:

#### 1&nbsp;·&nbsp;Annotate a *single* video  
* **Input** any `.mp4 / .mov` file  
--

#### 2&nbsp;·&nbsp;Annotate an *entire directory* of class-organised videos
* Input (root path)
<pre> root/
  │ 
  ├── object1/ 
  │ ├── clip_001.mp4 
  │ └── … 
  └── object2/ 
  └── … 
</pre>

The **Output** of both flows as follow
   <pre>
    res/
    ├── Images                # Contains the images
    │ ├── I1.jpg
    │ └── … 
    └── Annotations.json      # Contains the annotation images (COCO format)

  </pre>

  ---


## Stat
```bash
TODO
```

## Citations
```bash
TODO  
```

