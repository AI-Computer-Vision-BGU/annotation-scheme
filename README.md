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
./checkpoints/download_ckpts.sh    # download the weights for the model
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
    └── Annotations.json      # Contains the annotation.json fuck you   (COCO format)

  </pre>

  ---


The first window will look like this, where you can choose to skip this video [N], or exit the program [X] keywords
![First Window](assets/init_w.png)

To continue annotating the video, press any keyword (space, enter, etc...) -- This will get you to the annotator scheme.
Here, you can choose to annotate different objects (i.e., Tool & Hands) with convenient option (1 or 2 to change object -- 'b' to change the prompt from points to bounding box (and vice versa))

![Annotator Demo](assets/annotator.gif)

SAM2 will take these initial bb/point prompts to start annotate the next 50 frames automatically. This techniques is repeated untill the last frame.
After finishing -- we will get the following preview to validate the annotations:

![Preview](assets/annotator_res.png)


![Change Class](assets/change_class.gif)


![Edit Annotation](assets/edit_bb.gif)

---


---

## 🚀  Interactive preview workflow  

### 1 · Initial decision screen  
The very first pop-up lets you bail out early:

| Key | Action |
|-----|--------|
| **N** | Skip this video and move to the next one in the folder |
| **X** | Exit the entire program |

<p align="center">
  <img src="assets/init_w.png" width="480" alt="First window – skip or exit">
</p>

---

### 2 · Annotator GUI  
Press **any other key** (Space / Enter / click) to jump into the annotator loop.

* **1** → annotate the **tool**  
* **2** → annotate the **hands**  
* **b** → toggle between **Bounding-box** mode and **Point-prompt** mode  
* Draw / click → SAM 2 uses that prompt to propagate over the next **50 frames**  

<p align="center">
  <img src="assets/annotator.gif" width="640" alt="Live annotation demo">
</p>

---

### 3 · Preview & validation  
After each propagation step the scheme replays the segment so you can accept or refine:

| Key | Action |
|-----|--------|
| **e** | Edit BB / points on the current frame |
| **n** | Change the class label |
| **q** | Quit current video (saves progress) |
| **x** | Exit the program (saves progress) |

<p align="center">
  <img src="assets/annotator_res.png" width="640" alt="Preview window">
</p>

The loop repeats until the last frame is processed.  
Accepted annotations are written to **YOLO TXT** files (BBs) and **COCO JSON** (tool + hand polygons) under `annotation_results/`.

---



## Stat
```bash
TODO
```

## Citations
```bash
TODO  
```

