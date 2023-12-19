# Boundary-Patch-Refinement-Paper-Implementation-
Boundary Patch Refinement (Paper Implementation)

This repository provides a Python implementation of the Boundary Patch Refinement (BPR) algorithm for image segmentation, as described in the research paper: [https://arxiv.org/abs/2104.05239]. BPR refines blurry object boundaries in instance segmentation masks, leading to improved accuracy and performance.

Features
Implements the BPR algorithm for boundary refinement.
Supports training and applying BPR on various datasets.
Provides modular and well-documented code for easy understanding and modification.
Includes unit and integration tests for code reliability.
Installation
Clone this repository:
> git clone https://github.com/rushidarge/Boundary-Patch-Refinement-Paper-Implementation-.git

Install required dependencies:
> pip install -r requirements.txt

Documentation
A detailed overview of the BPR algorithm and its implementation is provided in the docs/ folder.
The code itself is extensively documented for easy comprehension.

License
This repository is licensed under the MIT License. See the LICENSE file for details.

Additional Sections:
Known limitations: 
1. Its increase the inference time of segmentation.

Future work: 
1. we can implement this in segmentation architecture itself.
2. reduce inference time

Citation: 
```
Tang, C., Chen, H., Li, X., Li, J., Zhang, Z., & Hu, X. (2021). Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation. ArXiv. /abs/2104.05239
```
