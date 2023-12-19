# Boundary-Patch-Refinement-Paper-Implementation-

Boundary Patch Refinement (BPR) is a post-processing framework that improves the quality of instance segmentation boundaries. It involves extracting and refining small boundary patches along the predicted instance boundaries.
![BPR explain](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*AiF7oNQaBwGQB_e73n0Kyw.png)
The process of Boundary Patch Refinement for Instance Segmentation involves the following steps: 
1. Concatenate the image patch and mask patch together
2. Feed the concatenated image and mask patch into a binary segmentation network
3. Perform binary segmentation to refine the coarse boundaries
4. Reassemble the refined boundary patches into a compact instance-level mask

The BPR framework has shown significant improvements in the segmentation of the Mask R-CNN model. 


This repository provides a Python implementation of the Boundary Patch Refinement (BPR) algorithm for image segmentation, as described in the research paper: [https://arxiv.org/abs/2104.05239]. BPR refines blurry object boundaries in instance segmentation masks, improving accuracy and performance.

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

### Additional Sections
#### Challenge:
Computational Cost: BPR involves extracting and refining small boundary patches, adding processing steps, and increasing computational load. 

Learning Bias: Refining small patches around boundaries can introduce a bias towards focusing on details at the expense of larger contextual information. 

Sensitivity to Training Data: The effectiveness of BPR heavily depends on the quality and diversity of the training data. Datasets with poor boundary annotations or limited object sizes might not translate well to real-world scenarios.

Known limitations: 
1. It slows down the segmentation inference time.

Future work: 
1. we can implement this in the segmentation architecture itself.
2. reduce inference time

Citation: 
```
Tang, C., Chen, H., Li, X., Li, J., Zhang, Z., & Hu, X. (2021).
Look Closer to Segment Better:
Boundary Patch Refinement for Instance Segmentation. ArXiv. /abs/2104.05239
```
