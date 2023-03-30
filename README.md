# Expand-and-Quantize : Unsupervised semantic segmentation using high-dimensional space and product quantization
## Submission to ICCV 2023

### Abstract
<img width="640" alt="스크린샷 2023-03-30 오후 11 46 03" src="https://user-images.githubusercontent.com/49435880/228874578-865030fb-d71d-472a-9843-666d1383516e.png">

Unsupervised semantic segmentation (USS) aims to discover and recognize meaningful categories without any labels. 
For a successful USS, two key abilities are required: 1) information compression and 2) clustering capability.
Previous methods have relied on feature dimension reduction for information compression, however, this approach may hinder the process of clustering.
In this paper, we propose a novel USS framework called Expand-and-Quantize Unsupervised Semantic Segmentation (EQUSS), which combines the benefits of high-dimensional spaces for better clustering and product quantization for effective information compression.
Our extensive experiments demonstrate that EQUSS achieves state-of-the-art results on three standard benchmarks.
In addition, we analyze the entropy of USS features, which is the first step towards understanding USS from the perspective of information theory.

### Overall Framework
<img width="979" alt="스크린샷 2023-03-30 오후 11 48 03" src="https://user-images.githubusercontent.com/49435880/228875143-85142d89-2345-469c-996e-6db5fc1f436c.png">

The expansion head E first expands the feature extracted from the backbone into high-dimensional spaces. Then, the quantization head Q applies product quantization to generate the quantized output.
Finally, this output is used for clustering and linear probing during the evaluation. 

### Experiment


### Analysis
