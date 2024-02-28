# Official code for Expand-and-Quantize : Unsupervised semantic segmentation using high-dimensional space and product quantization
## Accept for AAAI 2024 https://arxiv.org/abs/2312.07342

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
<img width="377" alt="스크린샷 2023-03-30 오후 11 49 44" src="https://user-images.githubusercontent.com/49435880/228875855-ff942180-dc31-45c2-944d-6832302208d5.png">
<img width="380" alt="스크린샷 2023-03-30 오후 11 49 50" src="https://user-images.githubusercontent.com/49435880/228875881-3bbae9ae-2047-45bc-9c9b-f00772a6bfb5.png">


### Analysis
<img width="788" alt="스크린샷 2023-03-30 오후 11 50 17" src="https://user-images.githubusercontent.com/49435880/228876043-e298a4a1-ff1b-444c-a3eb-1045f75e706b.png">

Analyses on EQUSS. (a) Per-class entropy of EQUSS and STEGO [21] (b) Frequency of each codeword to be selected, along with its corresponding codebook (c) Relationship between entropy and accuracy. Yellow points with particularly low accuracy correspond to the most infrequent classes (0.1% and 0.2%) (d) Inter-class distance between the combination of codewords. The darker the color, the closer the distance.
