🧠 MFFLN-EVT: A Lightweight Open-Set Domain Adaptive Network for Cross-Domain Unknown Fault Diagnosis of Chillers

Authors:
Xuejin Gao<sup>a,c</sup>, Shouqi Wang<sup>a,c,*</sup>, Yue Liu<sup>b</sup>, Huayun Han<sup>a,c</sup>, Huihui Gao<sup>a,c</sup>, Yongsheng Qi<sup>d</sup>

Affiliations:

<sup>a</sup> School of Information Science and Technology, Beijing University of Technology, Beijing 100124, China

<sup>b</sup> Key Laboratory of Shallow Geothermal Energy, Ministry of Natural Resources of the People’s Republic of China, Beijing 100195, China

<sup>c</sup> Engineering Research Center of Digital Community, Ministry of Education, Beijing 100124, China

<sup>d</sup> School of Electric Power, Inner Mongolia University of Technology, Hohhot, Inner Mongolia 010051, China

🌟 Overview

MFFLN-EVT is an open-set domain adaptation framework designed for cross-domain unknown fault diagnosis of chiller systems.
It integrates a Multi-level Feature Fusion Lightweight Network (MFFLN) with Extreme Value Theory (EVT) to achieve high diagnostic accuracy and efficient deployment on resource-constrained edge devices.

🚀 Motivation

In real-world chiller operation:

Strong coupling among system components leads to complex, correlated data distributions.

This blurs fault boundaries and increases the difficulty of identifying unknown faults.

Existing deep learning models are computationally expensive, limiting their edge deployment.

To address these challenges, MFFLN-EVT provides:
✅ Lightweight architecture for efficient inference
✅ Open-set adaptability for unseen fault detection
✅ Pseudo-label weighted adversarial learning for robust domain alignment
✅ EVT-based decision boundary modeling for identifying unknown classes

🧩 Methodology

Data Transformation:
Convert one-dimensional sequential chiller data into grayscale images to enhance spatial structural representation.

Multi-level Feature Fusion Lightweight Network (MFFLN):
Extracts multi-scale local and global features efficiently through depthwise separable convolution and multi-branch fusion.

Adversarial Domain Adaptation:
Aligns source and target feature distributions using pseudo-label weighted adversarial training, improving classification robustness under domain shifts.

Extreme Value Theory (EVT):
Models feature distributions near decision boundaries to detect unknown fault samples beyond the known class space.

📊 Experimental Results

Datasets: Two independent chiller datasets

Average Accuracy:

Dataset I: 88.28%

Dataset II: 88.22%

Performance: Outperforms existing domain adaptation and fault diagnosis baselines, demonstrating superior open-set adaptability and engineering applicability.

