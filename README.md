# FusionNet: Deep Learning with Multimodal Inputs for Enhanced Issue Resolution

## Overview
FusionNet is a **multimodal deep learning model** designed to classify GitHub issue reports into **"bug"** or **"feature"** categories. Unlike traditional text-only models, FusionNet leverages **text**, **code**, and **images** from issue reports to improve classification accuracy. By combining these modalities, FusionNet achieves significant performance improvements over unimodal approaches.

---

### Problem Context:
In open-source software development, GitHub's issue tracking system allows developers to report bugs and request features. These reports contain various types of data such as text, code, and images. Traditional classification models primarily use text, missing out on the full context provided by other data types.

### Objective:
This project proposes **FusionNet**, a multimodal deep learning model that combines text, code, and image data to improve the classification of issue reports into **bug** or **feature** categories.

### Goal:
Evaluate the performance of FusionNet across different data combinations (text, image, code) and demonstrate how leveraging multiple modalities enhances classification accuracy.

---

## Dataset

The dataset used in this project can be accessed via the following link:  
[Download Dataset](https://drive.google.com/file/d/1QbxubJ9nYoogzTI9a-LsG1DhCYgq_vwA/view?usp=sharing)

---

## Related Work


| Feature              | Text-only Models         | Multimodal Models         |
|----------------------|--------------------------|---------------------------|
| **Data Used**        | Text only                | Text, Image, and Code     |
| **Context Understanding** | Limited to textual context | Leverages richer context with image and code |
| **Accuracy**         | Often lower in complex tasks | Improved accuracy, especially in complex tasks |
| **Examples**         | TF-IDF, RNNs             | FusionNet, Multimodal CNNs |

---

## The Proposed Multimodal Approach

**FusionNet** is a deep learning framework designed for multimodal classification. It integrates text, image, and code data for classifying software issue reports. The goal is to enhance issue report classification accuracy by combining multiple data sources.

### Key Stages of FusionNet:
1. **Data Pre-Processing**: Text, image, and code inputs are pre-processed individually.
2. **Feature Extraction**: Pre-processed data are passed through CNN-based channels to generate feature vectors for each modality.
3. **Fusion**: Feature vectors from all modalities are integrated through element-wise multiplication to create a unified multimodal representation.
4. **Classification**: The fused representation is classified into **"bug"** or **"feature"** using a Softmax operation.

---

## Data Pre-Processing

### Text Pre-Processing:
- Convert to lowercase (Case-Folding)
- Tokenize text
- Remove stop words and non-alphabetic tokens

### Code Pre-Processing:
- Remove comments
- Replace special tokens (`\n`, `\t`)
- Convert to lowercase
- Tokenize code

### Image Pre-Processing:
- Resize to 258x258 pixels
- Normalize pixel values to [-1, 1]

---

## Feature Extraction

### Text & Code Data:
- **Embedding Layer**: Converts data into vector representations.
- **CNN Processing**: Applies convolution, max-pooling, and fully connected layers for feature extraction.

### Image Data:
- **CNN Processing**: Uses 3 convolution layers, max-pooling, and a fully connected layer to extract features.

---

## Fusion of Feature Vectors

- **Fusion Method**: Combines features from different modalities (text, code, image) using element-wise multiplication.
- **Goal**: Integrates data into a unified representation for better learning.
- **Benefit**: Simplifies and enhances multimodal learning, making classification more accurate.

### Classification into "Bug" or "Feature":
- **Final Step**: Process the fused representation through a fully connected layer.
- **Softmax**: Produces probabilities for classification into **"Bug"** or **"Feature"**.
- **Optimization**: CrossEntropyLoss is used for model optimization.
- **Scalability**: Can be extended for multiclass classification tasks.

---

## Experimental Setup

### Datasets:
- Open-source projects from GitHub (VS Code, Kubernetes, Flutter, Roslyn).
- Issues labeled as **"Bug"** or **"Feature"** for binary classification.
- **Data Sampling**: Downsampling applied to balance class distribution (80:20 training/test split).

### Models Compared:
- **Text Only Model**: Baseline using only text data.
- **Multimodal Models**:
  - **FusionNet𝑇𝐼**: Text and Image data.
  - **FusionNet𝑇𝐶**: Text and Code data.
  - **FusionNet𝑇𝐼𝐶**: Text, Image, and Code data.

### Evaluation Metrics:
- **Precision**, **Recall**, **F1-score** calculated for each class and averaged based on class distribution.

| Project     | Total Issues | Bug   | Feature | Total (After Sampling) |
|-------------|--------------|-------|---------|------------------------|
| VS Code     | 160,218      | 28,353| 20,074  | 48,427                 |
| Kubernetes  | 115,035      | 13,059| 5,184   | 18,243                 |
| Flutter     | 118,576      | 13,037| 9,967   | 23,004                 |
| Roslyn      | 66,464       | 12,882| 3,824   | 16,706                 |

---

## Results
![Results](https://github.com/marinaredamekhael/Deep-Learning-with-Multimodal-Inputs-for-Enhanced-Issue-Resolution/blob/main/result.png)
### Overall Findings:
- **FusionNet𝑇𝐼𝐶 (Text + Image + Code)** outperformed other models across all evaluation metrics (Precision, Recall, F1-score).
- The **Text Only** model was consistently outperformed by all multimodal models.

### Key Experiments:
- **Text + Image (FusionNet𝑇𝐼)**:
  - Improved Precision (up to **11.15%**), Recall (up to **9.65%**), and F1-score (up to **10.39%**) compared to the Text Only model.
  - Significant improvements seen in **VS Code** and **Roslyn**.
- **Text + Code (FusionNet𝑇𝐶)**:
  - Improved Precision (up to **13.06%**) and F1-score (up to **12.64%**), with **VS Code** showing slightly lower performance.
- **Text + Image + Code (FusionNet𝑇𝐼𝐶)**:
  - Best performance across all metrics:
    - **Precision**: +14.34%
    - **Recall**: +13.90%
    - **F1-score**: +14.12%

---

## Discussion

### Multimodal Fusion Benefits:
- The integration of **Text, Image, and Code** in FusionNet𝑇𝐼𝐶 significantly outperformed other models.
- The fusion of diverse data types enriched the classification process, achieving an **F1-score improvement of 5.07% to 14.12%** over the Text Only model.

### Text-Image Experiment Insights:
- **FusionNet𝑇𝐼 (Text + Image)** provided valuable improvements, especially in **VS Code** and **Roslyn**, showing that images enhance understanding of issue reports.

### Text-Code Experiment Challenges:
- **FusionNet𝑇𝐶 (Text + Code)** showed mixed results in **VS Code**, where the Text Only model performed slightly better.
- Potential challenges include:
  - **Token Length**: No significant data loss was found.
  - **Weight Assignment**: Equal weights to all modalities may have reduced synergy.
  - **Data Quality**: Variability in the quality of reports impacted the results.

### Future Work & Implications:
- **Researchers**: Extend multimodal approaches to multi-class and multi-label tasks.
- **Developers and Companies**: Leverage multimodal models to improve issue classification and streamline issue tracking.
- **Future Directions**:
  1. **State-of-the-Art Feature Extraction**: Employ models like BERT for text, CodeBERT for code, and Transformer-based vision models for images to improve accuracy.
  2. **Advanced Fusion Methods**: Explore bilinear pooling techniques like MCB, MLB, MUTAN, and MFB to enhance feature interaction while maintaining computational efficiency.

---

## Conclusions
- **FusionNet𝑇𝐼𝐶 (Text, Image, Code)** outperformed other models, with an **F1-score improvement of 5.07% to 14.12%**.
- Combining text, image, and code improves issue classification accuracy.
- Multimodal approaches are essential for handling diverse issue report data.
