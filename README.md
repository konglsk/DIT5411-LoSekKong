# DIT5411 - Traditional Chinese Handwriting Recognition (13,065 Characters)

## 1. Project Overview & Computing Environment
This project develops, optimizes, and compares multiple Convolutional Neural Network (CNN) architectures using **TensorFlow** and **OpenCV** to recognize **13,065 Traditional Chinese characters** from handwritten images. To ensure a realistic and leak-free evaluation, the dataset is strictly partitioned prior to data augmentation, and four model configurations with varying depths, architectures, and regularization strategies are evaluated on the unseen testing set.

* **Hardware & Environment**: Python 3.11, TensorFlow (Mixed Precision FP16), OpenCV, NVIDIA GeForce RTX 4080 Super (16GB VRAM), 32GB System RAM.
* **Notebook File**: [`Traditional_Chinese_Handwriting_Recognition.ipynb`](Traditional_Chinese_Handwriting_Recognition.ipynb)

---

## 2. Dataset & Train/Test Split
* **Source**: [AI-FREE-Team Traditional Chinese Handwriting Dataset (Full 13,065 Characters)](https://github.com)
* **Total Classes**: 13,065 Traditional Chinese characters (`684,677` grayscale handwriting images in total).
* **Strict Splitting Strategy (Zero Data Leakage)**:
  * **Training Set**: First **40 images** per character (`522,127` raw images).
  * **Testing Set**: Remaining ~10 images per character (`162,550` unseen images), kept completely untouched during data augmentation and training.

---

## 3. Image Preprocessing & Data Augmentation (OpenCV)
1. **Preprocessing**:
   * All images are loaded in grayscale using OpenCV (`cv2.imread`) and resized uniformly to **`48 × 48` pixels**.
   * Background intensity is automatically detected and inverted so handwriting strokes are represented as foreground (`1.0`) on a clean zero background (`0.0`).
2. **Data Augmentation (40 → 200 Images per Character)**:
   * To improve generalization across handwriting variations, each character's 40 training images are augmented **5×** to **200 images per character** (`2,610,635` training samples per epoch) using random affine transformations:
     * **Random Rotation**: ±7° to ±10°
     * **Random Translation**: ±2 to ±3 pixels (horizontal & vertical)
     * **Random Scaling / Zoom**: 0.90× – 1.10×

![OpenCV Data Augmentation Demo](augmentation_demo.png)

---

## 4. Model Architectures & Iterative Optimization
Four models with progressive architectural depth and regularization techniques were designed and compared:

1. **Model 1 (`Model_1_Basic_CNN`)**: A shallow 2-layer baseline CNN (`Conv2D(32)` → `MaxPool` → `Conv2D(64)` → `MaxPool` → `Flatten` → `Dense(512)` → `Dropout(0.3)`).
2. **Model 2 (`Model_2_Deep_VGG_CNN`)**: A deeper 4-layer VGG-style CNN (`64 → 128 → 256 → 512` filters) incorporating `BatchNormalization` after every convolution and `GlobalAveragePooling2D` to eliminate massive dense parameters.
3. **Model 3a (`Model_3_ResNet` - Initial)**: A 10-layer Residual Network featuring 3 residual blocks with identity/projection skip connections, trained with a fixed learning rate (`1e-3`) and no weight regularization.
4. **Model 3b (`Model_3_Optimized_ResNet` - Final)**: An enhanced 10-layer ResNet addressing the overfitting observed in Model 3a by introducing **L2 weight regularization (`1e-4`)**, a wider **`Dense(1024)`** feature bottleneck layer before the 13,065-class softmax output, `Dropout(0.5)`, and **`ReduceLROnPlateau`** dynamic learning rate scheduling.

---

## 5. Experimental Results & Comparison (Testing Set: 162,550 Images)

| Model Name | Depth & Architecture Highlights | Input Size | Test Loss | Testing Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Model 1: Basic CNN** | 2 Conv layers + Flatten + Dense(512) | 48 × 48 | — | **36.76%** |
| **Model 2: Deep VGG-like CNN** | 4 Conv layers + BatchNorm + GlobalAvgPool | 48 × 48 | — | **54.87%** |
| **Model 3a: ResNet (Initial)** | 10-layer Residual Net (Fixed LR, No L2) | 48 × 48 | 10.9600 | **41.88%** |
| **Model 3b: Optimized ResNet** | **10-layer ResNet + L2 Reg + Dense(1024) + LR Decay** | **48 × 48** | **1.1112** | **78.93%** |

### Key Findings & Analysis
* **Impact of Depth & Batch Normalization**: Increasing network depth from 2 layers (`36.76%`) to 4 layers with `BatchNormalization` (`54.87%`) boosted testing accuracy by **+18.11%**, demonstrating the necessity of hierarchical feature extraction for complex Traditional Chinese character strokes.
* **Overfitting vs. Regularization in Deep ResNets**: Without weight decay and learning rate scheduling, the initial 10-layer ResNet overfitted the augmented training set (`92.53%` train accuracy vs. `41.88%` test accuracy). Incorporating **L2 regularization**, expanding the classifier bottleneck to **`Dense(1024)`** to support 13,065 output classes, and applying **`ReduceLROnPlateau`** stabilized convergence and increased unseen testing accuracy to **78.93%**.

---

## 6. Sample Predictions on Unseen Testing Set
Below is a random sample of 10 unseen handwriting images from the Testing Set predicted by **`Model_3_Optimized_ResNet`** (achieving 9/10 accuracy, successfully recognizing complex high-stroke Traditional Chinese characters):

![Testing Set Predictions](prediction_demo.png)

---

## 7. References
* **Dataset Repository**: [AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset](https://github.com)
* **OpenCV Documentation**: [OpenCV-Python Tutorials](https://opencv.org)
