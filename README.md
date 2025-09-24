# Deep Learning-Based Classification of Wireless Modulation Schemes

This project implements a deep learning approach to classify **wireless modulation schemes** from raw **I/Q (in-phase and quadrature) radio signals**.  
The model is trained and evaluated on the **RadioML 2016.10a dataset**, a widely used benchmark for RF signal classification research.

---

## 📌 Project Overview
Modern wireless communication systems use different modulation schemes (BPSK, QPSK, QAM, etc.) to transmit information.  
Automatically identifying these modulation types is critical for:
- **Cognitive radios** (dynamic spectrum access)  
- **Signal intelligence and surveillance**  
- **IoT and 5G/6G networks**  

This project uses a **ResNet-style 1D CNN** to learn features directly from raw I/Q samples without manual feature engineering.

---

## ⚙️ Key Features
- **Dataset**: RadioML 2016.10a (24 modulation classes, various SNR levels from -20 dB to +18 dB).  
- **Preprocessing**:
  - Normalization of I/Q channels.  
  - Optional frame stacking for longer temporal context.  
- **Data Augmentation**:
  - Random phase shift  
  - Time shift  
  - Gaussian noise injection  
- **Model**:
  - Residual 1D CNN blocks with skip connections.  
  - Global average pooling and dense softmax classifier.  
- **Training Strategy**:
  - Warm-up without augmentation (3 epochs).  
  - Full training with augmentations (up to 100 epochs).  
  - AdamW optimizer with weight decay.  
  - Early stopping and learning rate scheduling.  
- **Evaluation**:
  - Train/Validation/Test splits with stratification.  
  - Accuracy vs SNR curve.  
  - Confusion matrix for per-class performance.  

---

## 📊 Example Results
### Accuracy vs SNR
The model achieves higher accuracy at higher SNRs, as expected in wireless systems.
![Accuracy_vs_SNR](https://github.com/user-attachments/assets/b27ee1a7-d74c-45af-9747-ea20291cc4f8)



- At **high SNR (>10 dB)** → >90% accuracy.  
- At **low SNR (<0 dB)** → accuracy drops due to noise corruption.  

---

## 🛠️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/modulation-classification.git
cd modulation-classification
pip install -r requirements.txt
