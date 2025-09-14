# Wireless Signal Classification using Deep Learning

This project implements a supervised deep learning pipeline to classify *wireless modulation schemes* (AM, FM, BPSK, QPSK, 16QAM, etc.) using the *RML2016.10a dataset* of raw IQ samples.  
It demonstrates the application of *machine learning for wireless communications, with relevance to **cognitive radio and spectrum sensing*.

---

## 📌 Features
- Preprocessing: per-channel normalization, optional frame stacking (128 → 256 samples).
- Data augmentation: probabilistic phase jitter, time shifts, and AWGN noise injection.
- Model: Residual 1D-CNN (TensorFlow/Keras) with AdamW optimizer and LR scheduling.
- Training: Early stopping, model checkpointing, Colab-compatible tf.data pipelines.
- Evaluation: Accuracy, confusion matrices, accuracy-vs-SNR plots.

---

## 📂 Repository Structure
