# Wireless Signal Classification using Deep Learning

This project implements a supervised ML pipeline in *TensorFlow/Keras* to classify wireless modulation schemes (AM, FM, BPSK, QPSK, 16QAM, etc.) using the *RML2016.10a dataset* of raw IQ samples.

## Features
- Residual 1D-CNN for IQ sequence classification.
- Data augmentation: phase jitter, time shift, AWGN.
- Per-channel normalization, optional frame stacking (128→256).
- Training with AdamW, LR scheduling, early stopping.
- Evaluation: confusion matrix, accuracy-vs-SNR plots.


## MIT License
### `requirements`
- tensorflow==2.19.0
- numpy
- scikit-learn
- matplotlib

