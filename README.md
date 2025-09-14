# Wireless Signal Classification using Deep Learning

This project implements a supervised ML pipeline in *TensorFlow/Keras* to classify wireless modulation schemes (AM, FM, BPSK, QPSK, 16QAM, etc.) using the *RML2016.10a dataset* of raw IQ samples.

## Features
- Residual 1D-CNN for IQ sequence classification.
- Data augmentation: phase jitter, time shift, AWGN.
- Per-channel normalization, optional frame stacking (128→256).
- Training with AdamW, LR scheduling, early stopping.
- Evaluation: confusion matrix, accuracy-vs-SNR plots.

## Usage
bash
pip install -r requirements.txt
python -m src.train --pkl_path data/RML2016.10a_dict.pkl --epochs 40 --batch 512

📂 Repo Layout

-src/            → models, data pipeline, training script
-notebooks/      → Colab notebook
-results/        → plots and logs
-data/           → dataset (not tracked)

📜 License

MIT License

---

### `requirements.txt`
txt
tensorflow==2.19.0
numpy
scikit-learn
matplotlib


---

📄 .gitignore

_pycache_/
*.pyc
.ipynb_checkpoints/
data/*
!data/.gitkeep
results/*
!results/.gitkeep
