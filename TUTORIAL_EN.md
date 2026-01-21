# Tutorial: LDCT Denoising with Pix2Pix + WGAN-GP

A step-by-step guide to training and using the LDCT denoising model.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Model Training](#3-model-training)
4. [Evaluation](#4-evaluation)
5. [Desktop Application](#5-desktop-application)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/Haymr/LDCT-Denoising-Pix2Pix-WGANGP.git
cd LDCT-Denoising-Pix2Pix-WGANGP
```

### 1.2 Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Verify GPU (For Training)

```python
import tensorflow as tf
print("GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

---

## 2. Data Preparation

### 2.1 Dataset Structure

The model expects paired low-dose and full-dose CT images:

```
data/
├── trainA/          # Low-dose images (.npy)
│   ├── patient1_0001.npy
│   ├── patient1_0002.npy
│   └── ...
└── trainB/          # Full-dose images (.npy)
    ├── patient1_0001.npy
    ├── patient1_0002.npy
    └── ...
```

### 2.2 DICOM Preprocessing

Open `notebooks/02_data_preprocessing.ipynb` and follow these steps:

1. **Set paths** to your DICOM directory
2. **Run all cells** to process:
   - Read DICOM files
   - Convert to Hounsfield Units (HU)
   - Clip to [-1000, 1000] HU
   - Resize to 256×256
   - Normalize to [-1, 1]
   - Save as .npy files

### 2.3 Preprocessing Formula

```python
# HU Conversion
hu = pixel_array * RescaleSlope + RescaleIntercept

# Normalization
normalized = (clipped_hu - (-1000)) / (1000 - (-1000))  # [0, 1]
normalized = normalized * 2 - 1  # [-1, 1]
```

---

## 3. Model Training

### 3.1 Using Google Colab (Recommended)

1. Upload `01_model_architecture.ipynb` and `03_training.ipynb` to Colab
2. Mount Google Drive
3. Update data paths in the notebook
4. Run all cells

### 3.2 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 4 | Adjust based on GPU memory |
| Learning Rate | 2e-4 | For both G and D |
| Lambda GP | 10 | Gradient penalty weight |
| Lambda L1 | 100 | L1 reconstruction weight |
| Epochs | 50 | ~4-6 hours on T4 GPU |

### 3.3 Monitoring Training

The `GANMonitor` callback:
- Saves sample images every epoch
- Saves model weights every 5 epochs

Check `results/` folder for training progress.

---

## 4. Evaluation

### 4.1 Internal Validation

Open `notebooks/04_validation_internal.ipynb`:

```python
# Expected results on Mayo dataset
PSNR: 37.75 dB
SSIM: 0.891
```

### 4.2 External Testing

Open `notebooks/05_external_test_phantomx.ipynb` to test on PhantomX dataset:

1. Prepare PhantomX DICOM files
2. Select reconstruction method (FBP recommended)
3. Run inference and compare metrics

---

## 5. Desktop Application

### 5.1 Running the App

```bash
python app/main.py
```

### 5.2 Using the App

1. **Drag and drop** a DICOM file onto the window
2. Wait for model to process (~2-3 seconds)
3. Toggle between **Side-by-Side** and **Slider** views
4. Use the slider to compare original and enhanced images

### 5.3 App Requirements

- PyQt5
- Model weights file (`G_epoch_50.h5`)
- Place `.h5` file in root directory

---

## 6. Troubleshooting

### CUDA/GPU Issues

```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Errors

- Reduce batch size to 2 or 1
- Use gradient checkpointing
- Process images in smaller batches

### PyQt5 Platform Plugin Error

```bash
# Mac with Anaconda
export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/anaconda3/plugins/platforms
```

### Model Not Loading

Ensure the model architecture matches the saved weights:
- Input shape: (256, 256, 1)
- Same number of filters in each layer

---

## Quick Reference

| Task | Command/File |
|------|--------------|
| Install | `pip install -r requirements.txt` |
| Preprocess | `notebooks/02_data_preprocessing.ipynb` |
| Train | `notebooks/03_training.ipynb` |
| Evaluate | `notebooks/04_validation_internal.ipynb` |
| External Test | `notebooks/05_external_test_phantomx.ipynb` |
| Desktop App | `python app/main.py` |

---

## Need Help?

- Check the [README.md](README.md) for project overview
- Open an issue on GitHub for bugs
- Review notebook markdown cells for detailed explanations
