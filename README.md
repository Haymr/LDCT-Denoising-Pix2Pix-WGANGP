# LDCT Denoising with Pix2Pix + WGAN-GP

A hybrid Pix2Pix + WGAN-GP model for Low-Dose CT image denoising.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Features

- **Hybrid Model**: Pix2Pix U-Net Generator + WGAN-GP loss
- **High Performance**: PSNR ~41.7 dB, SSIM ~0.94
- **Desktop Application**: Drag-and-drop DICOM processing
- **External Validation**: Tested on PhantomX dataset

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/LDCT-Denoising-Pix2Pix-WGANGP.git
cd LDCT-Denoising-Pix2Pix-WGANGP
pip install -r requirements.txt
```

### Desktop Application

```bash
python app/main.py
```

Simply drag and drop a DICOM file, and the model will automatically denoise it.

## 📁 Project Structure

```
├── app/                          # Desktop application
│   ├── main.py                   # PyQt5 GUI
│   ├── preprocessing.py          # DICOM processing
│   ├── model.py                  # U-Net Generator
│   └── comparison_widget.py      # Comparison views
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_model_architecture.ipynb    # Model definitions
│   ├── 02_data_preprocessing.ipynb    # Data preprocessing
│   ├── 03_training.ipynb              # Training loop
│   ├── 04_validation_internal.ipynb   # PSNR/SSIM evaluation
│   └── 05_external_test_phantomx.ipynb# External test
│
├── G_epoch_50.h5                 # Trained model weights
└── requirements.txt              # Dependencies
```

## 🔬 Model Architecture

```
Input (256x256x1)
    │
    ▼
┌─────────────────────────────┐
│     U-Net Generator         │
│  • 8-layer Encoder          │
│  • 7-layer Decoder          │
│  • Skip Connections         │
│  • tanh activation          │
└─────────────────────────────┘
    │
    ▼
Output (256x256x1)
```

**Loss Functions:**
- Wasserstein Loss + Gradient Penalty (discriminator)
- Wasserstein Loss + L1 Reconstruction (generator)

## 📊 Results

| Dataset | PSNR (dB) | SSIM |
|---------|-----------|------|
| Mayo (Validation) | 41.73 ± 5.84 | 0.941 ± 0.045 |
| PhantomX (External) | - | - |

## 🎯 Use Cases

- **Radiology**: Enhance low-dose CT scan quality
- **Research**: Develop LDCT denoising methods
- **Education**: Learn GAN architectures

## 📖 Notebooks

Run the notebooks in order to understand the project:

1. **01_model_architecture.ipynb** - Generator and Discriminator architectures
2. **02_data_preprocessing.ipynb** - DICOM to NPY conversion
3. **03_training.ipynb** - Model training
4. **04_validation_internal.ipynb** - PSNR/SSIM calculation
5. **05_external_test_phantomx.ipynb** - Independent dataset testing

## 🛠️ Requirements

- Python 3.9+
- TensorFlow 2.10+
- PyQt5 (for desktop app)
- CUDA-enabled GPU (recommended for training)

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Mayo Clinic LDCT Dataset
- PhantomX Abdomen/Pelvis Dataset
