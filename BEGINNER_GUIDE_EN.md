# 🎯 Beginner's Guide: LDCT Denoising

**A step-by-step guide for non-programmers**

> 💡 This guide is designed for users with no coding experience. For more technical details, see the [Advanced Tutorial](TUTORIAL_EN.md).

---

## 📑 Table of Contents

1. [What Does This Project Do?](#-what-does-this-project-do)
2. [Desktop Application (Easiest Way)](#-desktop-application-easiest-way)
3. [Training with Google Colab](#-training-with-google-colab)
4. [Running on Your Computer](#-running-on-your-computer)
5. [Frequently Asked Questions](#-frequently-asked-questions)
6. [Glossary](#-glossary)

---

## 🔬 What Does This Project Do?

### What is CT (Computed Tomography)?

CT scanning is a medical imaging technique that creates cross-sectional images of the body. It uses X-rays to show detailed internal structures.

### What is Low-Dose CT (LDCT)?

- **Normal CT**: High radiation dose → Clear image ✅ but radiation risk ⚠️
- **Low-Dose CT**: Low radiation dose → Safe ✅ but noisy (blurry) image ⚠️

### What Does This Project Do?

This project uses **artificial intelligence** to remove noise from Low-Dose CT images. This means:

- ✅ Lower radiation exposure during scanning (patient safety)
- ✅ Image quality is enhanced by AI (diagnostic accuracy)

### Example Result

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   BEFORE (Low-Dose)        AFTER (with AI)          │
│   ┌───────────────┐        ┌───────────────┐        │
│   │ ░░▒▒░░▒▒░░▒▒ │   →    │ ██████████████ │        │
│   │ ▒▒░░▒▒░░▒▒░░ │        │ ██          ██ │        │
│   │ ░░▒▒░░▒▒░░▒▒ │        │ ██  ████  ████ │        │
│   │ ▒▒░░▒▒░░▒▒░░ │        │ ██████████████ │        │
│   └───────────────┘        └───────────────┘        │
│        Noisy                   Clean                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🖥️ Desktop Application (Easiest Way)

With this method, you just need to **download and run the project**. The model is already trained.

### Requirements

- ✅ Python 3.9 or higher installed
- ✅ Windows, Mac, or Linux

### Step 1: Install Python

> ⚠️ Skip this step if Python is already installed.

**For Windows:**
1. Go to [python.org](https://www.python.org/downloads/)
2. Click "Download Python 3.x" button
3. Run the downloaded file
4. **IMPORTANT**: Check "Add Python to PATH" ✅
5. Click "Install Now"

**For Mac:**
1. Open Terminal (search for "Terminal" in Spotlight)
2. Type: `python3 --version`
3. If Python is not installed, an installation prompt will appear

### Step 2: Download the Project

1. Click the green **"Code"** button at the top of this page
2. Select **"Download ZIP"**
3. Extract the downloaded ZIP file (right-click → "Extract Here")

### Step 3: Install Dependencies

1. Open the downloaded folder
2. Copy the folder path (e.g., `C:\Users\John\Desktop\LDCT-Denoising-Pix2Pix-WGANGP`)

**Windows:**
1. Type "cmd" in the Start menu and press Enter
2. Type this command (replace the path with your folder):
   ```
   cd C:\Users\John\Desktop\LDCT-Denoising-Pix2Pix-WGANGP
   ```
3. Then type:
   ```
   pip install -r requirements.txt
   ```
4. Wait until installation completes (may take a few minutes)

**Mac/Linux:**
1. Open Terminal
2. Type this command (replace the path with your folder):
   ```
   cd /Users/John/Desktop/LDCT-Denoising-Pix2Pix-WGANGP
   ```
3. Then type:
   ```
   pip3 install -r requirements.txt
   ```

### Step 4: Run the Application

In the same terminal/command prompt:

**Windows:**
```
python app/main.py
```

**Mac/Linux:**
```
python3 app/main.py
```

### Step 5: Process Your DICOM File

1. The application window will open
2. **Drag and drop** your DICOM file onto the window
3. Wait 2-3 seconds
4. See the result! 🎉

**View Options:**
- **Side-by-Side**: View original and enhanced images side by side
- **Slider**: Compare using a sliding control

---

## ☁️ Training with Google Colab

Google Colab is a free online Python environment provided by Google. **Use this method if you want to train your own model.**

### Benefits of Colab

- ✅ No installation required on your computer
- ✅ Free GPU access (essential for training!)
- ✅ Works in your browser
- ✅ Integrated with Google Drive

### Step 1: Google Account

If you don't have one, create a free account at [accounts.google.com](https://accounts.google.com).

### Step 2: Go to Colab

1. Open [colab.research.google.com](https://colab.research.google.com) in your browser
2. Sign in with your Google account

### Step 3: Open the Notebook

**Method A - Directly from GitHub:**
1. In Colab, click "File" → "Open notebook"
2. Go to the "GitHub" tab
3. Paste this URL:
   ```
   https://github.com/Haymr/LDCT-Denoising-Pix2Pix-WGANGP
   ```
4. Select the notebook you want (e.g., `03_training.ipynb`)

**Method B - Upload File:**
1. Download the project as ZIP from GitHub
2. Extract the ZIP
3. In Colab, click "File" → "Upload notebook"
4. Select the `.ipynb` file from the `notebooks` folder

### Step 4: Enable GPU ⚡

This step is **very important**! Training without GPU will take days.

1. Click "Runtime" in the top menu
2. Select "Change runtime type"
3. Under "Hardware accelerator", select **"GPU"**
4. Click "Save"

> 💡 **Tip**: T4 GPU is free and sufficient. Training takes approximately 4-6 hours.

### Step 5: Connect Google Drive

Drive is needed to store your data and trained model.

Add this code at the beginning of your notebook and run it:

```python
from google.colab import drive
drive.mount('/content/drive')
```

In the popup:
1. Select your Google account
2. Click "Allow"

### Step 6: Upload Data

**Option A - Mayo Dataset (Recommended):**
If you have the Mayo LDCT dataset:
1. Upload it to your Google Drive
2. Update the data path in the notebook

**Option B - Your Own Data:**
1. Upload your DICOM files to Drive
2. Run the `02_data_preprocessing.ipynb` notebook
3. This notebook converts DICOMs to the format the model understands (.npy)

### Step 7: Start Training

1. Open the `03_training.ipynb` notebook
2. Click "Runtime" → "Run all" (or Ctrl+F9) in the top menu
3. Each cell will run sequentially
4. You'll see the training progress on screen

### Step 8: Download the Model

When training is complete:
1. Click "Files" (folder icon) in the left panel
2. Open the `results` folder
3. Find the `G_epoch_50.h5` file
4. Right-click and select "Download"

This file can now be used in the desktop application!

---

## 💻 Running on Your Computer

If you want to train on your own computer instead of Colab, you need a powerful GPU.

### Requirements

- ✅ NVIDIA GPU (at least 8GB VRAM recommended)
- ✅ CUDA and cuDNN installed
- ✅ Python 3.9+

> ⚠️ **Warning**: Training without a GPU will be **very slow** (could take days).

### Step by Step

1. **Download the project** (see the desktop application steps above)

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**:
   - Put your DICOM files in the `data/raw/` folder
   - Open Jupyter Notebook: `jupyter notebook`
   - Run `notebooks/02_data_preprocessing.ipynb`

5. **Start training**:
   - Open and run `notebooks/03_training.ipynb`

---

## ❓ Frequently Asked Questions

### "I don't have a GPU, can I train the model?"

**Short answer**: Use Google Colab! It provides free GPU.

**Long answer**: Training on CPU is technically possible but not practical. Training that takes 4-6 hours on Colab could take days on CPU.

---

### "Where can I find DICOM files?"

- **Hospital systems**: Export from PACS systems
- **Research datasets**: 
  - [Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
  - Mayo Clinic LDCT dataset

---

### "I'm getting an error, what should I do?"

**Most common errors:**

| Error | Solution |
|-------|----------|
| `No module named 'tensorflow'` | Run `pip install tensorflow` |
| `CUDA out of memory` | Reduce batch size (to 2 or 1) |
| `FileNotFoundError` | Check file paths |
| PyQt5 error (Mac) | In terminal: `export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/anaconda3/plugins/platforms` |

---

### "Where is the model file (.h5)?"

The `G_epoch_50.h5` file should be in the project root folder. If not:
1. Download from the Releases page
2. or train on Colab and download

---

### "Can I use my own CT images?"

Yes! Your images must be in DICOM format. The application automatically:
1. Reads the DICOM
2. Converts to Hounsfield Units
3. Sends to the model
4. Displays the result

---

## 📖 Glossary

| Term | Description |
|------|-------------|
| **CT** | Computed Tomography. Uses X-rays to create cross-sectional images of the body. |
| **LDCT** | Low-Dose CT. CT taken with reduced radiation. Safer but noisier. |
| **DICOM** | Digital Imaging and Communications in Medicine. Standard format for medical images. |
| **HU** | Hounsfield Unit. Measures tissue density in CT images. Water=0, Air=-1000, Bone=+1000 |
| **GPU** | Graphics Processing Unit. Performs very fast computations for AI training. |
| **GAN** | Generative Adversarial Network. An AI model where two neural networks compete. |
| **Pix2Pix** | A type of GAN that translates from one image to another. |
| **WGAN-GP** | Wasserstein GAN with Gradient Penalty. A GAN variant that provides more stable training. |
| **PSNR** | Peak Signal-to-Noise Ratio. Image quality metric. Higher = Better. |
| **SSIM** | Structural Similarity Index. Structural similarity metric. Closer to 1 = Better. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Batch Size** | Number of images processed simultaneously. |
| **NPY** | NumPy array format. Used to store numerical data in Python. |
| **Colab** | Google Colaboratory. Free online Python and GPU environment. |

---

## 🆘 Still Need Help?

1. 📖 Check the [Advanced Tutorial](TUTORIAL_EN.md)
2. 📝 Review the [README](README.md) file
3. 🐛 [Open an Issue](https://github.com/Haymr/LDCT-Denoising-Pix2Pix-WGANGP/issues) on GitHub

---

*This guide is part of the LDCT Denoising project. Distributed under the MIT License.*
