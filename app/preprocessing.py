"""
LDCT Denoising - Preprocessing Module
DICOM görüntülerini model için hazırlar.
"""

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None



# Preprocessing sabitleri (notebook'lardan alındı)
HU_MIN = -1000
HU_MAX = 1000
IMG_SIZE = (256, 256)


def preprocess_dicom(file_path: str) -> tuple:
    """
    DICOM dosyasını okur ve model için hazırlar.
    
    Args:
        file_path: DICOM dosya yolu
        
    Returns:
        tuple: (preprocessed_image, original_image_for_display)
    """
    if pydicom is None:
        raise ImportError("pydicom kütüphanesi yüklü değil. 'pip install pydicom' komutunu çalıştırın.")
    
    # 1. DICOM Okuma
    dcm = pydicom.dcmread(file_path)
    pixel_array = dcm.pixel_array.astype(np.float32)
    
    # 2. HU Dönüşümü (Rescale Slope & Intercept)
    intercept = dcm.RescaleIntercept if 'RescaleIntercept' in dcm else 0
    slope = dcm.RescaleSlope if 'RescaleSlope' in dcm else 1
    hu_image = pixel_array * slope + intercept
    
    # Orijinal görüntüyü sakla (display için)
    original_hu = hu_image.copy()
    
    # 3. Clipping [-1000, 1000]
    hu_image = np.clip(hu_image, HU_MIN, HU_MAX)
    
    # 4. Resize to 256x256 using PIL
    pil_img = Image.fromarray(hu_image)
    pil_img = pil_img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    hu_image = np.array(pil_img, dtype=np.float32)
    
    # 5. Normalizasyon [-1, 1]
    normalized = (hu_image - HU_MIN) / (HU_MAX - HU_MIN)  # [0, 1]
    normalized = (normalized * 2) - 1  # [-1, 1]
    
    # Model input shape: (1, 256, 256, 1)
    model_input = normalized.astype(np.float32)
    model_input = np.expand_dims(model_input, axis=(0, -1))
    
    return model_input, original_hu


def postprocess_output(model_output: np.ndarray) -> np.ndarray:
    """
    Model çıktısını görüntülenebilir formata dönüştürür.
    
    Args:
        model_output: Model'den gelen [-1, 1] aralığındaki çıktı
        
    Returns:
        numpy array: [0, 255] aralığında uint8 görüntü
    """
    # Model output shape: (1, 256, 256, 1) -> (256, 256)
    img = model_output.squeeze()
    
    # [-1, 1] -> [0, 1]
    img = (img + 1) / 2.0
    
    # [0, 1] -> [0, 255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    HU görüntüsünü display için normalize eder.
    
    Args:
        image: HU değerlerini içeren görüntü
        
    Returns:
        numpy array: [0, 255] aralığında uint8 görüntü
    """
    # Clip to window
    img = np.clip(image, HU_MIN, HU_MAX)
    
    # Normalize to [0, 255]
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    img = (img * 255).astype(np.uint8)
    
    return img


def model_input_to_display(model_input: np.ndarray) -> np.ndarray:
    """
    Model inputunu (256x256, normalized) display için dönüştürür.
    
    Args:
        model_input: Model input formatında görüntü
        
    Returns:
        numpy array: [0, 255] aralığında uint8 görüntü
    """
    # (1, 256, 256, 1) -> (256, 256)
    img = model_input.squeeze()
    
    # [-1, 1] -> [0, 1]
    img = (img + 1) / 2.0
    
    # [0, 1] -> [0, 255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img
