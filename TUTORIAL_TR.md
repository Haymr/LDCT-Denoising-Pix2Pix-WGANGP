# Eğitim Rehberi: Pix2Pix + WGAN-GP ile LDCT Gürültü Azaltma

> 💡 **Programlama bilmiyorsanız**: [Başlangıç Rehberi](BEGINNER_GUIDE_TR.md)'ne göz atın.

LDCT gürültü azaltma modelini eğitmek ve kullanmak için adım adım rehber.

## İçindekiler

1. [Ortam Kurulumu](#1-ortam-kurulumu)
2. [Veri Hazırlama](#2-veri-hazırlama)
3. [Model Eğitimi](#3-model-eğitimi)
4. [Değerlendirme](#4-değerlendirme)
5. [Masaüstü Uygulaması](#5-masaüstü-uygulaması)
6. [Sorun Giderme](#6-sorun-giderme)

---

## 1. Ortam Kurulumu

### 1.1 Repoyu Klonla

```bash
git clone https://github.com/Haymr/LDCT-Denoising-Pix2Pix-WGANGP.git
cd LDCT-Denoising-Pix2Pix-WGANGP
```

### 1.2 Sanal Ortam Oluştur (Önerilir)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 1.3 Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 1.4 GPU Kontrolü (Eğitim İçin)

```python
import tensorflow as tf
print("Kullanılabilir GPU:", len(tf.config.list_physical_devices('GPU')))
```

---

## 2. Veri Hazırlama

### 2.1 Veri Yapısı

Model, eşleştirilmiş düşük doz ve yüksek doz CT görüntüleri bekler:

```
data/
├── trainA/          # Düşük doz görüntüler (.npy)
│   ├── hasta1_0001.npy
│   ├── hasta1_0002.npy
│   └── ...
└── trainB/          # Yüksek doz görüntüler (.npy)
    ├── hasta1_0001.npy
    ├── hasta1_0002.npy
    └── ...
```

### 2.2 DICOM Ön İşleme

`notebooks/02_data_preprocessing.ipynb` dosyasını açın ve şu adımları izleyin:

1. **Yolları ayarlayın** - DICOM dizininizi belirtin
2. **Tüm hücreleri çalıştırın** - İşlenecek adımlar:
   - DICOM dosyalarını oku
   - Hounsfield Birimi'ne (HU) dönüştür
   - [-1000, 1000] HU aralığına kırp
   - 256×256 boyutuna yeniden boyutlandır
   - [-1, 1] aralığına normalize et
   - .npy dosyası olarak kaydet

### 2.3 Ön İşleme Formülü

```python
# HU Dönüşümü
hu = piksel_değeri * RescaleSlope + RescaleIntercept

# Normalizasyon
normalized = (kırpılmış_hu - (-1000)) / (1000 - (-1000))  # [0, 1]
normalized = normalized * 2 - 1  # [-1, 1]
```

---

## 3. Model Eğitimi

### 3.1 Google Colab Kullanımı (Önerilir)

1. `01_model_architecture.ipynb` ve `03_training.ipynb` dosyalarını Colab'a yükleyin
2. Google Drive'ı bağlayın
3. Notebook'taki veri yollarını güncelleyin
4. Tüm hücreleri çalıştırın

### 3.2 Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Batch Size | 4 | GPU belleğine göre ayarlayın |
| Learning Rate | 2e-4 | G ve D için |
| Lambda GP | 10 | Gradient penalty ağırlığı |
| Lambda L1 | 100 | L1 reconstruction ağırlığı |
| Epoch | 50 | T4 GPU'da ~4-6 saat |

### 3.3 Eğitimi İzleme

`GANMonitor` callback'i:
- Her epoch sonunda örnek görseller kaydeder
- Her 5 epoch'ta model ağırlıklarını kaydeder

İlerlemeyi görmek için `results/` klasörünü kontrol edin.

---

## 4. Değerlendirme

### 4.1 Dahili Doğrulama

`notebooks/04_validation_internal.ipynb` dosyasını açın:

```python
# Mayo veri setinde beklenen sonuçlar
PSNR: 37.75 dB
SSIM: 0.891
```

### 4.2 Harici Test

PhantomX veri setinde test için `notebooks/05_external_test_phantomx.ipynb` dosyasını açın:

1. PhantomX DICOM dosyalarını hazırlayın
2. Rekonstrüksiyon yöntemini seçin (FBP önerilir)
3. Inference yapın ve metrikleri karşılaştırın

---

## 5. Masaüstü Uygulaması

### 5.1 Uygulamayı Çalıştırma

```bash
python app/main.py
```

### 5.2 Uygulama Kullanımı

1. **Sürükle-bırak**: DICOM dosyasını pencereye sürükleyin
2. Model işlerken bekleyin (~2-3 saniye)
3. **Yan Yana** veya **Slider** görünümü seçin
4. Kaydırıcıyı kullanarak orijinal ve geliştirilmiş görüntüleri karşılaştırın

### 5.3 Uygulama Gereksinimleri

- PyQt5
- Model ağırlık dosyası (`G_epoch_50.h5`)
- `.h5` dosyasını ana dizine yerleştirin

---

## 6. Sorun Giderme

### CUDA/GPU Sorunları

```bash
# TensorFlow GPU desteğini kontrol et
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Bellek Hataları

- Batch size'ı 2 veya 1'e düşürün
- Gradient checkpointing kullanın
- Görüntüleri daha küçük batch'ler halinde işleyin

### PyQt5 Platform Plugin Hatası

```bash
# Anaconda ile Mac
export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/anaconda3/plugins/platforms
```

### Model Yüklenmiyor

Model mimarisinin kaydedilen ağırlıklarla eşleştiğinden emin olun:
- Giriş boyutu: (256, 256, 1)
- Her katmanda aynı filtre sayısı

---

## Hızlı Referans

| Görev | Komut/Dosya |
|-------|-------------|
| Kurulum | `pip install -r requirements.txt` |
| Ön İşleme | `notebooks/02_data_preprocessing.ipynb` |
| Eğitim | `notebooks/03_training.ipynb` |
| Değerlendirme | `notebooks/04_validation_internal.ipynb` |
| Harici Test | `notebooks/05_external_test_phantomx.ipynb` |
| Masaüstü Uygulama | `python app/main.py` |

---

## Yardım Lazım mı?

- Proje özeti için [README.md](README.md) dosyasına bakın
- Hatalar için GitHub'da issue açın
- Detaylı açıklamalar için notebook markdown hücrelerini inceleyin
