# 🎯 Başlangıç Rehberi: LDCT Gürültü Azaltma

**Programlama bilmeyenler için adım adım rehber**

> 💡 Bu rehber, hiç kod yazmamış kişilerin bile projeyi kullanabilmesi için hazırlanmıştır. Daha teknik detaylar için [Gelişmiş Tutorial](TUTORIAL_TR.md)'a bakabilirsiniz.

---

## 📑 İçindekiler

1. [Bu Proje Ne Yapıyor?](#-bu-proje-ne-yapıyor)
2. [Masaüstü Uygulaması (En Kolay Yol)](#-masaüstü-uygulaması-en-kolay-yol)
3. [Google Colab ile Eğitim](#-google-colab-ile-eğitim)
4. [Kendi Bilgisayarınızda Çalıştırma](#-kendi-bilgisayarınızda-çalıştırma)
5. [Sık Sorulan Sorular](#-sık-sorulan-sorular)
6. [Terimler Sözlüğü](#-terimler-sözlüğü)

---

## 🔬 Bu Proje Ne Yapıyor?

### CT (Bilgisayarlı Tomografi) Nedir?

CT taraması, vücudun kesitsel görüntülerini oluşturan bir tıbbi görüntüleme yöntemidir. X-ışınları kullanarak vücudun içini detaylı şekilde gösterir.

### Low-Dose CT (LDCT) Nedir?

- **Normal CT**: Yüksek radyasyon dozu → Net görüntü ✅ ama radyasyon riski ⚠️
- **Low-Dose CT**: Düşük radyasyon dozu → Güvenli ✅ ama gürültülü (bulanık) görüntü ⚠️

### Bu Proje Ne Yapıyor?

Bu proje, **yapay zeka** kullanarak Low-Dose CT görüntülerindeki gürültüyü temizler. Böylece:

- ✅ Düşük radyasyon dozuyla tarama yapılır (hasta güvenliği)
- ✅ Görüntü kalitesi yapay zeka ile iyileştirilir (tanı doğruluğu)

### Örnek Sonuç

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ÖNCE (Low-Dose)          SONRA (AI ile)           │
│   ┌───────────────┐        ┌───────────────┐        │
│   │ ░░▒▒░░▒▒░░▒▒ │   →    │ ██████████████ │        │
│   │ ▒▒░░▒▒░░▒▒░░ │        │ ██          ██ │        │
│   │ ░░▒▒░░▒▒░░▒▒ │        │ ██  ████  ████ │        │
│   │ ▒▒░░▒▒░░▒▒░░ │        │ ██████████████ │        │
│   └───────────────┘        └───────────────┘        │
│      Gürültülü              Temiz                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🖥️ Masaüstü Uygulaması (En Kolay Yol)

Bu yöntemle **sadece projeyi indirip çalıştırmanız yeterli**. Model zaten eğitilmiş durumda.

### Gereksinimler

- ✅ Python 3.9 veya üstü kurulu olmalı
- ✅ Windows, Mac veya Linux

### Adım 1: Python Kurulumu

> ⚠️ Zaten Python kuruluysa bu adımı atlayın.

**Windows için:**
1. [python.org](https://www.python.org/downloads/) adresine gidin
2. "Download Python 3.x" butonuna tıklayın
3. İndirilen dosyayı çalıştırın
4. **ÖNEMLİ**: "Add Python to PATH" kutusunu işaretleyin ✅
5. "Install Now" tıklayın

**Mac için:**
1. Terminal'i açın (Spotlight'ta "Terminal" yazın)
2. Şunu yazın: `python3 --version`
3. Eğer Python yoksa, yükleme penceresi açılacak

### Adım 2: Projeyi İndirin

1. Bu sayfanın üstündeki yeşil **"Code"** butonuna tıklayın
2. **"Download ZIP"** seçin
3. İndirilen ZIP dosyasını açın (sağ tık → "Buraya Çıkart")

### Adım 3: Bağımlılıkları Kurun

1. İndirilen klasörü açın
2. Klasör yolunu kopyalayın (örn: `C:\Users\Ahmet\Desktop\LDCT-Denoising-Pix2Pix-WGANGP`)

**Windows:**
1. Başlat menüsünde "cmd" yazın ve Enter'a basın
2. Şu komutu yazın (yolu kendi klasörünüzle değiştirin):
   ```
   cd C:\Users\Ahmet\Desktop\LDCT-Denoising-Pix2Pix-WGANGP
   ```
3. Sonra şu komutu yazın:
   ```
   pip install -r requirements.txt
   ```
4. Kurulum tamamlanana kadar bekleyin (birkaç dakika sürebilir)

**Mac/Linux:**
1. Terminal'i açın
2. Şu komutu yazın (yolu kendi klasörünüzle değiştirin):
   ```
   cd /Users/Ahmet/Desktop/LDCT-Denoising-Pix2Pix-WGANGP
   ```
3. Sonra şu komutu yazın:
   ```
   pip3 install -r requirements.txt
   ```

### Adım 4: Uygulamayı Çalıştırın

Aynı terminal/komut isteminde:

**Windows:**
```
python app/main.py
```

**Mac/Linux:**
```
python3 app/main.py
```

### Adım 5: DICOM Dosyanızı İşleyin

1. Uygulama penceresi açılacak
2. DICOM dosyanızı pencereye **sürükleyip bırakın**
3. 2-3 saniye bekleyin
4. Sonucu görün! 🎉

**Görünüm Seçenekleri:**
- **Yan Yana**: Orijinal ve iyileştirilmiş görüntüyü yan yana görün
- **Slider**: Kaydırıcı ile karşılaştırma yapın

---

## ☁️ Google Colab ile Eğitim

Google Colab, Google'ın ücretsiz sunduğu online Python ortamıdır. **Kendi modeli eğitmek istiyorsanız** bu yöntemi kullanın.

### Colab'ın Avantajları

- ✅ Bilgisayarınıza hiçbir şey kurmanız gerekmez
- ✅ Ücretsiz GPU kullanabilirsiniz (eğitim için şart!)
- ✅ Tarayıcıdan çalışır
- ✅ Google Drive ile entegre

### Adım 1: Google Hesabı

Eğer yoksa [accounts.google.com](https://accounts.google.com) adresinden ücretsiz hesap oluşturun.

### Adım 2: Colab'a Gidin

1. Tarayıcınızda [colab.research.google.com](https://colab.research.google.com) adresini açın
2. Google hesabınızla giriş yapın

### Adım 3: Notebook'u Açın

**Yöntem A - GitHub'dan Doğrudan:**
1. Colab'da "File" → "Open notebook" tıklayın
2. "GitHub" sekmesine geçin
3. URL kısmına şunu yapıştırın:
   ```
   https://github.com/Haymr/LDCT-Denoising-Pix2Pix-WGANGP
   ```
4. İstediğiniz notebook'u seçin (örn: `03_training.ipynb`)

**Yöntem B - Dosya Yükleyerek:**
1. GitHub'dan projeyi ZIP olarak indirin
2. ZIP'i açın
3. Colab'da "File" → "Upload notebook" tıklayın
4. `notebooks` klasöründen istediğiniz `.ipynb` dosyasını seçin

### Adım 4: GPU'yu Etkinleştirin ⚡

Bu adım **çok önemli**! GPU olmadan eğitim günlerce sürer.

1. Üst menüden "Runtime" (veya "Çalışma Zamanı") tıklayın
2. "Change runtime type" (veya "Çalışma zamanı türünü değiştir") seçin
3. "Hardware accelerator" kısmında **"GPU"** seçin
4. "Save" (veya "Kaydet") tıklayın

> 💡 **İpucu**: T4 GPU ücretsiz ve yeterlidir. Eğitim yaklaşık 4-6 saat sürer.

### Adım 5: Google Drive'ı Bağlayın

Verilerinizi ve eğitilmiş modeli saklamak için Drive gerekli.

Notebook'un başına şu kodu ekleyin ve çalıştırın:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Çıkan pencerede:
1. Google hesabınızı seçin
2. "İzin Ver" tıklayın

### Adım 6: Veri Yükleme

**Seçenek A - Mayo Dataset (Önerilen):**
Eğer Mayo LDCT dataset'iniz varsa:
1. Google Drive'ınıza yükleyin
2. Notebook'taki veri yolunu güncelleyin

**Seçenek B - Kendi Verileriniz:**
1. DICOM dosyalarınızı Drive'a yükleyin
2. `02_data_preprocessing.ipynb` notebook'unu çalıştırın
3. Bu notebook DICOM'ları modelin anlayacağı formata (.npy) çevirir

### Adım 7: Eğitimi Başlatın

1. `03_training.ipynb` notebook'unu açın
2. Üst menüden "Runtime" → "Run all" (veya Ctrl+F9) tıklayın
3. Her hücre sırayla çalışacak
4. Eğitim ilerlemesini ekranda göreceksiniz

### Adım 8: Modeli İndirin

Eğitim tamamlandığında:
1. Sol panelde "Files" (dosya simgesi) tıklayın
2. `results` klasörünü açın
3. `G_epoch_50.h5` dosyasını bulun
4. Sağ tıklayıp "Download" seçin

Bu dosya artık masaüstü uygulamasında kullanılabilir!

---

## 💻 Kendi Bilgisayarınızda Çalıştırma

Eğer Colab yerine kendi bilgisayarınızda eğitim yapmak istiyorsanız, güçlü bir GPU'nuz olmalı.

### Gereksinimler

- ✅ NVIDIA GPU (en az 8GB VRAM önerilir)
- ✅ CUDA ve cuDNN kurulu
- ✅ Python 3.9+

> ⚠️ **Uyarı**: GPU olmadan eğitim **çok yavaş** olacaktır (günler sürebilir).

### Adım Adım

1. **Projeyi indirin** (yukarıdaki masaüstü uygulaması adımlarına bakın)

2. **Sanal ortam oluşturun** (önerilir):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   # veya
   venv\Scripts\activate     # Windows
   ```

3. **Bağımlılıkları kurun**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verilerinizi hazırlayın**:
   - DICOM dosyalarınızı `data/raw/` klasörüne koyun
   - Jupyter Notebook'u açın: `jupyter notebook`
   - `notebooks/02_data_preprocessing.ipynb` çalıştırın

5. **Eğitimi başlatın**:
   - `notebooks/03_training.ipynb` açın ve çalıştırın

---

## ❓ Sık Sorulan Sorular

### "GPU'um yok, eğitim yapabilir miyim?"

**Kısa cevap**: Google Colab kullanın! Ücretsiz GPU sağlar.

**Uzun cevap**: CPU ile eğitim teknik olarak mümkün ama pratik değil. Colab'da 4-6 saat süren eğitim, CPU'da günlerce sürebilir.

---

### "DICOM dosyası nereden bulurum?"

- **Hastane sistemleri**: PACS sistemlerinden dışa aktarabilirsiniz
- **Araştırma veri setleri**: 
  - [Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
  - Mayo Clinic LDCT dataset

---

### "Hata alıyorum, ne yapmalıyım?"

**En yaygın hatalar:**

| Hata | Çözüm |
|------|-------|
| `No module named 'tensorflow'` | `pip install tensorflow` çalıştırın |
| `CUDA out of memory` | Batch size'ı küçültün (2 veya 1) |
| `FileNotFoundError` | Dosya yollarını kontrol edin |
| PyQt5 hatası (Mac) | Terminale: `export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/anaconda3/plugins/platforms` |

---

### "Model dosyası (.h5) nerede?"

Proje ana klasöründe `G_epoch_50.h5` dosyası bulunmalı. Eğer yoksa:
1. Releases sayfasından indirin
2. veya Colab'da eğitim yapıp indirin

---

### "Kendi CT görüntülerimi kullanabilir miyim?"

Evet! Görüntüleriniz DICOM formatında olmalı. Uygulama otomatik olarak:
1. DICOM'u okur
2. Hounsfield Unit'e çevirir
3. Modele gönderir
4. Sonucu gösterir

---

## 📖 Terimler Sözlüğü

| Terim | Açıklama |
|-------|----------|
| **CT** | Computed Tomography - Bilgisayarlı Tomografi. X-ışınları kullanarak vücudun kesit görüntülerini oluşturur. |
| **LDCT** | Low-Dose CT - Düşük radyasyon dozuyla çekilen CT. Daha güvenli ama daha gürültülü. |
| **DICOM** | Digital Imaging and Communications in Medicine - Tıbbi görüntüleme standart formatı. |
| **HU** | Hounsfield Unit - CT görüntülerinde doku yoğunluğunu gösteren birim. Su=0, Hava=-1000, Kemik=+1000 |
| **GPU** | Graphics Processing Unit - Ekran kartı. Yapay zeka eğitimi için çok hızlı işlem yapar. |
| **GAN** | Generative Adversarial Network - İki sinir ağının yarıştığı bir yapay zeka modeli. |
| **Pix2Pix** | Görüntüden görüntüye çeviri yapan bir GAN türü. |
| **WGAN-GP** | Wasserstein GAN with Gradient Penalty - Daha kararlı eğitim sağlayan GAN varyantı. |
| **PSNR** | Peak Signal-to-Noise Ratio - Görüntü kalitesi metriği. Yüksek = İyi. |
| **SSIM** | Structural Similarity Index - Yapısal benzerlik metriği. 1'e yakın = İyi. |
| **Epoch** | Eğitimde tüm veri setinin bir kez işlenmesi. |
| **Batch Size** | Aynı anda işlenen görüntü sayısı. |
| **NPY** | NumPy array formatı - Python'da sayısal verileri saklamak için kullanılır. |
| **Colab** | Google Colaboratory - Ücretsiz online Python ve GPU ortamı. |

---

## 🆘 Hala Yardım Lazım mı?

1. 📖 [Gelişmiş Tutorial](TUTORIAL_TR.md)'a bakın
2. 📝 [README](README.md) dosyasını inceleyin
3. 🐛 GitHub'da [Issue açın](https://github.com/Haymr/LDCT-Denoising-Pix2Pix-WGANGP/issues)

---

*Bu rehber, LDCT Denoising projesinin bir parçasıdır. MIT Lisansı altında dağıtılmaktadır.*
