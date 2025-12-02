# 📷 AI Background Blur (Portrait Mode)

<div align="center">
  <a href="#-english">🇺🇸 <strong>English</strong></a> | 
  <a href="#-türkçe">🇹🇷 <strong>Türkçe</strong></a>
</div>

<div align="center">
  <br>
</div>

---

<a name="-english"></a>
## 🇺🇸 English

**AI Background Blur** is a real-time computer vision application that automatically detects the person in the video feed and blurs the background. It mimics the "Portrait Mode" found in smartphones or the "Blur Background" feature in Zoom/Google Meet.

### 🌟 Key Features
* **Real-Time Segmentation:** Uses Google's MediaPipe Selfie Segmentation model to separate the subject from the background instantly.
* **Privacy Focused:** Perfect for video calls or recording where you want to hide your surroundings.
* **Gaussian Blur:** Applies a smooth, professional-looking blur effect to the non-human areas of the frame.
* **Lightweight:** Runs smoothly on CPU without needing a heavy GPU.

### 🛠 Tech Stack
* **Python**
* **OpenCV** (Image manipulation & Blur)
* **MediaPipe** (AI Segmentation Model)
* **NumPy** (Masking operations)

### 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/emreefeyuksel/AI-Background-Blur.git](https://github.com/emreefeyuksel/AI-Background-Blur.git)
    cd AI-Background-Blur
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python blur_bg.py
    ```
    *(Make sure your python file is named `blur_bg.py` or `main.py`)*

### 🎮 Controls
* **'q':** Quit the application.

---

<a name="-türkçe"></a>
## 🇹🇷 Türkçe

**AI Background Blur**, video görüntüsündeki kişiyi otomatik olarak algılayan ve arka planı bulanıklaştıran gerçek zamanlı bir bilgisayarlı görü (computer vision) uygulamasıdır. Akıllı telefonlardaki "Portre Modu"nu veya Zoom/Google Meet'teki "Arka Planı Bulanıklaştır" özelliğini taklit eder.

### 🌟 Temel Özellikler
* **Gerçek Zamanlı Bölütleme:** Kişiyi arka plandan anında ayırmak için Google MediaPipe Selfie Segmentation modelini kullanır.
* **Gizlilik Odaklı:** Çevrenizi gizlemek istediğiniz video görüşmeleri veya kayıtlar için idealdir.
* **Gaussian Blur:** Görüntünün arka plan kısımlarına pürüzsüz ve profesyonel görünen bir bulanıklık efekti uygular.
* **Hafif:** Güçlü bir GPU'ya ihtiyaç duymadan işlemci (CPU) üzerinde akıcı bir şekilde çalışır.

### 🛠 Kullanılan Teknolojiler
* **Python**
* **OpenCV** (Görüntü işleme ve Bulanıklaştırma)
* **MediaPipe** (AI Segmentasyon Modeli)
* **NumPy** (Maskeleme işlemleri)

### 🚀 Kurulum ve Kullanım

1.  **Projeyi indirin:**
    ```bash
    git clone [https://github.com/emreefeyuksel/AI-Background-Blur.git](https://github.com/emreefeyuksel/AI-Background-Blur.git)
    cd AI-Background-Blur
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı başlatın:**
    ```bash
    python blur_bg.py
    ```

### 🎮 Kontroller
* **'q':** Uygulamadan çıkış yap.

---
<div align="center">
  Developed by <a href="https://github.com/emreefeyuksel">YOUR NAME</a>
</div>
