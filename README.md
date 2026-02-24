# PyTorch MNIST Classification & Regularization Analysis

Bu proje, PyTorch kullanılarak MNIST veri seti üzerinde bir Çok Katmanlı Algılayıcı (MLP) modelinin eğitilmesini ve farklı hiperparametre/regularization tekniklerinin modelin **genelleştirme (generalization)** yeteneğine etkisini detaylıca incelemektedir.

## 📌 Projenin Amacı
Sadece yüksek doğruluk (accuracy) elde etmek değil; Overfitting (aşırı öğrenme) sorununu önleyerek, modelin görmediği test verisinde en iyi şekilde çalışmasını sağlayacak "ideal" parametreleri bulmaktır.

## 🧪 Yapılan Deneyler ve İzole Karşılaştırmalar

Deneyler, değişkenlerin etkisini net görebilmek için izole edilerek yapılmıştır.

### 1. L2 Regularization (Weight Decay) Analizi
* **Durum:** Dropout olmadan, `weight_decay=0.0` ile `weight_decay=1e-4` karşılaştırıldı.
* **Sonuç & Neden:** Hiçbir regülarizasyon olmadığında (Baseline), eğitim doğruluğu %100'e yaklaşırken test doğruluğu geride kalır (Train-Test Gap büyüktür). L2 regülarizasyonu eklemek, büyük ağ ağırlıklarını cezalandırarak modelin verideki gereksiz gürültüleri ezberlemesini önledi ve Test başarısını artırdı.

### 2. Aktivasyon Fonksiyonları Karşılaştırması
* **Deneyler:** `ReLU`, `Tanh` ve `LeakyReLU` (Standart Dropout ve L2 kullanılarak).
* **Sonuç & Neden:** * **ReLU:** Negatif değerleri sıfırlayarak ağa non-lineerlik katar. "Vanishing Gradient" problemini çözdüğü için en istikrarlı ve hızlı öğrenen fonksiyondur.
  * **LeakyReLU:** ReLU'nun aksine negatif değerlerde çok küçük bir eğim bırakır ("Dying ReLU" problemini önler). Genellikle ReLU ile başa baş performans gösterir.
  * **Tanh:** Sığ ağlarda iyi çalışsa da, bu derinlikteki bir MLP'de gradyanların doygunluğa ulaşmasına sebep olabileceği için ReLU türevlerinin biraz gerisinde kalmıştır.

### 3. Dropout Etkisi ve Oran Karşılaştırması
* **Deneyler:** Dropout yok (0.0), %10 (0.1), %20 (0.2), %30 (0.3) ve %50 (0.5) oranları test edildi.
* **Sonuç & Neden:**
  * Dropout olmadan model hızla overfitting'e gider.
  * Dropout oranı arttıkça, Eğitim (Train) ve Test arasındaki fark (Genelleme Gap'i) azalır.
  * Ancak **Dropout = 0.5** gibi yüksek bir değere çıkıldığında, ağ her adımda nöronların yarısını kaybettiği için "Underfitting" (öğrenememe) yaşamaya başlar ve test doğruluğu düşer.
  * **İdeal Nokta:** Bu mimari için 0.2 ve 0.3 oranları, ezberlemeyi durdururken test başarısını zirveye taşıyan "Sweet Spot" olarak bulunmuştur.

## 🏆 Hangi Teknik En İyi Generalization'ı Sağlıyor?
Deney sonuçlarına göre, en iyi genelleştirme (Minimum Train-Test Gap ve Maksimum Test Accuracy) **L2 Regularization (1e-4) ve Dropout'un (0.2) birlikte kullanıldığı (Act: ReLU)** senaryoda elde edilmektedir. 

Sadece birini seçmek gerekirse; **Dropout**, ağın belirli özelliklere aşırı güvenmesini (co-adaptation) doğrudan engellediği için L2'ye kıyasla Overfitting'i önlemede daha dramatik ve başarılı bir sonuç vermiştir.

---

## 💻 Kurulum ve Çalıştırma

Gerekli kütüphaneleri yükleyin:
```bash
pip install torch torchvision pandas numpy