# PyTorch ile MNIST Sınıflandırma: Regularization ve Hiperparametre Analizi

Bu proje, PyTorch kullanılarak MNIST veri seti üzerinde Çok Katmanlı Algılayıcı (MLP) modelinin eğitilmesini içermektedir. Projenin temel amacı sadece modeli eğitmek değil; **farklı aktivasyon fonksiyonlarının, Dropout oranlarının ve L2 Regularization (Weight Decay)** tekniklerinin modelin ezberlemesini (overfitting) nasıl engellediğini ve genelleştirme (generalization) yeteneğini nasıl artırdığını bilimsel bir yaklaşımla analiz etmektir.

## 📊 Deney Sonuçları (Gerçek Zamanlı Çıktılar)

Değişkenlerin etkisini net bir şekilde görebilmek için parametreler izole edilerek test edilmiştir. Aşağıdaki tablo, eğitim sonucunda elde edilen gerçek verileri göstermektedir (Test başarısına göre sıralanmıştır):

| Deney Adı | Aktivasyon | Dropout | L2 (WD) | Train Acc (%) | Test Acc (%) | Gap (Train-Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drop: 0.3** | **ReLU** | **0.3** | **0.0000** | 98.37 | **98.31** | **0.06** |
| L2 Only (1e-4) | ReLU | 0.0 | 0.0001 | 99.14 | 98.23 | 0.91 |
| Drop: 0.1 | ReLU | 0.1 | 0.0000 | 98.99 | 98.11 | 0.88 |
| Drop: 0.2 | ReLU | 0.2 | 0.0000 | 98.72 | 97.99 | 0.73 |
| Act: ReLU (Base)| ReLU | 0.2 | 0.0001 | 98.30 | 97.92 | 0.38 |
| Drop: 0.5 | ReLU | 0.5 | 0.0000 | 96.80 | 97.81 | -1.01 |
| Baseline (No Reg)| ReLU | 0.0 | 0.0000 | **99.36** | 97.79 | 1.57 |
| Act: LeakyReLU | LeakyReLU| 0.2 | 0.0001 | 98.30 | 97.70 | 0.60 |
| Act: Tanh | Tanh | 0.2 | 0.0001 | 97.69 | 97.48 | 0.21 |

---

[Image of overfitting and underfitting in neural networks]

## 🧪 Bulgular ve Teorik Analiz

Elde edilen sonuçlar, derin öğrenme teorisini birebir doğrulamaktadır:

### 1. Overfitting (Aşırı Öğrenme) ve Baseline Modeli
**Baseline (No Reg)** deneyinde hiçbir regülarizasyon kullanılmamıştır. Model eğitim verisinde **%99.36** gibi çok yüksek bir başarıya ulaşmış, ancak test verisinde **%97.79**'da kalmıştır. Aradaki **%1.57'lik fark (Gap)**, modelin eğitim verisindeki gürültüleri ezberlediğinin (Overfitting) en net kanıtıdır.

### 2. L2 Regularization'ın Etkisi
**L2 Only** deneyinde ağ ağırlıklarına ceza (weight decay) uygulandığında, Train-Test arasındaki fark (Gap) %1.57'den **%0.91**'e düşmüş ve test başarısı **%98.23**'e yükselmiştir. L2, modelin ezberlemesini zorlaştırarak daha iyi genelleştirme yapmasını sağlamıştır.

### 3. Dropout Oranları: Neden -1.01% Çıktı?
* **"Sweet Spot" (0.3):** %30 Dropout oranı, modelin ezberlemesini mükemmel bir şekilde durdurmuş, Gap farkını **%0.06**'ya (neredeyse sıfıra) indirerek **%98.31** ile en yüksek test başarısını getirmiştir.
* **Underfitting (-1.01% Gap):** %50 Dropout uygulandığında Test başarısı, Eğitim başarısından daha yüksek çıkmıştır (Negatif Gap). Bunun sebebi, `model.train()` aşamasında nöronların yarısının kapalı olması ve modelin zorlanması, `model.eval()` (test) aşamasında ise tüm nöronların açılarak tam kapasite çalışmasıdır. Ancak genel başarı %97.81'de kaldığı için %50 Dropout'un bu mimari için fazla olduğu (Underfitting) görülmüştür.

### 4. Aktivasyon Fonksiyonları
Bu model ve veri seti (MNIST) için en iyi performansı **ReLU** tabanlı fonksiyonlar göstermiştir. Tanh, gradyan akışında ReLU kadar etkili olamadığı için test başarısında (%97.48) geride kalmıştır.

## 🏆 Sonuç
Bu problem için en ideal konfigürasyon; ağın ezberlemesini engelleyen ancak öğrenme kapasitesini de boğmayan **Aktivasyon: ReLU** ve **Dropout: 0.3** kombinasyonudur.

---

## 💻 Kurulum ve Çalıştırma

Gerekli kütüphaneleri yükleyin:
```bash
pip install torch torchvision pandas numpy