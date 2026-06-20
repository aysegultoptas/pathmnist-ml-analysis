# PathMNIST Veri Kümesinde Hibrit Öznitelik Çıkarımı, Meta-Sezgisel Öznitelik Seçimi ve Sınıflandırma Analizi

Bu proje, **PathMNIST** (MedMNIST v2) histopatoloji görüntü veri kümesi üzerinde gerçekleştirilmiş; el tasarımı (handcrafted) hibrit doku/renk özniteliklerinin çıkarılmasını, meta-sezgisel ve istatistiksel yöntemlerle optimize edilmesini ve güçlü makine öğrenmesi sınıflandırıcıları ile analiz edilmesini kapsayan iki aşamalı (2-Step Pipeline) uçtan uca bir akademik araştırma ve uygulama boru hattıdır.

---

## 🚀 Proje Öne Çıkanları ve Kritik Bulgular

* **En Optimal Model Kombinasyonu:** Yapılan deneysel analizler sonucunda, **LASSO (L1-Gömülü) Öznitelik Seçimi + LightGBM** kombinasyonu **%86.75 test doğruluğu**, **0.872 Makro F1** ve **0.981 Makro AUC** skorları elde ederek çalışmanın en başarılı modeli olmuştur.
* **Yüksek Boyut ve Zaman Verimliliği:** Parçacık Sürüsü Optimizasyonu (PSO) tabanlı sarıcı (wrapper) yöntem, öznitelik uzayını **%89 oranında azaltarak** (560 boyuttan 60 boyuta) veri boyutunu dramatik şekilde küçültmüştür. [cite_start]Bu boyut indirgeme sayesinde LightGBM modelinin eğitim süresi, sınıflandırma performansından ödün verilmeksizin **147.9 saniyeden 37.9 saniyeye** düşürülmüştür[cite: 1].
* **Gürültü Filtreleme:** Gömülü bir yöntem olan LASSO, boyutu %45 azaltmasına rağmen gürültülü özellikleri başarıyla elediği için hiçbir özellik elenmeyen ham veri senaryosunu (%86.42) geride bırakmayı başarmıştır.
* **Sınıf Dengesi Yönetimi:** Sınıflandırma adımında `class_weight='balanced'` parametresi entegre edilerek, veri kümesindeki dengesiz dağılımın ve özellikle ayırt edilmesi zor olan doku yapılarının olumsuz etkisi minimize edilmiştir.

---

## 📊 Veri Kümesi: PathMNIST
PathMNIST, kolon kanseri histopatoloji görüntülerinden türetilmiş, 9 farklı doku sınıfına ait $28 \times 28$ boyutlarında RGB görüntüler içeren bir MedMNIST alt kümesidir. Projede yer alan sınıflar şunlardır:
1. **Adipose** (Yağ Dokusu)
2. **Background** (Arka Plan)
3. **Debris** (Hücresel Atıklar)
4. **Lymphocytes** (Lenfositler)
5. **Mucus** (Mukus)
6. **Smooth Muscle** (Düz Kas)
7. **Normal Colon Mucosa** (Normal Kolon Mukozası)
8. **Cancer-Assoc. Stroma** (Kanserle İlişkili Stroma)
9. **Colorectal Adenocarcinoma** (Kolorektal Adenokarsinom)

---

## 🛠️ Sistem Mimarisi ve Metodoloji

Proje, birbirini takip eden iki ana modülden oluşmaktadır:

### 1. Aşama: Öznitelik Çıkarımı ve Seçimi (`feature_extraction.py`)
* **Veri Artırımı (Augmentation):** Eğitim setindeki sınıf dengesizliğini gidermek adına, azınlık sınıflarına rotasyon, dikey/yatay aynalama, gama ayarı ve parlaklık manipülasyonları uygulanarak tüm sınıflar en yüksek örnek sayısına sahip sınıf düzeyinde dengelenir.
* **Hibrit Öznitelik Çıkarımı:** Görüntülerden toplam **560 boyutlu** öznitelik vektörü inşa edilir:
  * **HOG (Histogram of Oriented Gradients):** Hücre sınırlarını ve yönelimli kenar bilgilerini yakalamak için 144 öznitelik.
  * **LBP (Local Binary Patterns):** Mikroskobik yüzey ve doku dokusunu temsil etmek için 256 öznitelik.
  * **Renk Histogramı (RGB):** Adipose ve Mucus gibi renk bağımlı sınıfların ayırt edilebilmesi için kanal başına 32 bin (toplam 96 öznitelik).
  * **Gabor Filtreleri:** *Smooth Muscle* ve *Cancer-Assoc. Stroma* gibi lifli doku yapılarının açısal özelliklerini tam kapsamak adına **8 farklı theta açısı** ve 4 frekansta ortalama/standart sapma değerleri alınarak **64 öznitelik** çıkarılır.
* **Normalizasyon:** Grup bazlı hataları önlemek amacıyla tüm birleşik matris tek bir global `StandardScaler` ile ölçeklenir.
* **Öznitelik Seçim Yöntemleri:** * *Pearson Korelasyonu:* İstatiksel filtreleme ile en güçlü %25'lik kesim seçilir.
  * *CFS (Correlation-based Feature Selection):* Özellik-hedef bağımlılığını maksimize, özellik-özellik çoklu doğrusal bağlantısını minimize eder. Erken durma toleransı (`MERIT_TOLERANCE = 0.005`) eklenerek daha fazla derinlemesine keşif yapması sağlanmıştır.
  * *LASSO (L1 Regresyonu):* Önemsiz katsayıları sıfıra eşitleyerek seyreltik alt küme seçer.
  * *PSO (Parçacık Sürüsü Optimizasyonu):* Arama uzayı en iyi 100 Pearson özniteliğine genişletilmiş; fitness proxy modeli olarak KNN yerine LightGBM ile daha uyumlu çalışan ağaç tabanlı **RandomForest Lite** modeline geçilmiştir.

### 2. Aşama: Sınıflandırma ve Performans Analizi (`classify.py`)
Seçilen alt kümelerin tamamı (Tüm Öznitelikler, Pearson, CFS, LASSO, PSO) aşağıdaki hiper-parametreleri optimize edilmiş modellerle eğitilir:
* **SVM:** RBF çekirdeği ($C=100$) ve dengeli sınıf ağırlıkları ile doğrusal olmayan sınır ayrımı.
* **Random Forest:** 300 karar ağacı ve kök-kare özellik seçimiyle kolektif öğrenme.
* **LightGBM:** Geliştirilmiş mimari yapısıyla 500 ağaç (`n_estimators=500`), daha kararlı genelleme için düşük öğrenme oranı (`learning_rate=0.03`) ve derin ağaç yapısı (`num_leaves=127`) kullanılarak karmaşık dokuların yüksek doğrulukla çözülmesi sağlanmıştır.

---

## 📁 Klasör Yapısı

Projenin kök dizininde yer alan mimari yapı şu şekildedir:
pathmnist-ml-analysis/
├── feature_extraction.py     # Adım 1: Veri indirme, artırma, öznitelik çıkarma ve seçim kodları
├── classify.py               # Adım 2: Model eğitimi, test etme, kaydetme ve metrik analiz kodları
├── requirements.txt          # Projenin çalışabilmesi için gerekli kütüphane listesi
└── README.md                 # Proje genel tanıtımı ve kullanım kılavuzu

Çalıştırma esnasında oluşturulacak çıktı klasör yapısı:

outputs/
├── features/                 # Adım 1 çıktısı: Ölçeklenmiş matrisler (.npz) ve seçilen indeksler
└── results/                  # Adım 2 çıktısı: Raporlar (.csv), eğitilmiş modeller (.pkl) ve grafikler

---

## 📦 Kurulum ve Çalıştırma
1. Bağımlılıkların Yüklenmesi
Gerekli tüm kütüphaneleri sanal ortamınıza indirmek için terminalde aşağıdaki komutu çalıştırın:

pip install -r requirements.txt
2. Adım 1: Özniteliklerin Çıkarılması ve Seçilmesi
PathMNIST veri kümesini otomatik olarak indirmek, veri artırımını uygulamak ve öznitelik seçim süreçlerini başlatmak için şu komutu yürütün:

python feature_extraction.py --download
Bu işlem tamamlandığında ölçeklenmiş öznitelik dosyalarınız ve seçilen indeks matrisleri outputs/features/ klasörüne sıkıştırılmış .npz formatında kaydedilir.

3. Adım 2: Sınıflandırma ve Model Performans Raporlama
Bir önceki adımda üretilen öznitelik klasörünü girdi olarak alıp 15 farklı kombinasyonun SVM, RandomForest ve LightGBM modellerini eğitmek, en iyi modeli bulmak ve performans matrislerini .csv formatında almak için şu komutu çalıştırın:

python classify.py
Eğitim bittiğinde test doğruluğu özet tablosu konsola basılacak, tüm detaylar ile eğitim süreleri (results.csv ve timing.csv) outputs/results/ altına yazılacaktır.

💡 Görselleştirme ve Grafik Notu
Kod yapıları, terminal veya TRUBA gibi uzak sunucu ortamlarında diski gereksiz resim dosyalarıyla doldurmadan ışık hızında ve sorunsuz çalışabilmesi için varsayılan olarak hafif modda ayarlanmıştır (GRAFIKLERI_CIZ = False).

Eğer projenin ürettiği akademik analiz grafiklerini, 15 adet karmaşıklık matrisini (Confusion Matrix), radar grafiklerini, sınıf bazlı F1 kırılımlarını ve Random Forest öznitelik önem analizlerini otomatik olarak diske yazdırmak istiyorsanız; hem feature_extraction.py hem de classify.py kodlarının en üstünde yer alan kütüphane importlarının hemen altındaki şu satırı True konumuna getirmeniz yeterlidir:

# Akademik grafikleri çizdirerek outputs/ klasörüne kaydetmek için True yapın
GRAFIKLERI_CIZ = True
