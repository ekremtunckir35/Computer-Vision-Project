import streamlit as st
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Sayfa başlığı ve açıklama
st.title("Mekansal Birey Kalma Süresi Analizi")
st.write("""
Bu uygulama, belirli bir alanda bireylerin ne kadar süre kaldığını ve bu sürelerin mekânsal dağılımını analiz eder. YOLOv8 modeli kullanılarak bireylerin tespiti yapılır ve her bireyin kalma süresi hesaplanır.
Mekansal Birey Kalma Süresi Analizi, belirli bir alanda bireylerin ne kadar süre kaldığını ve bu sürelerin mekânsal dağılımını inceleyen bir veri analiz yöntemidir. Bu tür bir analiz, genellikle aşağıdaki amaçlarla yapılır:
Mekansal Hareketlilik Anlamlandırma: Analiz, belirli bir alandaki bireylerin hareketlilik paternlerini anlamaya yardımcı olur. Örneğin, bir alışveriş merkezi, park, havaalanı gibi alanlarda insanların hangi bölgelerde daha uzun süre kaldığını ve hangi bölgeleri hızlıca geçtiklerini tespit edebilir.

Yoğunluk ve Kalabalık Yönetimi: Bireylerin mekanda ne kadar süre kaldığını analiz ederek, yoğunluk alanları tespit edilebilir. Bu bilgi, kalabalık yönetimi, kaynakların daha verimli kullanımı ve hizmetlerin optimizasyonu için kullanılabilir.

Müşteri Davranışlarını Anlamak: Perakende sektöründe, müşteri davranışlarını anlamak için bu tür analizler yapılır. Örneğin, bir mağazada hangi reyonlarda müşterilerin daha uzun süre vakit geçirdiği veya hangi alanların daha az ilgi çektiği bu analizle tespit edilebilir.

Guvenlik ve Gözetim: Özellikle havaalanları, tren istasyonları gibi yüksek güvenlikli bölgelerde, bireylerin mekânsal kalış sürelerinin analiz edilmesi, güvenlik politikalarının ve gözetim stratejilerinin iyileştirilmesine yardımcı olabilir.
Bu tür bir analizde, kişileri tespit etmek ve takip etmek için genellikle nesne tespiti algoritmaları ve bilgisayarla görme teknikleri kullanılır.

Gerekenler:
•	Python kurulu bir ortam (tercihen Jupyter Notebook).
•	OpenCV ve YOLOv8 kütüphaneleri yüklü olmalı.
•	Roboflow'dan eğitilmiş bir YOLOv8 modeli.

""")

# Bölümler arasında gezinti
section = st.sidebar.selectbox("Bölüm Seçin", ["Giriş", "Analiz", "Sonuçlar ve Rapor", "Video Analizi"])

# Giriş Bölümü
if section == "Giriş":
    st.header("Giriş")
    st.write("""
    Bu analiz, özellikle yoğun alanlarda bireylerin hareketliliğini ve kalma sürelerini anlamaya yardımcı olur. 
    Güvenlik, yoğunluk yönetimi ve müşteri davranışlarını anlama gibi çeşitli alanlarda uygulanabilir.
    """)

# Analiz Bölümü
elif section == "Analiz":
    st.header("Analiz")
    st.write("""
    YOLOv8 nesne tespiti modeli kullanılarak bireylerin video üzerinden tespiti yapılır ve bu bireylerin belirli bir alanda ne kadar süre kaldıkları hesaplanır.
    """)

    # Video Yükleme ve İşleme
    uploaded_video = st.file_uploader("Bir video yükleyin", type=["mp4", "avi"])

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()

        # OpenCV ile video işleme
        st.video(video_bytes)  # Videoyu göster

        # Analiz işlemlerini başlat
        st.write("Video işleniyor...")

        # YOLO modelini yükle
        model = YOLO("../../yolov8n.pt")

        # Video üzerinde nesne tespiti ve kalma süresi analizi
        cap = cv2.VideoCapture(uploaded_video.name)

        stay_times = {}
        stay_durations = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO ile nesne tespiti
            results = model(frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    id = box.id

                    if id not in stay_times:
                        stay_times[id] = time.time()

                    stay_duration = time.time() - stay_times[id]
                    stay_durations[id] = stay_duration

                    # Görselleştirme
                    cv2.putText(frame, f"ID: {id}, Time: {stay_duration:.2f}s", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            st.image(frame, channels="BGR")  # Kareyi Streamlit'te göster

        cap.release()

        # Kalma sürelerini göster
        st.write("Bireylerin kalma süreleri:")
        for id, duration in stay_durations.items():
            st.write(f"Birey {id} toplamda {duration:.2f} saniye kaldı.")

# Sonuçlar ve Rapor Bölümü
elif section == "Sonuçlar ve Rapor":
    st.header("Sonuçlar ve Rapor")
    st.write("""
    Bireylerin hangi bölgelerde daha uzun süre kaldığı ve hangi alanların daha yoğun olduğu bu analizle belirlenmiştir. 
    Bu tür analizler, güvenlik ve yoğunluk yönetimi açısından önemlidir.
    """)

    # Örnek grafik oluşturma
    st.write("Bölgelerde Kalma Süresi Analizi")
    zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
    durations = [30, 90, 0, 0]

    plt.bar(zones, durations)
    plt.xlabel('Bölgeler')
    plt.ylabel('Toplam Kalma Süresi (saniye)')
    plt.title('Bölgelerde Kalma Süresi Analizi')
    st.pyplot(plt)

    st.write(""" Grafikteki dağılım incelendiğinde, yatay eksende (X ekseni) toplam kalma süresi (saniye cinsinden) yer almakta ve dikey eksende (Y ekseni) ise frekans (birey sayısı) gösterilmektedir. Görselde yalnızca bir adet sütun bulunmakta ve bu sütun, X ekseninde sıfır değerine karşılık gelmektedir.
Bu durum, verilerdeki kalma süresinin neredeyse sıfır olduğunu, yani analiz edilen bireylerin tespit edilen alanda çok kısa bir süre kaldıklarını veya tespit edilemediklerini göstermektedir. Ayrıca, kalma süresinin dağılımında çeşitlilik olmadığını da gözlemleyebiliriz; tüm veriler aynı noktada toplanmış.
Muhtemel Nedenler:
Tespit Edilen Bireylerin Hareketliliği: Bireyler çok hızlı hareket ediyor olabilir, bu da tespit edilen kalma süresini minimum seviyede tutmuş olabilir.
Algoritmanın Yetersizliği: YOLO modelinin tespit yapamaması veya hatalı tespit yapması nedeniyle bireylerin kalma süreleri sıfır olarak kaydedilmiş olabilir.
Veri Setinin Doğası: Veri seti, çok kısa süreli video karelerinden oluşuyor olabilir ve bu durum, tespit edilen kalma sürelerinin çok düşük olmasına yol açmış olabilir.
Sonuç olarak, grafik bize bireylerin analiz edilen alanda neredeyse hiç vakit geçirmediğini gösteriyor.""")

# Video Analizi Bölümü
elif section == "Video Analizi":
    st.header("Video Analizi")
    st.write("""
    Bu bölümde, video verisi üzerinden analiz yapılır. Bireylerin hangi bölgelerde daha uzun süre kaldığı belirlenir.
    """)

    # Video yükleme ve işleme
    uploaded_video = st.file_uploader("Bir video yükleyin", type=["mp4", "avi"])

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        st.video(video_bytes)
