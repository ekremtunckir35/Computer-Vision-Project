import streamlit as st
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Streamlit Ayarları ve Sayfa Başlığı
st.set_page_config(page_title="Mekansal Birey Kalma Süresi Analizi", layout="wide")
st.markdown(
    """
    <style>
    body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #f8f9fa;
        padding-top: 20px;
    }
    .main-title {
        text-align: center;
        color: #007bff;
        font-weight: bold;
    }
    .section-title {
        color: #495057;
        font-weight: bold;
        font-size: 24px;
    }
    .info-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sayfa Başlığı ve Açıklama
st.markdown("<h1 class='main-title'>Mekansal Birey Kalma Süresi Analizi</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Bu uygulama, havaalanı gibi alanlarda bireylerin belirli bölgelerde ne kadar süre kaldığını analiz eder.</p>",
    unsafe_allow_html=True)

# Bölüm Seçimi
section = st.sidebar.selectbox("Bölüm Seçin", ["Giriş", "Analiz", "Sonuçlar ve Rapor", "Video Analizi"])

# Giriş Bölümü
if section == "Giriş":
    st.markdown("<h2 class='section-title'>Giriş</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-section">
    Bu analiz, yoğun alanlarda bireylerin hareketliliğini ve kalma sürelerini anlamaya yardımcı olur. Güvenlik, yoğunluk yönetimi ve müşteri davranışlarını anlama gibi çeşitli alanlarda uygulanabilir.
    </div>
    """, unsafe_allow_html=True)

# Analiz Bölümü
elif section == "Analiz":
    st.markdown("<h2 class='section-title'>Analiz</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-section">
    YOLOv8 nesne tespiti modeli kullanılarak bireylerin video üzerinden tespiti yapılır ve bu bireylerin belirli bir alanda ne kadar süre kaldıkları hesaplanır.
    </div>
    """, unsafe_allow_html=True)

    # Video Yükleme ve İşleme
    uploaded_video = st.file_uploader("Bir video yükleyin", type=["mp4", "avi"])

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        st.video(video_bytes)  # Videoyu göster

        st.markdown("### Video İşleniyor...")

        # YOLO modelini yükle
        model = YOLO("yolov8n.pt")

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
        st.markdown("### Bireylerin kalma süreleri:")
        for id, duration in stay_durations.items():
            st.write(f"Birey {id} toplamda {duration:.2f} saniye kaldı.")

# Sonuçlar ve Rapor Bölümü
elif section == "Sonuçlar ve Rapor":
    st.markdown("<h2 class='section-title'>Sonuçlar ve Rapor</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-section">
    Bireylerin hangi bölgelerde daha uzun süre kaldığı ve hangi alanların daha yoğun olduğu bu analizle belirlenmiştir. Bu tür analizler, güvenlik ve yoğunluk yönetimi açısından önemlidir.
    </div>
    """, unsafe_allow_html=True)

    # Örnek grafik oluşturma
    st.markdown("#### Bölgelerde Kalma Süresi Analizi")
    zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
    durations = [30, 90, 0, 0]

    plt.bar(zones, durations)
    plt.xlabel('Bölgeler')
    plt.ylabel('Toplam Kalma Süresi (saniye)')
    plt.title('Bölgelerde Kalma Süresi Analizi')
    st.pyplot(plt)

# Video Analizi Bölümü
elif section == "Video Analizi":
    st.markdown("<h2 class='section-title'>Video Analizi</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-section">
    Bu bölümde, video verisi üzerinden analiz yapılır. Bireylerin hangi bölgelerde daha uzun süre kaldığı belirlenir.
    </div>
    """, unsafe_allow_html=True)

    # Video yükleme ve işleme
    uploaded_video = st.file_uploader("Bir video yükleyin", type=["mp4", "avi"])

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        st.video(video_bytes)
