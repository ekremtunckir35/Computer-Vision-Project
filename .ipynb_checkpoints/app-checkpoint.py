import streamlit as st
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from roboflow import Roboflow

# Streamlit başlığı ve açıklaması
st.title("Mekansal Birey Kalma Süresi Analizi")
st.write("""
Bu uygulama, belirli bir video üzerinde bireylerin kalma sürelerini analiz eder ve bu bilgiyi görselleştirir.
""")

# Roboflow API anahtarıyla bağlanma
rf = Roboflow(api_key="0i8v0fEn4mty6Zt83ACV")
project = rf.workspace("myworkspace-bkrfh").project("analiz-ssev8")
version = project.version(2)
dataset = version.download("yolov8")

# YOLOv8 Modelini Yükleme
model = YOLO("yolov8n.pt")

# Video yükleme
video_file = st.file_uploader("Bir video dosyası yükleyin", type=["mp4", "avi", "mov"])

if video_file:
    st.video(video_file)

    cap = cv2.VideoCapture(video_file.name)

    stay_times = {}
    stay_durations = {}

    st.write("Video işleniyor, lütfen bekleyin...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO ile nesne tespiti yap
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Her tespit için bbox verilerini al
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                id = box.id

                # Bireyin ID'sini belirle
                if id not in stay_times:
                    stay_times[id] = time.time()  # İlk tespit zamanı

                # Bu alanda kalma süresini hesapla
                stay_duration = time.time() - stay_times[id]
                stay_durations[id] = stay_duration  # Süreyi kaydet

                # Görselleştirme
                cv2.putText(frame, f"ID: {id}, Time: {stay_duration:.2f}s", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Frame'i göster
        st.image(frame, channels="BGR")
    
    cap.release()

    st.write("Analiz Tamamlandı!")

    # Kalma sürelerinin sonuçlarını görselleştirme
    st.subheader("Kalma Süresi Analizi Sonuçları")

    # Bireylerin toplam kalma sürelerini yazdır
    for id, duration in stay_durations.items():
        st.write(f"Birey {id} toplamda {duration:.2f} saniye kaldı.")

    # Toplam kalma süresini hesapla
    total_duration = np.sum(list(stay_durations.values()))
    st.write(f"Tüm bireylerin toplamda {total_duration:.2f} saniye kaldığı tespit edildi.")

    # Görselleştirme: Kalma süresi dağılımı
    st.subheader("Kalma Süresi Dağılımı")
    durations = list(stay_durations.values())
    plt.hist(durations, bins=5)
    plt.xlabel('Toplam Kalma Süresi (saniye)')
    plt.ylabel('Frekans')
    plt.title('Kalma Süresi Dağılımı')
    st.pyplot(plt)

    # Bölgesel analiz ve ısı haritası
    st.subheader("Bölgesel Kalma Süresi Analizi")

    zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
    zone_durations = [np.random.rand() * 100 for _ in zones]  # Örnek veri, kendi bölgesel analizinizi ekleyin

    plt.bar(zones, zone_durations)
    plt.xlabel('Bölgeler')
    plt.ylabel('Toplam Kalma Süresi (saniye)')
    plt.title('Bölgelerde Kalma Süresi Analizi')
    st.pyplot(plt)

    # Isı haritası
    data = np.array(zone_durations).reshape(1, -1)
    sns.heatmap(data, annot=True, xticklabels=zones, yticklabels=['Kalma Süresi'])
    st.pyplot(plt)
