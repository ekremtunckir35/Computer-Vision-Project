import streamlit as st
import cv2
import time
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Streamlit app title
st.title("Mekansal Birey Kalma Süresi Analizi")

# Sidebar for video upload
st.sidebar.title("Video Yükleme")
uploaded_video = st.sidebar.file_uploader("Lütfen analiz edilecek videoyu yükleyin", type=["mp4", "avi"])

# YOLOv8 Modelini Yükleme
model = YOLO("../../yolov8n.pt")

# Bölge tanımları (örneğin, 4 bölge: sol üst, sağ üst, sol alt, sağ alt)
zones = {
    "Zone 1": [(0, 0), (320, 240)],
    "Zone 2": [(320, 0), (640, 240)],
    "Zone 3": [(0, 240), (320, 480)],
    "Zone 4": [(320, 240), (640, 480)],
}

if uploaded_video is not None:
    # Video dosyasını OpenCV ile okuma
    video_bytes = uploaded_video.read()
    video_path = '../../uploaded_video.mp4'
    with open(video_path, 'wb') as video_file:
        video_file.write(video_bytes)
    
    # OpenCV ile videoyu yükleyin
    cap = cv2.VideoCapture(video_path)

    # Kalma süresi takibi için dictionary'ler
    stay_times = {}
    stay_durations = {}
    zone_durations = {zone: 0 for zone in zones}

    # Video analizine başla
    st.subheader("Video Analizi")
    st.text("Video analiz ediliyor, lütfen bekleyin...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO ile nesne tespiti yap
        results = model(frame)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                id = box.id

                # Bireyin ID'sini belirle
                if id not in stay_times:
                    stay_times[id] = time.time()  # İlk tespit zamanı
                
                # Bu alanda kalma süresini hesapla
                stay_duration = time.time() - stay_times[id]
                stay_durations[id] = stay_duration  # Süreyi kaydet
                
                # Bireyin bulunduğu bölgeyi tespit et ve kalma süresini kaydet
                for zone_name, ((zx1, zy1), (zx2, zy2)) in zones.items():
                    if zx1 <= box.xyxy[0][0] <= zx2 and zy1 <= box.xyxy[0][1] <= zy2:
                        zone_durations[zone_name] += stay_duration

                # Görselleştirme
                cv2.putText(frame, f"ID: {id}, Time: {stay_duration:.2f}s", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Frame'i görüntüleme
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()

    # Kalma süreleri ve bölge analiz sonuçlarını göster
    st.subheader("Kalma Süresi Analizi")
    st.write("Bireylerin hangi bölgelerde ne kadar süre kaldıklarını gösteren analiz sonuçları:")

    # Bölge sürelerini bar grafiği ile göster
    zones_list = list(zone_durations.keys())
    durations = list(zone_durations.values())

    fig, ax = plt.subplots()
    ax.bar(zones_list, durations, color='skyblue')
    ax.set_xlabel('Bölgeler')
    ax.set_ylabel('Toplam Kalma Süresi (saniye)')
    ax.set_title('Bölgelerde Kalma Süresi Analizi')
    st.pyplot(fig)

    # Bireylerin toplam kalma sürelerini listele
    st.subheader("Birey Bazlı Kalma Süreleri")
    for id, duration in stay_durations.items():
        st.write(f"Birey {id}: Toplamda {duration:.2f} saniye")

else:
    st.write("Lütfen analiz edilecek bir video yükleyin.")

# Footer
st.sidebar.markdown("""
---
Mekansal Birey Kalma Süresi Analizi
Geliştirici: [Ekrem Tunckir](https://yourportfolio.com)
""")
