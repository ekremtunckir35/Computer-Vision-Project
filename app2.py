import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow

# Streamlit başlığı ve açıklaması
st.title('Mekansal Birey Kalma Süresi Analizi')
st.write("Bu uygulama, belirli bir alanda bireylerin ne kadar süre kaldığını analiz eder ve sonuçları görselleştirir.")

# Kullanıcıdan video yüklemesi istenir
video_file = st.file_uploader("Bir video dosyası yükleyin", type=["mp4", "avi", "mov"])

# Eğer video yüklendiyse
if video_file is not None:
    # Video kaydetme ve yükleme
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_path)

    # YOLO modelini yükle
    model = YOLO("yolov8n.pt")

    # Video dosyasını yükle
    cap = cv2.VideoCapture(video_path)

    # Kalma süresi takibi için dictionary'ler
    stay_times = {}
    stay_durations = {}

    # Video oynatılmaya başlar
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

                # Görselleştirme
                cv2.putText(frame, f"ID: {id}, Time: {stay_duration:.2f}s", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Görüntü streamlit'te gösterilir
        st.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Toplam kalma süresini hesapla ve sonucu göster
    total_duration = np.sum(list(stay_durations.values()))
    st.write(f"Tüm bireylerin toplamda {total_duration:.2f} saniye kaldığı tespit edildi.")

    # Her bireyin toplam kalma süresini göster
    for id, duration in stay_durations.items():
        st.write(f"Birey {id} toplamda {duration:.2f} saniye kaldı.")
