import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mekansal Birey Kalma Süresi Analizi", layout="wide")

# App title and description
st.title("Mekansal Birey Kalma Süresi Analizi")
st.markdown("""
Bu uygulama, YOLOv8 kullanarak bireylerin mekanda ne kadar süre kaldıklarını analiz eder. Analiz, havaalanı, alışveriş merkezi gibi ortamlarda yoğunluk tespiti ve kalabalık yönetimi amacıyla kullanılabilir.
""")

# Sidebar for settings
st.sidebar.header("Ayarlar")
video_file = st.sidebar.file_uploader("Video Yükleyin", type=["mp4"])
model_option = st.sidebar.selectbox("Modeli Seçin", ["yolov8n", "yolov8s", "yolov8m"])

st.sidebar.markdown("### Ekstra Ayarlar")

# Load and process the video
if video_file is not None:
    st.video(video_file)
    cap = cv2.VideoCapture(video_file.name)

    # Load YOLO model
    model = YOLO(f"{model_option}.pt")
    st.write(f"Yüklenen model: {model_option}")

    # Process the video and calculate stay durations
    stay_times = {}
    stay_durations = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

                cv2.putText(frame, f"ID: {id}, Time: {stay_duration:.2f}s", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        st.image(frame, channels="BGR")

    cap.release()

# Visualization section
st.header("Kalma Süresi Analizi")
if stay_durations:
    durations = list(stay_durations.values())
    st.write(f"Toplam birey sayısı: {len(durations)}")

    # Bar plot
    st.subheader("Kalma Süresi Dağılımı")
    plt.hist(durations, bins=5)
    st.pyplot(plt)

    # Heatmap for zone-based analysis
    st.subheader("Bölgesel Kalma Süresi Analizi")
    data = np.array(durations).reshape(1, -1)
    sns.heatmap(data, annot=True, cmap="Blues", xticklabels=["Zone 1", "Zone 2", "Zone 3", "Zone 4"],
                yticklabels=['Kalma Süresi'])
    st.pyplot(plt)

# Adding CSS styling
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    color: #333;
}
h1 {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)
