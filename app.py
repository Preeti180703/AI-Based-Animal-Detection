import streamlit as st
import cv2
import numpy as np
from detect import detect_animals

st.title("🐾 Animal Detection & Carnivore Alert System")

option = st.radio("Choose Input Type:", ["Image", "Video"])

# ===== IMAGE =====
if option == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption="Original Image", channels="BGR")

        output, count = detect_animals(img)

        st.image(output, caption="Detected Output", channels="BGR")

        if count > 0:
            st.error(f"⚠ Carnivores Detected: {count}")
        else:
            st.success("No carnivores detected")

# ===== VIDEO =====
elif option == "Video":
    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file:
        tfile = open("temp.mp4", "wb")
        tfile.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")

        stframe = st.empty()

        carnivore_total = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output, count = detect_animals(frame)
            carnivore_total += count

            stframe.image(output, channels="BGR")

        cap.release()

        st.warning(f"Total Carnivores Detected: {carnivore_total}")