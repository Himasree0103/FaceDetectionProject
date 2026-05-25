import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load models
face_model = YOLO("models/yolov8n-face-lindevs.pt")
body_model = YOLO("yolov8n.pt")   # detects person/body

st.title("Face and Body Detection")

# Options
option = st.selectbox(
    "Choose option",
    ["Face Detection", "Body Detection", "Resize Image"]
)

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(image, width=400)

    image_np = np.array(image)

    # ---------- FACE ----------
    if option == "Face Detection":

        results = face_model(image_np)

        output = results[0].plot(
            line_width=1,
            font_size=0.5
        )

        count = len(results[0].boxes)

        st.write("Faces detected:", count)

        st.image(output,width=500)


    # ---------- BODY ----------
    elif option == "Body Detection":

        results = body_model(image_np)

        persons = []

        for box in results[0].boxes:

            cls = int(box.cls[0])

            if cls == 0:      # class 0 = person
                persons.append(box)

        count = len(persons)

        output = results[0].plot()

        st.write("Bodies detected:", count)

        st.image(output,width=500)


    # ---------- RESIZE ----------
    elif option == "Resize Image":

        width = st.number_input(
            "Width",
            value=300
        )

        height = st.number_input(
            "Height",
            value=300
        )

        resized = image.resize(
            (int(width),int(height))
        )

        st.image(
            resized,
            caption="Resized Image"
        )