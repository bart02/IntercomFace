from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import streamlit as st
from stqdm import stqdm
from ultralytics import YOLO

from detect import detect
from embed import create_pkl


@st.cache_data
def detect_face(cv2_img: np.ndarray):
    model = YOLO("models/yolov8n-face.pt")
    detections = detect(model, cv2_img, 0)
    return detections


st.set_page_config(
    page_title="Add Face",
    page_icon="👨",
)
st.title("👨 Add Face")

p = Path("faces")

upload = st.empty()
with upload.container():
    file = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg"))

    with st.expander("Or use a camera"):
        camera = st.camera_input("Camera input", label_visibility="hidden")

if file or camera:
    upload.empty()

    pic = file or camera
    st.image(pic, caption="Uploaded Picture", use_column_width=True)

    bytes_data = pic.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    detections = detect_face(cv2_img)

    faces = [cv2_img[e[1]:e[3], e[0]:e[2]] for e in detections[:3]]

    if len(faces) == 0:
        st.error("No faces detected")
        st.stop()
    elif len(faces) == 1:
        face = faces[0]
        st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        st.write("Click Next to continue")
    else:
        cols = st.columns(3)
        for i, face_img in enumerate(faces):
            with cols[i]:
                st.image(face_img, caption=i+1, use_column_width=True, channels="BGR")

        selected_face = st.radio("Select face", [i+1 for i in range(len(detections[:3]))], horizontal=True)
        face = faces[selected_face-1]

    next_btn = st.empty()
    with next_btn.container():
        text = st.text_input("Enter name")
        next = st.button("Next")
    if next:
        if not text:
            st.error("Please enter a name")
            st.stop()
        next_btn.empty()
        (p / text).mkdir(exist_ok=True)
        cv2.imwrite(str(p / text / f"{str(uuid4())[:8]}.jpg"), face)
        create_pkl("faces", tqdm=stqdm)
        st.success("Face saved successfully")
