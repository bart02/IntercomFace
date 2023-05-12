import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from detect import detect_yolo

model = YOLO("models/yolov8n-face.pt")

st.set_page_config(
    page_title="Add Face",
    page_icon="👨",
)

st.title("👨 Add Face")

file_upl = st.empty()
file = file_upl.file_uploader("Upload a file", type=("png", "jpg"))

camera_upl = st.empty()
with camera_upl.expander("Or use a camera"):
    camera = st.camera_input("Camera input", label_visibility="hidden")

if file or camera:
    file_upl.empty()
    camera_upl.empty()

    pic = file or camera

    st.image(pic, caption="Uploaded Picture", use_column_width=True)

    bytes_data = pic.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    cv2.imwrite("/Users/artem/Projects/IntercomFace/tmp.jpg", cv2_img)

    detections = detect_yolo(model, cv2_img, 0)
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
                st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption=i+1, use_column_width=True)

        selected_face = st.radio("Select face", [i+1 for i in range(len(detections[:3]))], horizontal=True)
        face = faces[selected_face-1]

    next_btn = st.empty()
    next = next_btn.button("Next")
    if next:
        next_btn.empty()
        cv2.imwrite(f"/Users/artem/Projects/IntercomFace/faces/add.jpg", face)
        st.success("Face saved successfully")
