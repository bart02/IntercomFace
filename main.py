import cv2
import numpy as np
from deepface import DeepFace
from ultralytics.yolo.engine.results import Results
from ultralytics import YOLO

model = YOLO("./yolov8n-face.pt")
cap = cv2.VideoCapture("/Users/artem/Documents/qq.mp4")


def detect_yolo(model: YOLO, frame: np.ndarray):
    output: list[Results] = model.predict(frame, verbose=False)
    xyxy_np = output[0].boxes.xyxy.cpu().numpy()
    conf = output[0].boxes.conf.cpu().numpy()
    xyxy_np = xyxy_np.astype(int)
    xyxy_np = xyxy_np[conf > 0.65]
    return xyxy_np


while True:
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)

    faces = detect_yolo(model, frame)
    # print(faces)

    for face_0 in faces:
        cv2.rectangle(frame, (face_0[0], face_0[1]), (face_0[2], face_0[3]), (0, 255, 0), 2)

    cv2.imshow('face', frame[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]])

    # faces = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
    # print(faces)


    # if 0.8 < faces[0]['confidence'] < 4.0:
    #     # print(faces[0]['confidence'])
    #     cv2.imshow('face', faces[0]['face'])
    #     print(DeepFace.find(faces[0]['face'], 'faces', detector_backend='opencv', enforce_detection=False))


    # DeepFace.find()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

