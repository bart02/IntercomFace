import cv2
from ultralytics import YOLO

from detect import detect, find
from embed import create_pkl

create_pkl("faces")

model = YOLO("models/yolov8n-face.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('face', frame)
    faces = detect(model, frame)

    if len(faces) > 0:
        for face_0 in faces:
            face_rect = cv2.rectangle(frame, (face_0[0], face_0[1]), (face_0[2], face_0[3]), (0, 255, 0), 2)
            new_img = frame[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]]
            faces_found = find(new_img, 'faces/representations.pkl')
            if len(faces_found) > 0:
                print(faces_found)
            cv2.imshow('face', face_rect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
