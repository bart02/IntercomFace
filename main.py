import cv2
from ultralytics import YOLO
from detect import detect, find
from embed import create_pkl
import pickle
import numpy as np
from deepface.commons import functions, realtime, distance as dst
from ultralytics.yolo.engine.results import Results
import os
from tqdm import tqdm
from deepface.DeepFace import represent
import pandas as pd
import time
import requests
from collections import deque
from datetime import datetime
import threading

create_pkl("faces")
model = YOLO("models/yolov8n-face.pt")

def open_door():
    url = "https://domofon.tattelecom.ru//v1/subscriber/open-intercom"
    headers = {
        "Host": "domofon.tattelecom.ru",
        "Cookie": "_csrf-backend=626dedc14d1f915b031402db06b12ae0898317098a95eb569f3ac390a4841baea%3A2%3A%7Bi%3A0%3Bs"
                  "%3A13%3A%22_csrf-backend%22%3Bi%3A1%3Bs%3A32%3A%22D7n8GYUEq-lX0lnENvvuas6uK8tKzvAt%22%3B%7D",
        "Accept": "*/*",
        "Content-Type": "application/json",
        "User-Agent": "TTC%20Intercom/4 CFNetwork/1399 Darwin/22.1.0",
        "access-token": "8iMKsIUur1lTeZlZ0qow_rMJ3oRkx1YO",
        "Accept-Language": "ru",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    }
    data = {
        "phone": "79069595030",
        "device_code": "14A5F9E0-753F-4139-B418-528609905E53",
        "intercom_id": "3497"
    }

    resp = requests.post(url, headers=headers, json=data)
    print(resp.text)


class Camera:
    def __init__(self, rtsp_link):
        self.ret, self.frame = False, None
        self.capture = cv2.VideoCapture(rtsp_link)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self):
        while True:
            self.ret, self.frame = self.capture.read()



link = 'https://streamer109.tattelecom.ru/intercom_3497/mpegts?token=2.CdOZqaunAmQABftqL-ZwiffDuV0yYbGTVARII9pEvgARNp_h'
cam = Camera(link)

duration = 30
there_are_faces_deq = deque(maxlen=duration)
print("start")
while True:
    frame = cam.frame
    if frame is None:
        continue
    height, width, channels = frame.shape
    if height == 0 and width == 0:
        continue

    cv2.imshow('frame', frame)

    faces = detect_yolo(model, frame)
    if len(faces) > 0:
        there_are_faces_deq.append(1)
    else:
        there_are_faces_deq.append(0)

    values = list(there_are_faces_deq)
    ones = values.count(1)

    if ones == duration:
        for face_0 in faces:
            # face_rect = cv2.rectangle(frame, (face_0[0], face_0[1]), (face_0[2], face_0[3]), (0, 255, 0), 2)
            new_img = frame[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]]

            faces_found = find(new_img, 'database/representations.pkl')

            if len(faces_found):
                response = requests.get('https://webhook.site/16c53608-1a3c-447d-b577-f716bba3810a/')
                # open_door()

            # cv2.imshow('face', face_rect)
    else:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time, ones)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
