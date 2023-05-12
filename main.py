import pickle
import cv2
import numpy as np
from deepface.commons import functions, realtime, distance as dst
from ultralytics.yolo.engine.results import Results
from ultralytics import YOLO
import os
from tqdm import tqdm
from deepface.DeepFace import represent
import pandas as pd
import time
import requests
from collections import deque
from datetime import datetime
import threading
from threading import Lock


def detect_yolo(model: YOLO, frame: np.ndarray):
    output: list[Results] = model.predict(frame, verbose=False)
    xyxy_np = output[0].boxes.xyxy.cpu().numpy()
    conf = output[0].boxes.conf.cpu().numpy()
    xyxy_np = xyxy_np.astype(int)
    xyxy_np = xyxy_np[conf > 0.6]
    return xyxy_np


def create_pkl(db_path):
    model_name = "VGG-Face"
    detector_backend = "opencv"
    enforce_detection = True
    align = True
    normalization = "base"
    silent = True

    file_name = f"representations.pkl"

    target_size = functions.find_target_size(model_name=model_name)
    employees = []

    for r, _, f in os.walk(db_path):
        for file in f:
            if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
            ):
                exact_path = r + "/" + file
                employees.append(exact_path)

    if len(employees) == 0:
        raise ValueError(
            "There is no image in ",
            db_path,
            " folder! Validate .jpg or .png files exist in this path.",
        )

    representations = []

    # for employee in employees:
    pbar = tqdm(
        range(0, len(employees)),
        desc="Finding representations",
        disable=silent,
    )
    for index in pbar:
        employee = employees[index]

        img_objs = functions.extract_faces(
            img=employee,
            target_size=target_size,
            detector_backend="skip",
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,

        )

        for img_content, _, _ in img_objs:
            embedding_obj = represent(
                img_path=img_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img_representation = embedding_obj[0]["embedding"]

            instance = []
            instance.append(employee)
            instance.append(img_representation)
            representations.append(instance)

    # -------------------------------

    with open(f"{db_path}/{file_name}", "wb") as f:
        pickle.dump(representations, f)

    if not silent:
        print(
            f"Representations stored in {db_path}/{file_name} file."
            + "Please delete this file when you add new identities in your database."
        )


def find(img, reprs):
    tic = time.time()

    model_name = "VGG-Face"
    detector_backend = "opencv"
    enforce_detection = False
    align = True
    normalization = "base"
    silent = True
    target_size = functions.find_target_size(model_name=model_name)
    distance_metric = "cosine"

    with open(reprs, "rb") as f:
        representations = pickle.load(f)

    df = pd.DataFrame(representations, columns=["identity", f"{model_name}_representation"])

    # img path might have more than once face
    target_objs = functions.extract_faces(
        img=img,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    faces_found = []

    for target_img, target_region, _ in target_objs:
        target_embedding_obj = represent(
            img_path=target_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances

        threshold = dst.findThreshold(model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)

        # print(result_df)

        faces_found.extend(result_df['identity'].tolist())

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return faces_found


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

create_pkl("database")
model = YOLO("./yolov8n-face.pt")
link = 'https://streamer109.tattelecom.ru/intercom_3497/mpegts?token=2.CdOZqaunAmQABftqL-ZwiffDuV0yYbGTVARII9pEvgARNp_h'
cam = Camera(link)
# cap = cv2.VideoCapture(link)

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


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
