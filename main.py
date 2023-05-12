import pickle
import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
from ultralytics.yolo.engine.results import Results
from ultralytics import YOLO
import os
from tqdm import tqdm
from deepface.DeepFace import represent
import pandas as pd
import time
from IPython.display import display


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
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= 1]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)

        print(result_df)

        faces_found.extend(result_df['identity'].tolist())

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return faces_found


create_pkl("database")

model = YOLO("./yolov8n-face.pt")
cap = cv2.VideoCapture(0)

# frame = cv2.imread('me2.jpg')
# faces = detect_yolo(model, frame)
#
# for i, face_0 in enumerate(faces):
#     cv2.rectangle(frame, (face_0[0], face_0[1]), (face_0[2], face_0[3]), (0, 255, 0), 2)
#     cv2.imwrite(f'rect_me_{i}.jpg', frame)

# cv2.imshow('frame', frame)
# result = find('me.jpg', 'database/representations.pkl')
# print(result)
#

while True:
    ret, frame = cap.read()
    cv2.imshow('face', frame)
    faces = detect_yolo(model, frame)

    if len(faces) > 0:
        for face_0 in faces:
            face_rect = cv2.rectangle(frame, (face_0[0], face_0[1]), (face_0[2], face_0[3]), (0, 255, 0), 2)
            new_img = frame[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]]
            faces_found = find(new_img, 'database/representations.pkl')
            if len(faces_found) > 0:
                print(faces_found)
            cv2.imshow('face', face_rect)
    #
    #     # faces = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
    #     # print(faces)
    #
    #     # if 0.8 < faces[0]['confidence'] < 4.0:
    #     #     # print(faces[0]['confidence'])
    #     #     cv2.imshow('face', faces[0]['face'])
    #     #     print(DeepFace.find(faces[0]['face'], 'faces', detector_backend='opencv', enforce_detection=False))
    #
    #     # DeepFace.find()
    #
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
