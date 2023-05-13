import pickle

import numpy as np
import pandas as pd
from deepface.DeepFace import represent
from deepface.commons import distance as dst
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

from distance import find_distance


def detect(model: YOLO, frame: np.ndarray, thresh=0.6):
    output: list[Results] = model.predict(frame, verbose=False)
    xyxy_np = output[0].boxes.xyxy.cpu().numpy()
    conf = output[0].boxes.conf.cpu().numpy()
    xyxy_np = xyxy_np.astype(int)
    xyxy_np = xyxy_np[conf > thresh]
    return xyxy_np


def find(img, reprs, model_name="VGG-Face", normalization="base", distance_metric="cosine", threshold=None):
    if threshold is None:
        threshold = dst.findThreshold(model_name, distance_metric)

    with open(reprs, "rb") as f:
        representations = pickle.load(f)
    df = pd.DataFrame(representations, columns=["identity", f"{model_name}_representation"])

    target_representation = represent(
        img_path=img,
        model_name=model_name,
        detector_backend="skip",
        normalization=normalization,
    )[0]["embedding"]

    distances = [
        find_distance(source[f"{model_name}_representation"], target_representation, distance_metric)
        for _, source in df.iterrows()
    ]

    metric_column = f"{model_name}_{distance_metric}"

    result_df = df.copy().drop(columns=[f"{model_name}_representation"])
    result_df[metric_column] = distances
    result_df = result_df[result_df[metric_column] <= threshold]
    result_df = result_df.sort_values(by=[metric_column], ascending=True).reset_index(drop=True)

    return result_df['identity'].tolist()
