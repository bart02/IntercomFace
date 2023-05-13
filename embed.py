import numpy as np
from sklearn.cluster import DBSCAN
import pickle
from tqdm import tqdm as otqdm
from deepface.DeepFace import represent
from pathlib import Path


def create_pkl(db_path,
               model_name="VGG-Face",
               normalization="base",
               file_name="representations.pkl",
               use_cluster=True,
               tqdm=otqdm):
    p = Path(db_path)
    photo_paths = [str(f) for f in p.glob("**/*.jpg")] + \
                  [str(f) for f in p.glob("**/*.jpeg")] + \
                  [str(f) for f in p.glob("**/*.png")]

    embeddings = [
        represent(
            img_path=e,
            model_name=model_name,
            detector_backend="skip",
            normalization=normalization,
        )[0]["embedding"]
        for e in tqdm(photo_paths)
    ]

    if use_cluster:
        labels = DBSCAN(eps=0.2, min_samples=1, metric="cosine").fit_predict(embeddings)
        np_paths = np.array(photo_paths)
        nemb = np.array(embeddings)
        representations = [(','.join(np_paths[labels == i]), np.mean(nemb[labels == i], axis=0).tolist())
                           for i in np.unique(labels)]
    else:
        representations = list(zip(photo_paths, embeddings))

    with open(f"{db_path}/{file_name}", "wb") as f:
        pickle.dump(representations, f)


if __name__ == '__main__':
    create_pkl("faces")
