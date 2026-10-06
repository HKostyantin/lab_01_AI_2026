import os
import numpy as np
from PIL import Image

ROWS, COLS = 4, 5  


def euclid(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))

def manhattan(a, b):
    return float(np.sum(np.abs(a - b)))

def chebyshev(a, b):
    return float(np.max(np.abs(a - b)))

METRICS = {"Евклідова": euclid, "Манхетенська": manhattan, "Чебишева": chebyshev}

def extract_features(img):
    black = np.array(img.convert("L")) < 128
    h, w = black.shape
    ys = np.linspace(0, h, ROWS + 1).astype(int)
    xs = np.linspace(0, w, COLS + 1).astype(int)
    abs_vec = np.array([black[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].sum()
                        for r in range(ROWS) for c in range(COLS)], dtype=float)
    total = abs_vec.sum()
    norm_vec = abs_vec / total if total else abs_vec
    return abs_vec, norm_vec


def load_training(folder):
    data = {}
    for cls in sorted(os.listdir(folder)):
        path = os.path.join(folder, cls)
        if os.path.isdir(path):
            files = [f for f in sorted(os.listdir(path)) if f.lower().endswith((".bmp", ".png", ".jpg"))]
            data[cls.replace("class_", "")] = [extract_features(Image.open(os.path.join(path, f)))[1] for f in files]
    return data


def list_samples(folder):
    return {c.replace("class_", ""): [os.path.join(folder, c, f) for f in sorted(os.listdir(os.path.join(folder, c)))
                                      if f.lower().endswith((".bmp", ".png", ".jpg"))]
            for c in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, c))}


class Classifier:
    def __init__(self):
        self.clusters = {}

    def train(self, data):
        self.clusters = {}
        for label, vecs in data.items():
            arr = np.array(vecs)
            self.clusters[label] = {"n": len(arr), "center": arr.mean(axis=0),
                                    "min": arr.min(axis=0), "max": arr.max(axis=0)}

    def etalons(self, data):
        self.clusters = {l: {"n": 1, "center": np.array(v[0]), "min": np.array(v[0]), "max": np.array(v[0])}
                         for l, v in data.items() if v}

    def predict(self, vec, method):
        if method == "Ортогональні області": 
            scores = {l: int(np.sum((vec < c["min"] - 1e-3) | (vec > c["max"] + 1e-3)))
                      for l, c in self.clusters.items()}
        else:
            scores = {l: round(METRICS[method](vec, c["center"]), 4) for l, c in self.clusters.items()}
        return min(scores, key=scores.get), scores

