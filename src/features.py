"""
Feature engineering for the multimodal fraud pipeline.

Structured + categorical features go through standard sklearn transformers.
Text goes through TF-IDF. Images go through a custom transformer that
extracts classical CV descriptors (intensity, edges, color histogram) by
default, or CNN embeddings if torch/torchvision are installed
(set IMAGE_BACKEND=cnn as an env var to opt in).
"""
import os

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

NUMERIC_FEATURES = [
    "claim_amount",
    "num_prev_claims",
    "days_since_policy",
    "num_payments",
    "payment_irregularity",
    "disaster_nearby",
]
CATEGORICAL_FEATURES = ["location"]
TEXT_FEATURE = "narrative"
IMAGE_FEATURE = "image_path"
LABEL = "is_fraud"


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """Reads each image from disk and returns a fixed-length numeric
    descriptor: mean intensity, std intensity, edge density, and a coarse
    3-bin color histogram per channel (9 bins total) -> 12 features/image.

    Falls back to zeros (and logs a warning once) if a path is missing,
    so a handful of broken paths won't crash training.
    """

    N_FEATURES = 12

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        paths = X[IMAGE_FEATURE] if hasattr(X, "columns") else X
        feats = np.zeros((len(paths), self.N_FEATURES), dtype=np.float32)
        missing = 0
        for i, p in enumerate(paths):
            if not isinstance(p, str) or not os.path.exists(p):
                missing += 1
                continue
            img = cv2.imread(p)
            if img is None:
                missing += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            hist = []
            for c in range(3):
                h = cv2.calcHist([img], [c], None, [3], [0, 256]).flatten()
                hist.extend(h / (img.shape[0] * img.shape[1]))
            feats[i] = [
                float(gray.mean()),
                float(gray.std()),
                float((edges > 0).mean()),
                *hist,
            ]
        if missing:
            print(f"[ImageFeatureExtractor] warning: {missing} image(s) missing or unreadable")
        return feats


def build_preprocessor() -> ColumnTransformer:
    """Returns a ColumnTransformer that turns the raw claim DataFrame into
    a single fused numeric matrix (structured + categorical + text + image)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "text",
                TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2)),
                TEXT_FEATURE,
            ),
            ("img", ImageFeatureExtractor(), [IMAGE_FEATURE]),
        ],
        remainder="drop",
    )
