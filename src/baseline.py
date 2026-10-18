"""TF-IDF linear binary classifier for sentence–gloss compatibility."""
from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from .preprocessing import pair_text

def build_baseline(max_features=100000, C=2.0, max_iter=2000):
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=max_features, sublinear_tf=True, strip_accents=None)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=max_features, sublinear_tf=True, strip_accents=None)),
    ])
    return Pipeline([("features", features), ("classifier", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=42))])

def train_baseline(train, max_features=100000, C=2.0, max_iter=2000):
    model=build_baseline(max_features,C,max_iter)
    model.fit([pair_text(r) for r in train.to_dict("records")], train.label.astype(int))
    return model

def save_baseline(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)

def load_baseline(path): return joblib.load(path)

