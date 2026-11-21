import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans


def compute_trustworthiness(X_o, X_e, k=5):
    n_samples = X_o.shape[0]
    nbrs_o = NearestNeighbors(n_neighbors=k + 1).fit(X_o)
    indices_o = nbrs_o.kneighbors(X_o, return_distance=False)[:, 1:]

    nbrs_e = NearestNeighbors(n_neighbors=k + 1).fit(X_e)
    indices_e = nbrs_e.kneighbors(X_e, return_distance=False)[:, 1:]

    trustworthiness = 0.0
    for i in range(n_samples):
        u_set = set(indices_o[i])
        v_set = set(indices_e[i])
        intersect = u_set.intersection(v_set)
        trustworthiness += len(intersect) / k
    return trustworthiness / n_samples


def evaluate_dimensionality_reduction(X_original, X_embedded, y=None, n_neighbors=5):
    results = {}
    X_orig = np.asarray(X_original)
    X_emb = np.asarray(X_embedded)

    try:
        results["trustworthiness"] = compute_trustworthiness(X_orig, X_emb, k=n_neighbors)
    except Exception:
        results["trustworthiness"] = None

    if y is not None:
        y_arr = np.asarray(y)
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(clf, X_emb, y_arr, cv=3, scoring='accuracy')
            results["classification_accuracy_mean"] = float(np.mean(scores))
            results["classification_accuracy_std"] = float(np.std(scores))
        except Exception:
            pass

        try:
            if len(np.unique(y_arr)) > 1 and len(y_arr) > n_neighbors:
                results["silhouette_score_labels"] = float(silhouette_score(X_emb, y_arr))
        except Exception:
            pass
    else:
        try:
            kmeans = KMeans(n_clusters=min(3, len(X_emb)), random_state=42, n_init='auto').fit(X_emb)
            results["unsupervised_silhouette_score"] = float(silhouette_score(X_emb, kmeans.labels_))
        except Exception:
            pass

    results["original_dim"] = X_orig.shape[1]
    results["embedded_dim"] = X_emb.shape[1]
    results["compression_ratio"] = float(X_emb.shape[1] / X_orig.shape[1])

    return results