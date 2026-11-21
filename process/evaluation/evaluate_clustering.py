from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score,  normalized_mutual_info_score


def evaluate_clustering(X, labels_pred, labels_true=None):
    results = {
        "Silhouette_Score": silhouette_score(X, labels_pred),
        "Davies_Bouldin": davies_bouldin_score(X, labels_pred),
        "Calinski_Harabasz": calinski_harabasz_score(X, labels_pred)
    }

    if labels_true is not None:
        results["ARI"] = adjusted_rand_score(labels_true, labels_pred)
        results["NMI"] = normalized_mutual_info_score(labels_true, labels_pred)

    return results
