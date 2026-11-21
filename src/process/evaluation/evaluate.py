from process.evaluation.evaluate_classification import evaluate_classification
from process.evaluation.evaluate_clustering import evaluate_clustering
from process.evaluation.evaluate_dim_reduction import evaluate_dimensionality_reduction
from process.evaluation.evaluate_regression import evaluate_regression


def evaluate_model(type, X, labels_pred, y_test, y_pred, X_original, X_embedded, y=None, n_neighbors=5):
    if type =="classification":
        clf_evaluations = evaluate_classification(y_test, y_pred)
        return clf_evaluations
    elif type =="regression":
        reg_evaluations = evaluate_regression(y_test, y_pred)
        return reg_evaluations
    elif type =="clustering":
        cls_evaluations = evaluate_clustering(X, labels_pred)
        return cls_evaluations
    elif type == "dimension reduction":
        dim_evaluations = evaluate_dimensionality_reduction(X_original, X_embedded, y=y, n_neighbors=n_neighbors)
        return dim_evaluations