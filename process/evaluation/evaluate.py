from process.evaluation.evaluate_classification import evaluate_classification
from process.evaluation.evaluate_clustering import evaluate_clustering
from process.evaluation.evaluate_dim_reduction import evaluate_dimensionality_reduction
from process.evaluation.evaluate_regression import evaluate_regression


def evaluate_model(type, X, labels_pred, y_test, y_pred, X_original, X_embedded, y=None, n_neighbors=5):
    eval_type = type.value if hasattr(type, "value") else str(type)
    normalized = eval_type.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized == "classification":
        return evaluate_classification(y_test, y_pred)
    elif normalized == "regression":
        return evaluate_regression(y_test, y_pred)
    elif normalized == "clustering":
        return evaluate_clustering(X, labels_pred)
    elif normalized in {"dimensionality_reduction", "dimension_reduction", "dim_reduction"}:
        return evaluate_dimensionality_reduction(
            X_original,
            X_embedded,
            y=y,
            n_neighbors=n_neighbors,
        )

    return {
        "error": f"Unsupported evaluation type: {eval_type}",
    }