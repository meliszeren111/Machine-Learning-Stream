
import numpy as np
import pandas as pd


def _resolve_model(model, model_store: dict | None = None):
    if isinstance(model, dict):
        if "model_id" in model and model_store and model["model_id"] in model_store:
            return model_store[model["model_id"]]
        if "model" in model and hasattr(model["model"], "predict"):
            return model["model"]
    if isinstance(model, str) and model_store and model in model_store:
        return model_store[model]
    return model


def _prepare_X_test(X_test):
    if X_test is None:
        return None
    if isinstance(X_test, list):
        if len(X_test) > 0 and isinstance(X_test[0], dict):
            return pd.DataFrame(X_test)
        return np.asarray(X_test)
    return X_test


def predict_data(model, X_test, model_store: dict | None = None):
    estimator = _resolve_model(model, model_store=model_store)
    if not hasattr(estimator, "predict"):
        raise TypeError(
            f"Tahmin yapabilecek geçerli bir model bulunamadı; gelen tip: {type(model).__name__}. "
            "Modelin eğitildiğinden ve MODEL_STORE üzerinden çözülebildiğinden emin olun."
        )
    X_prepared = _prepare_X_test(X_test)
    return estimator.predict(X_prepared)