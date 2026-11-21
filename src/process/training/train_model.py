
import numpy as np
import pandas as pd
from process.model_selection.select_model import select_model


def _extract_estimator(model):
    # select_model (model, model_desc) tuple/list döndürdüyse ilk elemanı al
    if isinstance(model, (list, tuple)) and len(model) > 0:
        candidate = model[0]
        if hasattr(candidate, "fit"):
            return candidate
        if isinstance(candidate, dict) and "model_name" in candidate:
            est, _ = select_model(**candidate)
            return est
        if isinstance(candidate, str):
            est, _ = select_model(model_type="regression", model_name=candidate)
            return est

    if isinstance(model, dict):
        if "model_name" in model:
            est, _ = select_model(**model)
            return est

    if isinstance(model, str):
        # Gerekirse model_type bilgisini duruma göre ayarla
        est, _ = select_model(model_type="regression", model_name=model)
        return est

    return model


def _prepare_fit_inputs(X_train, y_train):
    X_arr = pd.DataFrame(X_train) if isinstance(X_train, list) and X_train and isinstance(X_train[0], dict) else (
        np.asarray(X_train) if isinstance(X_train, list) else X_train
    )
    y_arr = np.asarray(y_train) if isinstance(y_train, list) else y_train
    return X_arr, y_arr


def train_model(model, X_train, y_train):
    estimator = _extract_estimator(model)
    if not hasattr(estimator, "fit"):
        raise TypeError(
            f"Geçerli bir model nesnesi bulunamadı; gelen tip: {type(model).__name__}. "
            "Önce /select-data/ çıktısını veya eğitilebilir bir estimator geçin."
        )

    X_fit, y_fit = _prepare_fit_inputs(X_train, y_train)
    return estimator.fit(X_fit, y_fit)


