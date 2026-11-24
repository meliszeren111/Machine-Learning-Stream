import gc
import inspect
import io
import os
import pickle
import sys
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.model_catalogue import MODEL_CATALOG
from process.data_loading.load_csv import load_csv_file
from process.data_loading.load_tsv import load_tsv_file
from process.data_loading.load_xlsx import load_xlsx_file
from process.data_splitting.split_data2 import split_data as run_split_data
from process.data_preprocessing.preprocess_data import preprocess_data as run_preprocess_data
from process.get_data.get_data2 import get_data_from_file_binary
from process.evaluation.evaluate import evaluate_model
from process.model_selection.select_model2 import select_model as run_select_model
from process.prediction.predict_data import predict_data as run_predict_data
from process.training.train_model import train_model as run_train_model

st.set_page_config(
    page_title="Machine Learning Stream (Local)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 2.8em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE = {
    "temp_id": None,
    "preprocessed_temp_id": None,
    "split_id": None,
    "model_id": None,
    "predictions_id": None,
    "columns": [],
    "upload_info": None,
    "preprocess_info": None,
    "split_info": None,
    "select_info": None,
    "train_info": None,
    "predict_info": None,
    "evaluate_info": None,
    "one_click_info": None,
    "artifacts": {},
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = {} if k == "artifacts" else v


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _ensure_df(data: Any) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    if hasattr(data, "toarray"):
        return pd.DataFrame(data.toarray())
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "df" in data and data["df"] is not None:
            return _ensure_df(data["df"])
        return pd.DataFrame([data])
    return pd.DataFrame(data)


def read_uploaded_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    filename_lower = filename.lower()
    try:
        df = get_data_from_file_binary(raw_bytes, filename_lower)
        df = _ensure_df(df)
        if df is not None:
            return df
    except Exception:
        pass

    bio = io.BytesIO(raw_bytes)
    if filename_lower.endswith(".csv"):
        return _ensure_df(load_csv_file(bio))
    if filename_lower.endswith(".tsv"):
        return _ensure_df(load_tsv_file(bio))
    if filename_lower.endswith((".xlsx", ".xls")):
        return _ensure_df(load_xlsx_file(bio))
    if filename_lower.endswith(".txt"):
        return pd.read_csv(bio, sep=None, engine="python")
    raise ValueError(f"Unsupported file format: {filename}")


class DataTypeEnum(str, Enum):
    ordinal = "ordinal"
    nonordinal = "nonordinal"
    image = "image"


class ModelTypeEnum(str, Enum):
    regression = "regression"
    classification = "classification"
    clustering = "clustering"
    dimensionality_reduction = "dimensionality_reduction"


_MODEL_VALID_LOSSES = {
    "HistGradientBoostingClassifier": {"log_loss"},
    "HistGradientBoostingRegressor": {"squared_error", "absolute_error", "huber", "quantile"},
    "GradientBoostingClassifier": {"log_loss"},
    "GradientBoostingRegressor": {"squared_error", "absolute_error", "huber", "quantile"},
    "SGDClassifier": {"hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"},
    "SGDRegressor": {"squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"},
}

_MODEL_DEFAULT_LOSS = {
    "HistGradientBoostingClassifier": "log_loss",
    "HistGradientBoostingRegressor": "squared_error",
    "GradientBoostingClassifier": "log_loss",
    "GradientBoostingRegressor": "squared_error",
    "SGDClassifier": "hinge",
    "SGDRegressor": "squared_error",
}

_MODEL_VALID_CRITERIA = {
    "DecisionTreeClassifier": {"gini", "entropy", "log_loss"},
    "RandomForestClassifier": {"gini", "entropy", "log_loss"},
    "ExtraTreesClassifier": {"gini", "entropy", "log_loss"},
    "DecisionTreeRegressor": {"squared_error", "friedman_mse", "absolute_error", "poisson"},
    "RandomForestRegressor": {"squared_error", "friedman_mse", "absolute_error", "poisson"},
    "ExtraTreesRegressor": {"squared_error", "friedman_mse", "absolute_error", "poisson"},
}

_MODEL_DEFAULT_CRITERION = {
    "DecisionTreeClassifier": "gini",
    "RandomForestClassifier": "gini",
    "ExtraTreesClassifier": "gini",
    "DecisionTreeRegressor": "squared_error",
    "RandomForestRegressor": "squared_error",
    "ExtraTreesRegressor": "squared_error",
}


def sanitize_model_params(model_type_val: str, model_name_val: str, params: dict) -> dict:
    cleaned = dict(params)

    # 1. Loss uyumluluğu
    raw_loss = cleaned.get("loss")
    if model_name_val in _MODEL_VALID_LOSSES:
        if raw_loss not in _MODEL_VALID_LOSSES[model_name_val]:
            cleaned["loss"] = _MODEL_DEFAULT_LOSS[model_name_val]
    elif model_type_val == "classification" and raw_loss == "squared_error":
        cleaned["loss"] = "log_loss"
    elif model_type_val == "regression" and raw_loss in {
        "hinge",
        "log_loss",
        "modified_huber",
        "squared_hinge",
        "perceptron",
    }:
        cleaned["loss"] = "squared_error"

    # 2. Criterion uyumluluğu
    raw_criterion = cleaned.get("criterion")
    if model_name_val in _MODEL_VALID_CRITERIA:
        if raw_criterion not in _MODEL_VALID_CRITERIA[model_name_val]:
            cleaned["criterion"] = _MODEL_DEFAULT_CRITERION[model_name_val]
    elif model_type_val == "classification" and raw_criterion in {
        "squared_error",
        "friedman_mse",
        "absolute_error",
        "poisson",
    }:
        cleaned["criterion"] = "gini"
    elif model_type_val == "regression" and raw_criterion in {"gini", "entropy", "log_loss"}:
        cleaned["criterion"] = "squared_error"

    # 3. Kernel uyumluluğu
    if cleaned.get("kernel") == "precomputed":
        cleaned["kernel"] = "rbf"

    # 4. t-SNE init uyumluluğu
    if model_name_val == "TSNE" and cleaned.get("init") not in {"pca", "random"}:
        cleaned["init"] = "pca"

    # 5. run_select_model imzasıyla eşleşmeyen ekstra anahtarları ayıkla
    try:
        sig = inspect.signature(run_select_model)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_kwargs:
            valid_keys = set(sig.parameters.keys())
            cleaned = {k: v for k, v in cleaned.items() if k in valid_keys}
    except Exception:
        pass

    return cleaned


def _call_select_model(model_type_val: str, model_name_val: str, params: dict):
    safe_params = sanitize_model_params(model_type_val, model_name_val, params)
    return run_select_model(
        model_type=model_type_val,
        model_name=model_name_val,
        alpha=float(safe_params.get("alpha", 1.0)),
        C=float(safe_params.get("C", 1.0)),
        l1_ratio=float(safe_params.get("l1_ratio", 0.5)),
        dual=bool(safe_params.get("dual", True)),
        fit_intercept=bool(safe_params.get("fit_intercept", True)),
        max_iter=int(safe_params.get("max_iter", 100)),
        selection=safe_params.get("selection", "random"),
        degree=int(safe_params.get("degree", 3)),
        kernel=safe_params.get("kernel", "rbf"),
        max_depth=int(safe_params.get("max_depth", 5)),
        neighbors=int(safe_params.get("neighbors", 5)),
        n_estimators=int(safe_params.get("n_estimators", 100)),
        weights=safe_params.get("weights", "uniform"),
        algorithm=safe_params.get("algorithm", "auto"),
        criterion=safe_params.get("criterion", "squared_error"),
        n_neighbors=int(safe_params.get("n_neighbors", 10)),
        max_features=safe_params.get("max_features", "sqrt"),
        priors=safe_params.get("priors", None),
        loss=safe_params.get("loss", "squared_error"),
        learning_rate=float(safe_params.get("learning_rate", 0.05)),
        n_clusters=int(safe_params.get("n_clusters", 3)),
        eps=float(safe_params.get("eps", 0.5)),
        min_samples=int(safe_params.get("min_samples", 5)),
        linkage=safe_params.get("linkage", "ward"),
        bandwidth=safe_params.get("bandwidth", None),
        damping=float(safe_params.get("damping", 0.5)),
        n_components=int(safe_params.get("n_components", 2)),
        gamma=safe_params.get("gamma", "scale"),
        perplexity=float(safe_params.get("perplexity", 30.0)),
        init=safe_params.get("init", "pca"),
    )


class LocalPipelineEngine:
    """FastAPI endpoint'lerinin yerini alan yerel çalıştırma katmanı."""

    def __init__(self, artifacts: dict):
        self.artifacts = artifacts
        self.temp_dir = os.path.join(tempfile.gettempdir(), "ml_pipeline_tmp_local")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.model_store: Dict[str, Any] = {}
        self.latest_state: Dict[str, Optional[str]] = {
            "raw_temp_id": None,
            "preprocessed_temp_id": None,
            "active_temp_id": None,
            "split_id": None,
            "model_id": None,
            "predictions_id": None,
        }

    def _save_temp(self, payload: Any, existing_id: Optional[str] = None) -> str:
        temp_id = existing_id or new_id("tmp")
        self.artifacts[temp_id] = payload
        file_path = os.path.join(self.temp_dir, f"{temp_id}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return temp_id

    def _load_temp(self, temp_id: Optional[str]) -> Any:
        if not temp_id:
            return None
        if temp_id in self.artifacts:
            return self.artifacts[temp_id]
        file_path = os.path.join(self.temp_dir, f"{temp_id}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
                self.artifacts[temp_id] = obj
                return obj
        raise FileNotFoundError(f"Temporary data '{temp_id}' not found.")

    def _resolve_available_temp_id(self, preferred_order: list[str]) -> Optional[str]:
        for key in preferred_order:
            candidate_id = self.latest_state.get(key)
            if candidate_id and (
                candidate_id in self.artifacts
                or os.path.exists(os.path.join(self.temp_dir, f"{candidate_id}.pkl"))
            ):
                return candidate_id
        return None

    def upload_data(self, filename: str, raw_bytes: bytes) -> dict:
        df = read_uploaded_dataframe(filename, raw_bytes)
        df = _ensure_df(df)
        temp_id = self._save_temp(df)

        self.latest_state["raw_temp_id"] = temp_id
        self.latest_state["preprocessed_temp_id"] = None
        self.latest_state["active_temp_id"] = temp_id
        self.latest_state["split_id"] = None
        self.latest_state["predictions_id"] = None

        return {
            "temp_id": temp_id,
            "filename": filename,
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "preview": df.head(3).to_dict(orient="records"),
        }

    def preprocess_data(self, params: dict) -> dict:
        resolved_temp_id = (
            params.get("temp_id")
            or st.session_state.temp_id
            or self._resolve_available_temp_id(["raw_temp_id", "active_temp_id"])
        )
        source_data = self._load_temp(resolved_temp_id) if resolved_temp_id else params.get("df")
        payload_df = (
            source_data.get("df")
            if isinstance(source_data, dict) and "df" in source_data and len(source_data) == 1
            else source_data
        )
        parsed_df = _ensure_df(payload_df)
        if parsed_df is None:
            raise ValueError("No available dataset found for preprocessing.")

        data_type = params.get("data_type", "ordinal")
        dt_val = data_type.value if hasattr(data_type, "value") else data_type

        result = run_preprocess_data(
            parsed_df,
            scale=bool(params.get("scale", True)),
            normalize=bool(params.get("normalize", False)),
            encode=bool(params.get("encode", True)),
            data_type=dt_val,
        )
        processed_df = _ensure_df(result)
        saved_id = self._save_temp(processed_df)

        self.latest_state["preprocessed_temp_id"] = saved_id
        self.latest_state["active_temp_id"] = saved_id
        self.latest_state["split_id"] = None
        self.latest_state["predictions_id"] = None

        return {
            "temp_id": saved_id,
            "source_temp_id": resolved_temp_id,
            "rows": int(processed_df.shape[0]),
            "columns": list(processed_df.columns),
            "preview": processed_df.head(3).to_dict(orient="records"),
        }

    def split_data(self, params: dict) -> dict:
        resolved_temp_id = (
            params.get("temp_id")
            or st.session_state.preprocessed_temp_id
            or st.session_state.temp_id
            or self._resolve_available_temp_id([
                "preprocessed_temp_id",
                "active_temp_id",
                "raw_temp_id",
            ])
        )
        source_data = self._load_temp(resolved_temp_id) if resolved_temp_id else params.get("df")
        payload_df = (
            source_data.get("df")
            if isinstance(source_data, dict) and "df" in source_data and len(source_data) == 1
            else source_data
        )
        parsed_df = _ensure_df(payload_df)

        data_type = params.get("data_type", "ordinal")
        dt_val = data_type.value if hasattr(data_type, "value") else data_type

        X_train, X_test, y_train, y_test = run_split_data(
            df=parsed_df,
            huggingface_dataset=params.get("huggingface_dataset"),
            data_type=dt_val,
            ylabel=params.get("ylabel"),
            test_size=float(params.get("test_size", 0.2)),
            train_size=float(params.get("train_size", 0.8)),
            split_by=params.get("split_by", "train"),
            left_size=float(params.get("left_size", 0.8)),
            right_size=float(params.get("right_size", 0.2)),
            shuffle=bool(params.get("shuffle", True)),
            random_state=int(params.get("random_state", 42)),
        )

        split_id = self._save_temp({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        })
        self.latest_state["split_id"] = split_id
        self.latest_state["predictions_id"] = None

        return {
            "split_id": split_id,
            "source_temp_id": resolved_temp_id,
            "X_train_shape": getattr(X_train, "shape", None),
            "X_test_shape": getattr(X_test, "shape", None),
        }

    def select_model(self, params: dict) -> dict:
        model_type = params.get("model_type", "regression")
        model_name = params.get("model_name", "RandomForestRegressor")
        model_type_val = model_type.value if hasattr(model_type, "value") else model_type
        model_name_val = model_name.value if hasattr(model_name, "value") else model_name

        estimator, model_desc = _call_select_model(model_type_val, model_name_val, params)

        model_id = new_id("model")
        self.model_store[model_id] = estimator
        self.artifacts[model_id] = estimator
        self.latest_state["model_id"] = model_id

        return {
            "model_id": model_id,
            "model_name": model_name_val,
            "model_type": model_type_val,
            "model_info": model_desc,
        }

    def train_model(self, params: dict) -> dict:
        resolved_split_id = (
            params.get("split_id")
            or st.session_state.split_id
            or self.latest_state.get("split_id")
        )
        if not resolved_split_id:
            raise ValueError("No split dataset found. Pass split_id or run Split Dataset first.")

        split_bundle = self._load_temp(resolved_split_id)
        if not isinstance(split_bundle, dict):
            raise ValueError(f"Invalid split data in '{resolved_split_id}'.")

        X_train = split_bundle.get("X_train")
        y_train = split_bundle.get("y_train")

        resolved_model_id = (
            params.get("model_id")
            or st.session_state.model_id
            or self.latest_state.get("model_id")
        )
        target_model = self.model_store.get(resolved_model_id) or self.artifacts.get(resolved_model_id)
        if not resolved_model_id or target_model is None:
            raise ValueError("No model found. Call Create / Select Model first or pass a valid model_id.")

        trained = run_train_model(target_model, X_train, y_train)
        self.model_store[resolved_model_id] = trained
        self.artifacts[resolved_model_id] = trained
        self.latest_state["model_id"] = resolved_model_id
        self.latest_state["split_id"] = resolved_split_id

        return {
            "model_id": resolved_model_id,
            "split_id": resolved_split_id,
            "status": "trained",
        }

    def predict_data(self, params: dict) -> dict:
        resolved_split_id = (
            params.get("split_id")
            or st.session_state.split_id
            or self.latest_state.get("split_id")
        )
        if not resolved_split_id:
            raise ValueError("No split dataset found. Pass split_id or run Split Dataset first.")

        split_bundle = self._load_temp(resolved_split_id)
        if not isinstance(split_bundle, dict):
            raise ValueError(f"Invalid split data in '{resolved_split_id}'.")

        X_test = split_bundle.get("X_test")
        if X_test is None:
            raise ValueError(f"X_test not found in split '{resolved_split_id}'.")

        resolved_model_id = (
            params.get("model_id")
            or st.session_state.model_id
            or self.latest_state.get("model_id")
        )
        target_model = self.model_store.get(resolved_model_id) or self.artifacts.get(resolved_model_id)
        if not resolved_model_id or target_model is None:
            raise ValueError("No trained model found for prediction. Run Train Model first or pass a valid model_id.")

        try:
            predictions = run_predict_data(target_model, X_test)
        except Exception:
            if hasattr(target_model, "transform"):
                predictions = target_model.transform(X_test)
            else:
                raise

        pred_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions
        predictions_id = self._save_temp(pred_list)

        self.latest_state["predictions_id"] = predictions_id
        self.latest_state["model_id"] = resolved_model_id
        self.latest_state["split_id"] = resolved_split_id

        return {
            "predictions_id": predictions_id,
            "model_id": resolved_model_id,
            "split_id": resolved_split_id,
            "predictions": pred_list,
        }

    def evaluate_model(self, params: dict) -> dict:
        resolved_split_id = (
            params.get("split_id")
            or params.get("split_temp_id")
            or st.session_state.split_id
            or self.latest_state.get("split_id")
        )
        split_bundle = self._load_temp(resolved_split_id) if resolved_split_id else {}
        if not isinstance(split_bundle, dict):
            split_bundle = {}

        X = split_bundle.get("X_test")
        y_test = split_bundle.get("y_test")
        y = split_bundle.get("y_test")
        labels_pred = None
        X_original = None
        X_embedded = None

        resolved_predictions_id = (
            params.get("predictions_id")
            or st.session_state.predictions_id
            or self.latest_state.get("predictions_id")
        )
        y_pred = self._load_temp(resolved_predictions_id) if resolved_predictions_id else None

        eval_type = params.get("type", "regression")
        eval_type_val = eval_type.value if hasattr(eval_type, "value") else eval_type

        if y_pred is not None and eval_type_val == "clustering":
            labels_pred = np.asarray(y_pred)

        if eval_type_val == "dimensionality_reduction":
            raw_orig = X if X is not None else split_bundle.get("X_train")
            X_original = np.asarray(raw_orig) if raw_orig is not None else None
            X_embedded = np.asarray(y_pred) if y_pred is not None else None

        raw_eval = evaluate_model(
            eval_type_val,
            X,
            labels_pred,
            y_test,
            y_pred,
            X_original,
            X_embedded,
            y,
            int(params.get("n_neighbors", 10)),
        )

        if raw_eval is None or (isinstance(raw_eval, dict) and len(raw_eval) == 0):
            return {
                "status": "evaluated_empty",
                "type": eval_type_val,
                "X_original_shape": getattr(X_original, "shape", None),
                "X_embedded_shape": getattr(X_embedded, "shape", None),
                "note": "evaluate_model returned None/empty for dimensionality_reduction; verify process/evaluation/evaluate.py.",
            }

        return raw_eval

    def one_click(self, filename: str, raw_bytes: bytes, params: dict) -> dict:
        filename_lower = filename.lower()
        bio = io.BytesIO(raw_bytes)
        if filename_lower.endswith(".csv"):
            df = load_csv_file(bio)
        elif filename_lower.endswith(".tsv"):
            df = load_tsv_file(bio)
        elif filename_lower.endswith((".xlsx", ".xls")):
            df = load_xlsx_file(bio)
        elif filename_lower.endswith(".txt"):
            df = pd.read_csv(bio, sep=None, engine="python")
        else:
            raise ValueError("File format not supported.")

        data_type = params.get("data_type", "ordinal")
        dt_val = data_type.value if hasattr(data_type, "value") else data_type

        df = run_preprocess_data(
            df,
            bool(params.get("scale", True)),
            bool(params.get("normalize", False)),
            bool(params.get("encode", True)),
            data_type=dt_val,
        )
        df = _ensure_df(df)

        X_train, X_test, y_train, y_test = run_split_data(
            df=df,
            data_type=dt_val,
            ylabel=params.get("ylabel"),
            test_size=float(params.get("test_size", 0.2)),
            train_size=float(params.get("train_size", 0.8)),
            split_by=params.get("split_by", "train"),
            left_size=float(params.get("left_size", 0.8)),
            right_size=float(params.get("right_size", 0.2)),
            shuffle=bool(params.get("shuffle", True)),
            random_state=int(params.get("random_state", 42)),
        )

        model_type = params.get("model_type", "regression")
        model_name = params.get("model_name", "RandomForestRegressor")
        model_type_val = model_type.value if hasattr(model_type, "value") else model_type
        model_name_val = model_name.value if hasattr(model_name, "value") else model_name

        model, model_desc = _call_select_model(model_type_val, model_name_val, params)

        trained_model = run_train_model(model, X_train, y_train)
        try:
            predictions = run_predict_data(trained_model, X_test)
        except Exception:
            if hasattr(trained_model, "transform"):
                predictions = trained_model.transform(X_test)
            else:
                raise

        is_dim_red = model_type_val == "dimensionality_reduction"
        is_clustering = model_type_val == "clustering"

        evaluation_results = evaluate_model(
            type=model_type_val,
            X=X_test,
            labels_pred=np.asarray(predictions) if is_clustering else None,
            y_test=y_test,
            y_pred=predictions,
            X_original=np.asarray(X_test) if is_dim_red and X_test is not None else None,
            X_embedded=np.asarray(predictions) if is_dim_red and predictions is not None else None,
            y=y_test,
            n_neighbors=int(params.get("n_neighbors", 10)),
        )

        result = {
            "status": "Success",
            "filename": filename,
            "model_info": model_desc,
            "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions,
            "evaluation": evaluation_results,
        }

        for var_name in ("df", "X_train", "X_test", "y_train", "y_test", "model", "trained_model", "predictions"):
            if var_name in locals():
                del locals()[var_name]
        gc.collect()
        return result


engine = LocalPipelineEngine(st.session_state.artifacts)


def parse_optional_float_or_none(raw_value: str):
    cleaned = raw_value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def parse_max_features(raw_value: str):
    cleaned = raw_value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return None
    if cleaned.lower() in {"sqrt", "log2"}:
        return cleaned.lower()
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def parse_priors(raw_value: str):
    cleaned = raw_value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return None
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    parsed = []
    for p in parts:
        try:
            parsed.append(float(p))
        except ValueError:
            parsed.append(p)
    return parsed if parsed else None


def parse_gamma(raw_value: str):
    cleaned = raw_value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return None
    if cleaned.lower() in {"scale", "auto"}:
        return cleaned.lower()
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def api_request(method: str, path: str, **kwargs):
    params = kwargs.get("params") or {}
    files = kwargs.get("files")
    if path == "/upload-data/" and files and "file" in files:
        filename, raw_bytes = files["file"][0], files["file"][1]
        return engine.upload_data(filename, raw_bytes)
    if path == "/preprocess-data/":
        return engine.preprocess_data(params)
    if path == "/split-data/":
        return engine.split_data(params)
    if path == "/select-model/":
        return engine.select_model(params)
    if path == "/train-model/":
        return engine.train_model(params)
    if path == "/predict-data/":
        return engine.predict_data(params)
    if path == "/evaluate-model/":
        return engine.evaluate_model(params)
    if path == "/one-click/" and files and "file" in files:
        filename, raw_bytes = files["file"][0], files["file"][1]
        return engine.one_click(filename, raw_bytes, params)
    raise ValueError(f"Unsupported local path: {path}")



import inspect


_MODEL_VALID_LOSSES = {
    "HistGradientBoostingClassifier": {"log_loss"},
    "HistGradientBoostingRegressor": {"squared_error", "absolute_error", "huber", "quantile"},
    "GradientBoostingClassifier": {"log_loss"},
    "GradientBoostingRegressor": {"squared_error", "absolute_error", "huber", "quantile"},
    "SGDClassifier": {"hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"},
    "SGDRegressor": {"squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"},
}

_MODEL_DEFAULT_LOSS = {
    "HistGradientBoostingClassifier": "log_loss",
    "HistGradientBoostingRegressor": "squared_error",
    "GradientBoostingClassifier": "log_loss",
    "GradientBoostingRegressor": "squared_error",
    "SGDClassifier": "hinge",
    "SGDRegressor": "squared_error",
}

_MODEL_VALID_CRITERIA = {
    "DecisionTreeClassifier": {"gini", "entropy", "log_loss"},
    "RandomForestClassifier": {"gini", "entropy", "log_loss"},
    "ExtraTreesClassifier": {"gini", "entropy", "log_loss"},
    "DecisionTreeRegressor": {"squared_error", "friedman_mse", "absolute_error", "poisson"},
    "RandomForestRegressor": {"squared_error", "friedman_mse", "absolute_error", "poisson"},
    "ExtraTreesRegressor": {"squared_error", "friedman_mse", "absolute_error", "poisson"},
}

_MODEL_DEFAULT_CRITERION = {
    "DecisionTreeClassifier": "gini",
    "RandomForestClassifier": "gini",
    "ExtraTreesClassifier": "gini",
    "DecisionTreeRegressor": "squared_error",
    "RandomForestRegressor": "squared_error",
    "ExtraTreesRegressor": "squared_error",
}


def sanitize_model_params(model_type_val: str, model_name_val: str, params: dict) -> dict:
    cleaned = dict(params)

    # 1. Loss uyumluluğu
    raw_loss = cleaned.get("loss")
    if model_name_val in _MODEL_VALID_LOSSES:
        if raw_loss not in _MODEL_VALID_LOSSES[model_name_val]:
            cleaned["loss"] = _MODEL_DEFAULT_LOSS[model_name_val]
    elif model_type_val == "classification" and raw_loss == "squared_error":
        cleaned["loss"] = "log_loss"
    elif model_type_val == "regression" and raw_loss in {"hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"}:
        cleaned["loss"] = "squared_error"

    # 2. Criterion uyumluluğu
    raw_criterion = cleaned.get("criterion")
    if model_name_val in _MODEL_VALID_CRITERIA:
        if raw_criterion not in _MODEL_VALID_CRITERIA[model_name_val]:
            cleaned["criterion"] = _MODEL_DEFAULT_CRITERION[model_name_val]
    elif model_type_val == "classification" and raw_criterion in {"squared_error", "friedman_mse", "absolute_error", "poisson"}:
        cleaned["criterion"] = "gini"
    elif model_type_val == "regression" and raw_criterion in {"gini", "entropy", "log_loss"}:
        cleaned["criterion"] = "squared_error"

    # 3. Kernel uyumluluğu (SVC / SVR / KernelPCA vb.)
    if cleaned.get("kernel") == "precomputed":
        cleaned["kernel"] = "rbf"

    # 4. t-SNE init uyumluluğu
    if model_name_val == "TSNE" and cleaned.get("init") not in {"pca", "random"}:
        cleaned["init"] = "pca"

    # 5. run_select_model imzasıyla eşleşmeyen ekstra anahtarları ayıkla
    try:
        sig = inspect.signature(run_select_model)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not accepts_kwargs:
            valid_keys = set(sig.parameters.keys())
            cleaned = {k: v for k, v in cleaned.items() if k in valid_keys}
    except Exception:
        pass

    return cleaned


col_title, col_status = st.columns([4, 1], vertical_alignment="center")
with col_title:
    st.title("Machine Learning Stream")
    st.caption("Step-by-step & one-click local machine learning pipeline interface")

with col_status:
    st.success("Local Engine Active")
    st.caption("Mode: Direct Python (No API)")

st.divider()

with st.sidebar:
    st.header("Pipeline Arrangements")

    if st.button("Reset Pipeline State"):
        for k, v in DEFAULT_STATE.items():
            st.session_state[k] = {} if k == "artifacts" else v
        st.rerun()

    st.markdown("### Active Pipeline IDs")
    st.code(
        f"temp_id:         {st.session_state.temp_id}\n"
        f"prep_temp_id:    {st.session_state.preprocessed_temp_id}\n"
        f"split_id:        {st.session_state.split_id}\n"
        f"model_id:        {st.session_state.model_id}\n"
        f"predictions_id:  {st.session_state.predictions_id}",
        language="text",
    )

    with st.expander("1. Data Pre-processing", expanded=True):
        data_type = st.selectbox("Data Type", ["ordinal", "nonordinal", "image"], index=0)
        scale = st.checkbox("Scale", value=True)
        normalize = st.checkbox("Normalization (Normalize)", value=False)
        encode = st.checkbox("Encode", value=True)

    with st.expander("2. Train-Test & Split Settings", expanded=True):
        available_cols = st.session_state.columns
        if available_cols:
            ylabel_option = st.selectbox(
                "Target Column (ylabel)",
                options=["<None>"] + available_cols,
                index=0,
            )
            ylabel_custom = st.text_input("Or custom ylabel", value="")
            ylabel = ylabel_custom.strip() or (None if ylabel_option == "<None>" else ylabel_option)
        else:
            ylabel_input = st.text_input("Target Column (ylabel, optional)", value="")
            ylabel = ylabel_input.strip() or None

        huggingface_dataset_input = st.text_input("HuggingFace Dataset (optional)", value="")
        huggingface_dataset = huggingface_dataset_input.strip() or None

        split_by = st.selectbox("Split By", ["train", "test", "left", "right"], index=0)
        test_size = st.slider("Test Size", 0.0, 0.9, 0.2, 0.05)
        train_size = st.slider("Train Size", 0.0, 1.0, 0.8, 0.05)
        left_size = st.slider("Left Size", 0.0, 1.0, 0.8, 0.05)
        right_size = st.slider("Right Size", 0.0, 1.0, 0.2, 0.05)
        shuffle = st.checkbox("Shuffle", value=True)
        random_state = st.number_input("Random State", value=42, step=1)

    with st.expander("3. Model Selection", expanded=True):
        model_types = list(MODEL_CATALOG.keys())
        model_type = st.selectbox("Problem Type", model_types, index=0)
        model_name = st.selectbox("Algorithm", MODEL_CATALOG[model_type], index=0)

    with st.expander("4. General & Linear / Tree / Neighbor Hyperparameters", expanded=False):
        alpha = st.number_input("Alpha", value=1.0, step=0.1, format="%.4f")
        C = st.number_input("C", value=1.0, step=0.1, format="%.4f")
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.05)
        dual = st.checkbox("Dual", value=True)
        fit_intercept = st.checkbox("Fit Intercept", value=True)
        max_iter = st.number_input("Max Iter", min_value=1, value=100, step=10)
        selection = st.selectbox("Selection", ["random", "cyclic"], index=0)
        degree = st.number_input("Degree", min_value=1, value=3, step=1)
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid", "precomputed"], index=0)
        max_depth = st.number_input("Max Depth", min_value=1, value=5, step=1)
        neighbors = st.number_input("Neighbors", min_value=1, value=5, step=1)
        n_estimators = st.number_input("N Estimators", min_value=1, value=100, step=10)
        weights = st.selectbox("Weights", ["uniform", "distance"], index=0)
        algorithm = st.selectbox(
            "Algorithm (Nearest / Tree)", ["auto", "ball_tree", "kd_tree", "brute"], index=0
        )
        criterion = st.selectbox(
            "Criterion",
            [
                "squared_error",
                "friedman_mse",
                "absolute_error",
                "poisson",
                "gini",
                "entropy",
                "log_loss",
            ],
            index=0,
        )
        n_neighbors = st.number_input("N Neighbors", min_value=1, value=10, step=1)
        max_features_raw = st.text_input(
            "Max Features (int, float, 'sqrt', 'log2', or None)", value="sqrt"
        )
        priors_raw = st.text_input("Priors (comma-separated floats or None)", value="")
        loss = st.selectbox(
            "Loss",
            [
                "squared_error",
                "log_loss",
                "absolute_error",
                "huber",
                "quantile",
                "hinge",
                "modified_huber",
                "squared_hinge",
                "perceptron",
            ],
            index=0,
        )
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, value=0.05, step=0.01, format="%.4f"
        )

    with st.expander("5. Clustering & Dimensionality Reduction Hyperparameters", expanded=False):
        n_clusters = st.number_input("N Clusters", min_value=1, value=3, step=1)
        eps = st.number_input("Eps", min_value=0.01, value=0.5, step=0.05, format="%.4f")
        min_samples = st.number_input("Min Samples", min_value=1, value=5, step=1)
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)
        bandwidth_raw = st.text_input("Bandwidth (float or None)", value="")
        damping = st.slider("Damping", 0.5, 0.99, 0.5, 0.01)
        n_components = st.number_input("N Components", min_value=1, value=2, step=1)
        gamma_raw = st.text_input("Gamma ('scale', 'auto', float, or None)", value="scale")
        perplexity = st.number_input("Perplexity", min_value=1.0, value=30.0, step=1.0)
        init = st.selectbox("Init", ["pca", "random"], index=0)


def normalize_model_type(raw_type: str) -> str:
    cleaned = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "regression": "regression",
        "classification": "classification",
        "clustering": "clustering",
        "dimensionality_reduction": "dimensionality_reduction",
        "dim_reduction": "dimensionality_reduction",
    }
    return mapping.get(cleaned, cleaned)


def build_select_params() -> dict:
    return {
        "model_type": normalize_model_type(model_type),
        "model_name": model_name,
        "alpha": float(alpha),
        "C": float(C),
        "l1_ratio": float(l1_ratio),
        "dual": bool(dual),
        "fit_intercept": bool(fit_intercept),
        "max_iter": int(max_iter),
        "selection": selection,
        "degree": int(degree),
        "kernel": kernel,
        "max_depth": int(max_depth),
        "neighbors": int(neighbors),
        "n_estimators": int(n_estimators),
        "weights": weights,
        "algorithm": algorithm,
        "criterion": criterion,
        "n_neighbors": int(n_neighbors),
        "max_features": parse_max_features(max_features_raw),
        "priors": parse_priors(priors_raw),
        "loss": loss,
        "learning_rate": float(learning_rate),
        "n_clusters": int(n_clusters),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "linkage": linkage,
        "bandwidth": parse_optional_float_or_none(bandwidth_raw),
        "damping": float(damping),
        "n_components": int(n_components),
        "gamma": parse_gamma(gamma_raw),
        "perplexity": float(perplexity),
        "init": init,
    }


tab_steps, tab_one_click = st.tabs(["Step-by-Step Pipeline", "One-Click Pipeline"])

with tab_steps:
    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a dataset file (.csv, .tsv, .xlsx, .xls, .txt)",
        type=["csv", "tsv", "xlsx", "xls", "txt"],
    )
    if st.button("Upload File", key="btn_upload"):
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            try:
                with st.spinner("Uploading and parsing dataset..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    data = api_request("POST", "/upload-data/", files=files)
                st.session_state.temp_id = data.get("temp_id")
                st.session_state.preprocessed_temp_id = None
                st.session_state.split_id = None
                st.session_state.predictions_id = None
                st.session_state.columns = data.get("columns", [])
                st.session_state.upload_info = data
                st.success(f"Uploaded! temp_id: {st.session_state.temp_id}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.upload_info:
        info = st.session_state.upload_info
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", info.get("rows", "-"))
        c2.metric("Columns", len(info.get("columns", [])))
        c3.code(f"temp_id: {info.get('temp_id')}")
        if info.get("preview"):
            st.dataframe(pd.DataFrame(info["preview"]), use_container_width=True)

    st.divider()

    st.subheader("2. Preprocess Data")
    prep_temp_id_input = st.text_input(
        "temp_id for preprocessing (leave empty to use latest uploaded data)",
        value=st.session_state.temp_id or "",
        key="prep_temp_id_input",
    )
    if st.button("Run Preprocessing", key="btn_preprocess"):
        try:
            with st.spinner("Preprocessing dataset..."):
                params = {
                    "scale": bool(scale),
                    "normalize": bool(normalize),
                    "encode": bool(encode),
                    "data_type": data_type,
                }
                if prep_temp_id_input.strip():
                    params["temp_id"] = prep_temp_id_input.strip()
                data = api_request("POST", "/preprocess-data/", params=params)
            st.session_state.preprocessed_temp_id = data.get("temp_id")
            st.session_state.split_id = None
            st.session_state.predictions_id = None
            st.session_state.columns = data.get("columns", st.session_state.columns)
            st.session_state.preprocess_info = data
            st.success(f"Preprocessed! New temp_id: {st.session_state.preprocessed_temp_id}")
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")

    if st.session_state.preprocess_info:
        pinfo = st.session_state.preprocess_info
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed Rows", pinfo.get("rows", "-"))
        c2.metric("Processed Columns", len(pinfo.get("columns", [])))
        c3.code(f"temp_id: {pinfo.get('temp_id')}")
        if pinfo.get("preview"):
            st.dataframe(pd.DataFrame(pinfo["preview"]), use_container_width=True)

    st.divider()

    st.subheader("3. Split Data")
    default_split_source = (
        st.session_state.preprocessed_temp_id or st.session_state.temp_id or ""
    )
    split_temp_id_input = st.text_input(
        "temp_id for splitting (leave empty to auto-use latest available dataset)",
        value=default_split_source,
        key="split_temp_id_input",
    )
    if st.button("Split Dataset", key="btn_split"):
        try:
            with st.spinner("Splitting dataset..."):
                params = {
                    "data_type": data_type,
                    "test_size": float(test_size),
                    "train_size": float(train_size),
                    "split_by": split_by,
                    "left_size": float(left_size),
                    "right_size": float(right_size),
                    "shuffle": bool(shuffle),
                    "random_state": int(random_state),
                }
                if split_temp_id_input.strip():
                    params["temp_id"] = split_temp_id_input.strip()
                if ylabel:
                    params["ylabel"] = ylabel
                if huggingface_dataset:
                    params["huggingface_dataset"] = huggingface_dataset

                data = api_request("POST", "/split-data/", params=params)
            st.session_state.split_id = data.get("split_id")
            st.session_state.predictions_id = None
            st.session_state.split_info = data
            st.success(f"Split complete! split_id: {st.session_state.split_id}")
        except Exception as exc:
            st.error(f"Split failed: {exc}")

    if st.session_state.split_info:
        st.json(st.session_state.split_info)

    st.divider()

    st.subheader("4. Select Model")
    if st.button("Create / Select Model", key="btn_select"):
        try:
            with st.spinner("Initializing model..."):
                data = api_request("POST", "/select-model/", params=build_select_params())
            st.session_state.model_id = data.get("model_id")
            st.session_state.select_info = data
            st.success(f"Model selected! model_id: {st.session_state.model_id}")
        except Exception as exc:
            st.error(f"Model selection failed: {exc}")

    if st.session_state.select_info:
        st.json(st.session_state.select_info)

    st.divider()

    st.subheader("5. Train Model")
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        tr_model_id = st.text_input(
            "model_id (optional, defaults to latest)",
            value=st.session_state.model_id or "",
            key="tr_model_id",
        )
    with col_tr2:
        tr_split_id = st.text_input(
            "split_id (optional, defaults to latest)",
            value=st.session_state.split_id or "",
            key="tr_split_id",
        )

    if st.button("Train Model", key="btn_train"):
        try:
            with st.spinner("Training model..."):
                params = {}
                if tr_model_id.strip():
                    params["model_id"] = tr_model_id.strip()
                if tr_split_id.strip():
                    params["split_id"] = tr_split_id.strip()
                data = api_request("POST", "/train-model/", params=params)
            if data.get("model_id"):
                st.session_state.model_id = data["model_id"]
            if data.get("split_id"):
                st.session_state.split_id = data["split_id"]
            st.session_state.train_info = data
            st.success(f"Training completed! status: {data.get('status')}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    if st.session_state.train_info:
        st.json(st.session_state.train_info)

    st.divider()

    st.subheader("6. Predict")
    col_pr1, col_pr2 = st.columns(2)
    with col_pr1:
        pr_model_id = st.text_input(
            "model_id for prediction (optional)",
            value=st.session_state.model_id or "",
            key="pr_model_id",
        )
    with col_pr2:
        pr_split_id = st.text_input(
            "split_id for prediction (optional)",
            value=st.session_state.split_id or "",
            key="pr_split_id",
        )

    if st.button("Generate Predictions", key="btn_predict"):
        try:
            with st.spinner("Running inference..."):
                params = {}
                if pr_model_id.strip():
                    params["model_id"] = pr_model_id.strip()
                if pr_split_id.strip():
                    params["split_id"] = pr_split_id.strip()
                data = api_request("POST", "/predict-data/", params=params)
            st.session_state.predictions_id = data.get("predictions_id")
            st.session_state.predict_info = data
            st.success(f"Predictions ready! predictions_id: {st.session_state.predictions_id}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if st.session_state.predict_info:
        p_res = st.session_state.predict_info
        st.code(f"predictions_id: {p_res.get('predictions_id')}")
        preds = p_res.get("predictions")
        if isinstance(preds, list):
            st.dataframe(pd.DataFrame({"prediction": preds}), use_container_width=True)
        else:
            st.json(preds)

    st.divider()

    st.subheader("7. Evaluate Model")
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    with col_ev1:
        eval_type = st.selectbox(
            "Evaluation Type",
            ["regression", "classification", "clustering", "dimensionality_reduction"],
            index=0,
        )
    with col_ev2:
        ev_split_id = st.text_input(
            "split_id (optional)",
            value=st.session_state.split_id or "",
            key="ev_split_id",
        )
    with col_ev3:
        ev_predictions_id = st.text_input(
            "predictions_id (optional)",
            value=st.session_state.predictions_id or "",
            key="ev_predictions_id",
        )

    if st.button("Evaluate Model", key="btn_evaluate"):
        try:
            with st.spinner("Evaluating model..."):
                params = {
                    "type": eval_type,
                    "n_neighbors": int(n_neighbors),
                }
                if ev_split_id.strip():
                    params["split_id"] = ev_split_id.strip()
                if ev_predictions_id.strip():
                    params["predictions_id"] = ev_predictions_id.strip()
                data = api_request("POST", "/evaluate-model/", params=params)
            st.session_state.evaluate_info = data
            st.success("Evaluation completed!")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    if st.session_state.evaluate_info is not None:
        st.json(st.session_state.evaluate_info)


with tab_one_click:
    st.subheader("One-Click End-to-End Pipeline")
    oc_file = st.file_uploader(
        "Upload file for one-click run (.csv, .tsv, .xlsx, .xls, .txt)",
        type=["csv", "tsv", "xlsx", "xls", "txt"],
        key="oc_file_uploader",
    )

    if st.button("Run One-Click Pipeline", key="btn_one_click"):
        if oc_file is None:
            st.warning("Please upload a file first.")
        else:
            try:
                with st.spinner("Running full pipeline locally..."):
                    files = {
                        "file": (
                            oc_file.name,
                            oc_file.getvalue(),
                            oc_file.type or "application/octet-stream",
                        )
                    }
                    params = {
                        "scale": bool(scale),
                        "normalize": bool(normalize),
                        "encode": bool(encode),
                        "data_type": data_type,
                        "test_size": float(test_size),
                        "train_size": float(train_size),
                        "split_by": split_by,
                        "left_size": float(left_size),
                        "right_size": float(right_size),
                        "shuffle": bool(shuffle),
                        "random_state": int(random_state),
                        **build_select_params(),
                    }
                    if ylabel:
                        params["ylabel"] = ylabel

                    data = api_request("POST", "/one-click/", files=files, params=params)
                st.session_state.one_click_info = data
                st.success("One-click pipeline finished!")
            except Exception as exc:
                st.error(f"One-click pipeline failed: {exc}")

    if st.session_state.one_click_info:
        oc = st.session_state.one_click_info
        st.json(
            {
                "status": oc.get("status"),
                "training": "trained",
                "filename": oc.get("filename"),
                "model_info": oc.get("model_info"),
                "evaluation": oc.get("evaluation"),
            }
        )
        if oc.get("predictions") is not None:
            st.dataframe(pd.DataFrame({"prediction": oc["predictions"]}), use_container_width=True)