import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.config import API_URL
from data.model_catalogue import MODEL_CATALOG

BASE_URL = API_URL.rstrip("/")

st.set_page_config(
    page_title="Machine Learning Stream",
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
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    url = f"{BASE_URL}{path}"
    timeout = kwargs.pop("timeout", 120)
    resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
    return resp.json()


col_title, col_status = st.columns([4, 1], vertical_alignment="center")
with col_title:
    st.title("Machine Learning Stream")
    st.caption("Step-by-step & one-click FastAPI machine learning pipeline interface")

with col_status:
    try:
        res = requests.get(f"{BASE_URL}/", timeout=15)
        if res.status_code == 200:
            st.success("API Connected")
        else:
            st.error(f"API Error: {res.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
    st.caption(f"API URL: {BASE_URL}")

st.divider()

# Sidebar: Pipeline & Hyperparameter Settings
with st.sidebar:
    st.header("Pipeline Arrangements")

    if st.button("Reset Pipeline State"):
        for k, v in DEFAULT_STATE.items():
            st.session_state[k] = v
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


# TAB 1: Step-by-Step Pipeline
with tab_steps:
    # 1. Upload
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

    # 2. Preprocess
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

    # 3. Split
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

    # 4. Select Model
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

    # 5. Train
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

    # 6. Predict
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

    # 7. Evaluate
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

    if st.session_state.evaluate_info:
        st.json(st.session_state.evaluate_info)



# TAB 2: One-Click Pipeline
with tab_one_click:
    st.subheader("One-Click End-to-End Pipeline (/one-click/)")
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
                with st.spinner("Running full pipeline on server..."):
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