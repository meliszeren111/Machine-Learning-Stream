import streamlit as st
import requests
import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.config import API_URL

st.set_page_config(
    page_title="Machine Learning Stream",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .status-badge {
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-ok {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .status-err {
        background-color: #f8d7da;
        color: #842029;
    }
    </style>
    """, unsafe_allow_html=True)

MODEL_CATALOG = {
    "Regression": [
        "RandomForestRegressor", "GradientBoostingRegressor", "HistGradientBoostingRegressor",
        "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor", "GaussianProcessRegressor",
        "LinearRegression", "Lasso", "Ridge", "LassoCV", "RidgeCV", "KNeighborsRegressor",
        "MLPRegressor", "SVR", "DecisionTreeRegressor", "XGBRegressor", "LGBMRegressor",
        "CatBoostRegressor", "PolynomialRegression", "SupportVectorRegressor", "MultiLayerPerceptron"
    ],
    "Classification": [
        "RandomForestClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier",
        "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier", "LogisticRegression",
        "GaussianNB", "KNeighborsClassifier", "SVC", "DecisionTreeClassifier", "XGBClassifier",
        "LGBMClassifier", "CatBoostClassifier", "KNN", "NaiveBayes", "SupportVectorClassifier"
    ],
    "Clustering": [
        "KMeans", "DBSCAN", "AgglomerativeClustering", "SpectralClustering",
        "MeanShift", "AffinityPropagation", "OPTICS", "Birch", "HDBSCAN", "GaussianMixture"
    ],
    "Dimensionality Reduction": [
        "PCA", "TruncatedSVD", "KernelPCA", "FastICA", "FactorAnalysis",
        "NMF", "DictionaryLearning", "LDA", "QDA", "TSNE", "UMAP"
    ]
}


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


col_title, col_status = st.columns([4, 1], vertical_alignment="center")
with col_title:
    st.title("Machine Learning Stream")
    st.caption("FastAPI-based end-to-end machine learning training and evaluation interface")

with col_status:
    try:
        res = requests.get(f"{API_URL}/", timeout=2)
        if res.status_code == 200:
            st.markdown('<div style="text-align:right;"><span class="status-badge status-ok">API Connected</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:right;"><span class="status-badge status-err">API Error</span></div>', unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div style="text-align:right;"><span class="status-badge status-err">No Connection</span></div>', unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.header("Pipeline Arrangements")

    with st.expander("1. Data Pre-processing", expanded=True):
        data_type = st.selectbox("Data Type", ["ordinal", "nonordinal", "image"], index=0)
        scale = st.checkbox("Scale", value=True)
        normalize = st.checkbox("Normalization (Normalize)", value=False)
        encode = st.checkbox("Encode", value=True)

    with st.expander("2. Train-Test & Split Settings", expanded=False):
        split_by = st.selectbox("Split By", ["train", "test", "left", "right"], index=0)
        test_size = st.slider("Test Size", 0.0, 0.9, 0.2, 0.05)
        train_size = st.slider("Train Size", 0.0, 1.0, 0.8, 0.05)
        left_size = st.slider("Left Size", 0.0, 1.0, 0.8, 0.05)
        right_size = st.slider("Right Size", 0.0, 1.0, 0.2, 0.05)
        shuffle = st.checkbox("Shuffle", value=True)
        random_state = st.number_input("Random State", value=42, step=1)

    with st.expander("3. Model Selection", expanded=True):
        model_type = st.selectbox("Problem Type", list(MODEL_CATALOG.keys()), index=0)
        model_name = st.selectbox("Algorithm", MODEL_CATALOG[model_type], index=0)

    with st.expander("4. General & Linear / Tree / Neighbor Hyperparameters", expanded=True):
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
        algorithm = st.selectbox("Algorithm (Nearest / Tree)", ["auto", "ball_tree", "kd_tree", "brute"], index=0)
        criterion = st.selectbox(
            "Criterion",
            ["entropy", "gini", "log_loss", "squared_error", "friedman_mse", "absolute_error", "poisson"],
            index=0
        )
        n_neighbors = st.number_input("N Neighbors", min_value=1, value=10, step=1)
        max_features_raw = st.text_input("Max Features (int, float, 'sqrt', 'log2', or None)", value="4")
        priors_raw = st.text_input("Priors (comma-separated floats or None)", value="")
        loss = st.selectbox(
            "Loss",
            ["log_loss", "squared_error", "absolute_error", "huber", "quantile", "hinge", "modified_huber", "squared_hinge", "perceptron"],
            index=0
        )
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.05, step=0.01, format="%.4f")

    with st.expander("5. Clustering & Dimensionality Reduction Hyperparameters", expanded=False):
        n_clusters = st.number_input("N Clusters", min_value=1, value=3, step=1)
        eps = st.number_input("Eps", min_value=0.01, value=0.5, step=0.05, format="%.4f")
        min_samples = st.number_input("Min Samples", min_value=1, value=5, step=1)
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)
        bandwidth_raw = st.text_input("Bandwidth (float or None)", value="")
        damping = st.slider("Damping", 0.5, 0.99, 0.5, 0.01)
        n_components = st.number_input("N Components", min_value=1, value=2, step=1)
        gamma_raw = st.text_input("Gamma (float, 'scale', 'auto', or None)", value="")
        perplexity = st.number_input("Perplexity", min_value=1.0, value=30.0, step=1.0, format="%.2f")
        init = st.selectbox("Init", ["pca", "random", "k-means++"], index=0)

with st.container(border=True):
    st.subheader("1. Loading Dataset")
    uploaded_file = st.file_uploader(
        "Select or drag the data file (.csv, .tsv, .xlsx, .txt)",
        type=["csv", "tsv", "xlsx", "xls", "txt"]
    )

    columns = []
    if uploaded_file is not None:
        if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("The dataset is being analyzed..."):
                try:
                    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                    uploaded_file.seek(0)
                    if ext in {"xlsx", "xls"}:
                        df_preview = pd.read_excel(uploaded_file)
                    elif ext == "tsv":
                        df_preview = pd.read_csv(uploaded_file, sep="\t")
                    else:
                        df_preview = pd.read_csv(uploaded_file, sep=None, engine="python")

                    st.session_state.preview_data = {
                        "filename": uploaded_file.name,
                        "rows": len(df_preview),
                        "columns": list(df_preview.columns),
                        "preview": df_preview.head(10).to_dict(orient="records"),
                    }
                    st.session_state.uploaded_filename = uploaded_file.name
                except Exception as e:
                    st.warning(f"Local preview could not be parsed ({e}). You can still run training directly via /one-click/.")
                    st.session_state.preview_data = {
                        "filename": uploaded_file.name,
                        "rows": "N/A",
                        "columns": [],
                        "preview": [],
                    }
                    st.session_state.uploaded_filename = uploaded_file.name

    if uploaded_file is not None and "preview_data" in st.session_state:
        pdata = st.session_state.preview_data
        columns = pdata.get("columns", [])
        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("File name", pdata["filename"])
        m2.metric("Number of Rows", pdata["rows"])
        m3.metric("Number of Columns", len(columns))
        m4.metric("Selected Model", model_name)

        if pdata.get("preview"):
            st.dataframe(pd.DataFrame(pdata["preview"]), use_container_width=True)

    col_target, col_btn = st.columns([3, 1], vertical_alignment="bottom")
    with col_target:
        if columns:
            default_idx = columns.index("target") if "target" in columns else len(columns) - 1
            ylabel = st.selectbox(
                "Target Column (ylabel)",
                options=columns,
                index=default_idx
            )
        else:
            ylabel = st.text_input("Target Column (ylabel)", value="target")

    with col_btn:
        run_button = st.button("Start Training", type="primary")

    if run_button:
        if uploaded_file is None:
            st.warning("Please upload a file before starting training.")
        else:
            with st.spinner(f"{model_name} model is being trained..."):
                query_params = {
                    "scale": scale,
                    "normalize": normalize,
                    "encode": encode,
                    "data_type": data_type,
                    "ylabel": ylabel,
                    "test_size": test_size,
                    "train_size": train_size,
                    "split_by": split_by,
                    "left_size": left_size,
                    "right_size": right_size,
                    "shuffle": shuffle,
                    "random_state": int(random_state),
                    "model_type": str(model_type).strip().lower(),
                    "model_name": model_name,
                    "alpha": float(alpha),
                    "C": float(C),
                    "l1_ratio": float(l1_ratio),
                    "dual": dual,
                    "fit_intercept": fit_intercept,
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
                    "gamma": parse_optional_float_or_none(gamma_raw),
                    "perplexity": float(perplexity),
                    "init": init,
                }

                cleaned_query_params = {k: v for k, v in query_params.items() if v is not None}

                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                try:
                    response = requests.post(
                        f"{API_URL}/one-click/",
                        params=cleaned_query_params,
                        files=files
                    )
                    if response.status_code == 200:
                        st.session_state.last_result = response.json()
                        st.session_state.last_params = query_params
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Request Error: {str(e)}")


if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.markdown("")
    with st.container(border=True):
        st.subheader("2. Results of the training")
        st.success("Model training and evaluation have been completed..")

        tab1, tab2, tab3 = st.tabs(["Evaluation Metrics", "Predictions and Graph", "Model Details"])

        with tab1:
            eval_res = result.get("evaluation", {})
            if isinstance(eval_res, dict) and eval_res:
                numeric_items = {k: v for k, v in eval_res.items() if isinstance(v, (int, float))}
                other_items = {k: v for k, v in eval_res.items() if not isinstance(v, (int, float))}

                if numeric_items:
                    cols = st.columns(min(len(numeric_items), 4))
                    for idx, (metric_name, metric_val) in enumerate(numeric_items.items()):
                        cols[idx % len(cols)].metric(metric_name.upper(), f"{metric_val:.4f}")

                for k, v in other_items.items():
                    st.write(f"**{k}:** {v}")
            else:
                st.json(eval_res)

        with tab2:
            preds = result.get("predictions", [])
            pred_df = pd.DataFrame({"Prediction": preds})

            g_col, t_col = st.columns([2, 1])
            with g_col:
                st.caption("Prediction Series")
                st.line_chart(pred_df, use_container_width=True)
            with t_col:
                st.caption("Prediction Table")
                st.dataframe(pred_df, use_container_width=True, height=280)

            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"{model_name}_predictions.csv",
                mime="text/csv"
            )

        with tab3:
            st.info(f"**Model Description:** {result.get('model_info', 'Information is not available.')}")
            st.json(st.session_state.get("last_params", {}))
