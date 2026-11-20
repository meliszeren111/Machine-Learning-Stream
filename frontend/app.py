import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="AutoML Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

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
    </style>
    """, unsafe_allow_html=True)

MODEL_CATALOG = {
    "regression": [
        "RandomForestRegressor", "GradientBoostingRegressor", "HistGradientBoostingRegressor",
        "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor", "GaussianProcessRegressor",
        "LinearRegression", "Lasso", "Ridge", "LassoCV", "RidgeCV", "KNeighborsRegressor",
        "MLPRegressor", "SVR", "DecisionTreeRegressor", "XGBRegressor", "LGBMRegressor",
        "CatBoostRegressor", "PolynomialRegression", "SupportVectorRegressor", "MultiLayerPerceptron"
    ],
    "classification": [
        "RandomForestClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier",
        "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier", "LogisticRegression",
        "GaussianNB", "KNeighborsClassifier", "SVC", "DecisionTreeClassifier", "XGBClassifier",
        "LGBMClassifier", "CatBoostClassifier", "KNN", "NaiveBayes", "SupportVectorClassifier"
    ],
    "clustering": [
        "KMeans", "DBSCAN", "AgglomerativeClustering", "SpectralClustering",
        "MeanShift", "AffinityPropagation", "OPTICS", "Birch", "HDBSCAN", "GaussianMixture"
    ],
    "dimensionality_reduction": [
        "PCA", "TruncatedSVD", "KernelPCA", "FastICA", "FactorAnalysis",
        "NMF", "DictionaryLearning", "LDA", "QDA", "TSNE", "UMAP"
    ]
}

col_title, col_status = st.columns([4, 1], vertical_alignment="center")
with col_title:
    st.title("AutoML Implementation Studio")
    st.caption("FastAPI-based end-to-end machine learning training and evaluation interface")

with col_status:
    try:
        res = requests.get(f"{API_URL}/", timeout=2)
        if res.status_code == 200:
            st.markdown('<div style="text-align:right;"><span class="status-badge status-ok">API Bagli</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:right;"><span class="status-badge status-err">API Hatasi</span></div>', unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div style="text-align:right;"><span class="status-badge status-err">Baglanti Yok</span></div>', unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.header("Pipeline Ayarlari")

    with st.expander("1. Data Pre-processing", expanded=True):
        data_type = st.selectbox("Data Type", ["ordinal", "nonordinal", "image"])
        scale = st.checkbox("Scale", value=True)
        normalize = st.checkbox("Normalization (Normalize)", value=False)
        encode = st.checkbox("Encode", value=True)

    with st.expander("2. Train-Test Split", expanded=False):
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        train_size = round(1.0 - test_size, 2)
        shuffle = st.checkbox("Shuffle", value=True)
        random_state = st.number_input("Random State", value=42, step=1)

    with st.expander("3. Model Selection", expanded=True):
        model_type = st.selectbox("Problem Type", list(MODEL_CATALOG.keys()))
        model_name = st.selectbox("Algorithm", MODEL_CATALOG[model_type])

    with st.expander("4. Hiperparameters", expanded=True):
        params = {}

        if model_type in ["Regression", "Classification"]:
            if any(k in model_name for k in ["Forest", "Boosting", "Trees", "XGB", "LGBM", "CatBoost", "Bagging"]):
                params["n_estimators"] = st.number_input("Number of Trees (n_estimators)", 10, 1000, 100, 10)
                params["max_depth"] = st.slider("Maximum Depth", 1, 50, 5)

            if any(k in model_name for k in ["Boosting", "XGB", "LGBM", "CatBoost", "MLP"]):
                params["learning_rate"] = st.number_input("Learning Rate", 0.001, 1.0, 0.05, format="%.3f")

            if any(k in model_name for k in ["Lasso", "Ridge"]):
                params["alpha"] = st.number_input("Alpha", 0.01, 100.0, 1.0)

            if any(k in model_name for k in ["SV", "Logistic"]):
                params["C"] = st.number_input("C Value", 0.01, 100.0, 1.0)
                params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])

            if "Neighbors" in model_name or "KNN" in model_name:
                params["n_neighbors"] = st.slider("Number of Neightbors (n_neighbors)", 1, 50, 5)
                params["weights"] = st.selectbox("Weights", ["uniform", "distance"])

        elif model_type == "clustering":
            if model_name in ["KMeans", "AgglomerativeClustering", "SpectralClustering", "Birch", "GaussianMixture"]:
                params["n_clusters"] = st.slider("Number of Clusters (n_clusters)", 2, 20, 3)
            if model_name in ["DBSCAN", "OPTICS"]:
                params["eps"] = st.number_input("Epsilon (eps)", 0.1, 10.0, 0.5)
                params["min_samples"] = st.slider("Min Samples", 1, 50, 5)

        elif model_type == "dimensionality_reduction":
            params["n_components"] = st.slider("Number of Components (n_components)", 2, 50, 2)
            if model_name == "TSNE":
                params["perplexity"] = st.slider("Perplexity", 5.0, 100.0, 30.0)

with st.container(border=True):
    st.subheader("1. Loading Dataset")
    uploaded_file = st.file_uploader(
        "Select or drag the data file (.csv, .tsv, .xlsx, .txt)",
        type=["csv", "tsv", "xlsx", "xls", "txt"]
    )

    if uploaded_file is not None:
        if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("The dataset is being analyzed..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    upload_res = requests.post(f"{API_URL}/upload-data/", files=files)
                    if upload_res.status_code == 200:
                        st.session_state.preview_data = upload_res.json()
                        st.session_state.uploaded_filename = uploaded_file.name
                    else:
                        st.error(f"File reading error: {upload_res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    if "preview_data" in st.session_state:
        pdata = st.session_state.preview_data
        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("File name", pdata["filename"])
        m2.metric("Number of Rows", pdata["rows"])
        m3.metric("Number of Columns", len(pdata["columns"]))
        m4.metric("Selected Model", model_name)

        st.dataframe(pd.DataFrame(pdata["preview"]), use_container_width=True)

        col_target, col_btn = st.columns([3, 1], vertical_alignment="bottom")
        with col_target:
            ylabel = st.selectbox(
                "Target Column",
                options=pdata["columns"],
                index=len(pdata["columns"]) - 1
            )
        with col_btn:
            run_button = st.button("Start Training", type="primary")

        if run_button:
            with st.spinner(f"{model_name} model is being trained..."):
                query_params = {
                    "scale": scale,
                    "normalize": normalize,
                    "encode": encode,
                    "data_type": data_type,
                    "ylabel": ylabel,
                    "test_size": test_size,
                    "train_size": train_size,
                    "shuffle": shuffle,
                    "random_state": random_state,
                    "model_type": model_type,
                    "model_name": model_name,
                    **params
                }
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                try:
                    response = requests.post(
                        f"{API_URL}/one-click/",
                        params=query_params,
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