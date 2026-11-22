

print("[BOOT] 1. api/main.py started", flush=True)
import gc
import io
import os
from enum import Enum
from typing import Any, Optional

import pandas as pd
print("[BOOT] 2. pandas imported", flush=True)
from fastapi import APIRouter, Body, FastAPI, File, HTTPException, UploadFile
print("[BOOT] 3. fastapi imported", flush=True)

from process.data_loading.load_csv import *
print("[BOOT] 4. load_csv imported", flush=True)
from process.data_loading.load_tsv import *
print("[BOOT] 5. load_tsv imported", flush=True)
from process.data_loading.load_xlsx import *
print("[BOOT] 6. load_xlsx imported", flush=True)
from process.data_splitting.split_data2 import *
print("[BOOT] 7. split_data2 imported", flush=True)
from process.data_preprocessing.preprocess_data import *
print("[BOOT] 8. preprocess_data imported", flush=True)
from process.get_data.get_data2 import *
print("[BOOT] 9. get_data2 imported", flush=True)
from process.evaluation.evaluate import evaluate_model
print("[BOOT] 10. evaluate imported", flush=True)
from process.model_selection.select_model2 import select_model
print("[BOOT] 11. select_model imported", flush=True)
from process.prediction.predict_data import *
print("[BOOT] 12. predict_data imported", flush=True)
from process.training.train_model import train_model
print("[BOOT] 13. train_model imported", flush=True)

app = FastAPI(title="ML Implementation API", version="1.0")
upload_router = APIRouter(prefix="/upload-data", tags=["Upload"])
preprocess_router = APIRouter(prefix="/preprocess-data", tags=["Preprocess"])
split_router = APIRouter(prefix="/split-data", tags=["Split"])
select_router = APIRouter(prefix="/select-data", tags=["Select"])
train_router = APIRouter(prefix="/train-model", tags=["Train"])
predict_router = APIRouter(prefix="/predict-data", tags=["Predict"])
evaluate_router = APIRouter(prefix="/evaluate-model", tags=["Evaluate"])
one_click_router = APIRouter(prefix="/one-click", tags=["OneClick"])

app.include_router(upload_router)
app.include_router(preprocess_router)
app.include_router(split_router)
app.include_router(select_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(evaluate_router)
app.include_router(one_click_router)


def _to_df(data: Any) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "df" in data and data["df"] is not None:
            return _to_df(data["df"])
        return pd.DataFrame([data])
    return pd.DataFrame(data)


@app.get("/")
def read_root():
    return {"message": "ML Implementation API is running successfully."}


@upload_router.post("/")
async def upload(file: UploadFile = File(...)):
    filename_lower = file.filename.lower()
    contents = await file.read()
    df = get_data_from_file_binary(contents, filename_lower)
    return {
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "preview": df.head(3).to_dict(orient="records"),
    }


@preprocess_router.post("/")
async def preprocess(
    df: Any = Body(None),
    scale: Optional[bool] = None,
    normalize: Optional[bool] = None,
    encode: Optional[bool] = None,
    data_type: Optional[str] = None,
):
    payload_df = df.get("df") if isinstance(df, dict) and "df" in df and len(df) == 1 else df
    parsed_df = _to_df(payload_df)
    if parsed_df is None:
        raise HTTPException(status_code=400, detail="df is required for preprocessing.")

    result = preprocess_data(
        parsed_df,
        scale=scale,
        normalize=normalize,
        encode=encode,
        data_type=data_type,
    )
    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    return result


@split_router.post("/")
async def split(
    df: Any = Body(None),
    huggingface_dataset: Optional[str] = None,
    data_type: Optional[str] = None,
    ylabel: Optional[str] = None,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    split: Optional[str] = None,
    left_size: Optional[float] = None,
    right_size: Optional[float] = None,
    shuffle: Optional[bool] = None,
    random_state: Optional[int] = None,
):
    payload_df = df.get("df") if isinstance(df, dict) and "df" in df and len(df) == 1 else df
    parsed_df = _to_df(payload_df)

    X_train, X_test, y_train, y_test = split_data(
        df=parsed_df,
        huggingface_dataset=huggingface_dataset,
        data_type=data_type,
        ylabel=ylabel,
        test_size=test_size,
        train_size=train_size,
        left_size=left_size,
        right_size=right_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    def _serialize(obj: Any):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    return {
        "X_train": _serialize(X_train),
        "X_test": _serialize(X_test),
        "y_train": _serialize(y_train),
        "y_test": _serialize(y_test),
    }
import uuid
from typing import Any, Dict, Optional

MODEL_STORE: Dict[str, Any] = {}


@select_router.post("/")
async def select(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    alpha: Optional[float] = None,
    C: Optional[float] = None,
    l1_ratio: Optional[float] = None,
    dual: Optional[bool] = None,
    fit_intercept: Optional[bool] = None,
    max_iter: Optional[int] = None,
    selection: Optional[str] = None,
    degree: Optional[int] = None,
    kernel: Optional[str] = None,
    max_depth: Optional[int] = None,
    neighbors: Optional[int] = None,
    n_estimators: Optional[int] = None,
    weights: Optional[str] = None,
    algorithm: Optional[str] = None,
    criterion: Optional[str] = None,
    n_neighbors: Optional[int] = None,
    max_features: Any = None,
    priors: Any = None,
    loss: Optional[str] = None,
    learning_rate: Optional[float] = None,
    n_clusters: Optional[int] = None,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    linkage: Optional[str] = None,
    bandwidth: Optional[float] = None,
    damping: Optional[float] = None,
    n_components: Optional[int] = None,
    gamma: Any = None,
    perplexity: Optional[float] = None,
    init: Optional[str] = None,
):
    estimator, model_desc = select_model(
        model_type=model_type,
        model_name=model_name,
        alpha=alpha,
        C=C,
        l1_ratio=l1_ratio,
        dual=dual,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        selection=selection,
        degree=degree,
        kernel=kernel,
        max_depth=max_depth,
        neighbors=neighbors,
        n_estimators=n_estimators,
        weights=weights,
        algorithm=algorithm,
        criterion=criterion,
        n_neighbors=n_neighbors,
        max_features=max_features,
        priors=priors,
        loss=loss,
        learning_rate=learning_rate,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
        linkage=linkage,
        bandwidth=bandwidth,
        damping=damping,
        n_components=n_components,
        gamma=gamma,
        perplexity=perplexity,
        init=init,
    )
    model_id = str(uuid.uuid4())
    MODEL_STORE[model_id] = estimator
    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "model_info": model_desc,
    }


@train_router.post("/")
async def train(
    model: Any = Body(None),
    X_train: Any = Body(None),
    y_train: Any = Body(None),
):
    target_model = model
    if isinstance(model, dict) and "model_id" in model:
        target_model = MODEL_STORE.get(model["model_id"], model)
    elif isinstance(model, str) and model in MODEL_STORE:
        target_model = MODEL_STORE[model]

    trained = train_model(target_model, X_train, y_train)
    if isinstance(model, dict) and "model_id" in model:
        MODEL_STORE[model["model_id"]] = trained
        return {"model_id": model["model_id"], "status": "trained"}
    elif isinstance(model, str) and model in MODEL_STORE:
        MODEL_STORE[model] = trained
        return {"model_id": model, "status": "trained"}
    return trained


@predict_router.post("/")
async def predict(
    model: Any = Body(None),
    X_test: Any = Body(None),
):
    predictions = predict_data(model, X_test)
    return predictions.tolist() if hasattr(predictions, "tolist") else predictions


@evaluate_router.post("/")
async def evaluate(
    type: Optional[str] = None,
    X: Any = Body(None),
    labels_pred: Any = Body(None),
    y_test: Any = Body(None),
    y_pred: Any = Body(None),
    X_original: Any = Body(None),
    X_embedded: Any = Body(None),
    y: Any = Body(None),
    n_neighbors: Optional[int] = None,
):
    return evaluate_model(
        type,
        X,
        labels_pred,
        y_test,
        y_pred,
        X_original,
        X_embedded,
        y,
        n_neighbors,
    )


class DataTypeEnum(str, Enum):
    ordinal = "ordinal"
    nonordinal = "nonordinal"
    image = "image"


class ModelTypeEnum(str, Enum):
    regression = "regression"
    classification = "classification"
    clustering = "clustering"
    dimensionality_reduction = "dimensionality_reduction"


class ModelNameEnum(str, Enum):
    random_forest_reg = "RandomForestRegressor"
    gradient_boosting_reg = "GradientBoostingRegressor"
    random_forest_clf = "RandomForestClassifier"
    gradient_boosting_clf = "GradientBoostingClassifier"
    hist_gradient_boosting_reg = "HistGradientBoostingRegressor"
    hist_gradient_boosting_clf = "HistGradientBoostingClassifier"
    ada_boost_reg = "AdaBoostRegressor"
    ada_boost_clf = "AdaBoostClassifier"
    bagging_reg = "BaggingRegressor"
    bagging_clf = "BaggingClassifier"
    extra_trees_reg = "ExtraTreesRegressor"
    extra_trees_clf = "ExtraTreesClassifier"
    gaussian_process_reg = "GaussianProcessRegressor"
    rbf_sampler = "RBFSampler"
    linear_regression = "LinearRegression"
    logistic_regression = "LogisticRegression"
    lasso = "Lasso"
    ridge = "Ridge"
    lasso_cv = "LassoCV"
    ridge_cv = "RidgeCV"
    gaussian_nb = "GaussianNB"
    k_neighbors_reg = "KNeighborsRegressor"
    k_neighbors_clf = "KNeighborsClassifier"
    mlp_reg = "MLPRegressor"
    svr = "SVR"
    svc = "SVC"
    polynomial_features = "PolynomialFeatures"
    spline_transformer = "SplineTransformer"
    decision_tree_reg = "DecisionTreeRegressor"
    decision_tree_clf = "DecisionTreeClassifier"
    xgb_clf = "XGBClassifier"
    xgb_reg = "XGBRegressor"
    lgbm_reg = "LGBMRegressor"
    lgbm_clf = "LGBMClassifier"
    cat_boost_reg = "CatBoostRegressor"
    cat_boost_clf = "CatBoostClassifier"
    polynomial_regression = "PolynomialRegression"
    support_vector_regressor = "SupportVectorRegressor"
    multi_layer_perceptron = "MultiLayerPerceptron"
    radial_basis_function = "RadialBasisFunction"
    knn = "KNN"
    naive_bayes = "NaiveBayes"
    support_vector_classifier = "SupportVectorClassifier"
    pca = "PCA"
    truncated_svd = "TruncatedSVD"
    kernel_pca = "KernelPCA"
    fast_ica = "FastICA"
    factor_analysis = "FactorAnalysis"
    nmf = "NMF"
    dictionary_learning = "DictionaryLearning"
    lda = "LDA"
    qda = "QDA"
    tsne = "TSNE"
    umap = "UMAP"
    kmeans = "KMeans"
    dbscan = "DBSCAN"
    agglomerative_clustering = "AgglomerativeClustering"
    spectral_clustering = "SpectralClustering"
    mean_shift = "MeanShift"
    affinity_propagation = "AffinityPropagation"
    optics = "OPTICS"
    birch = "Birch"
    hdbscan = "HDBSCAN"
    gaussian_mixture = "GaussianMixture"


@one_click_router.post("/")
async def one_click(
    file: UploadFile = File(...),
    scale: Optional[bool] = None,
    normalize: Optional[bool] = None,
    encode: Optional[bool] = None,
    data_type: Optional[DataTypeEnum] = None,
    ylabel: Optional[str] = None,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    split_by: Optional[str] = None,
    left_size: Optional[float] = None,
    right_size: Optional[float] = None,
    shuffle: Optional[bool] = None,
    random_state: Optional[int] = None,
    model_type: Optional[ModelTypeEnum] = None,
    model_name: Optional[ModelNameEnum] = None,
    alpha: Optional[float] = None,
    C: Optional[float] = None,
    l1_ratio: Optional[float] = None,
    dual: Optional[bool] = None,
    fit_intercept: Optional[bool] = None,
    max_iter: Optional[int] = None,
    selection: Optional[str] = None,
    degree: Optional[int] = None,
    kernel: Optional[str] = None,
    max_depth: Optional[int] = None,
    neighbors: Optional[int] = None,
    n_estimators: Optional[int] = None,
    weights: Optional[str] = None,
    algorithm: Optional[str] = None,
    criterion: Optional[str] = None,
    n_neighbors: Optional[int] = None,
    max_features: Any = None,
    priors: Any = None,
    loss: Optional[str] = None,
    learning_rate: Optional[float] = None,
    n_clusters: Optional[int] = None,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    linkage: Optional[str] = None,
    bandwidth: Optional[float] = None,
    damping: Optional[float] = None,
    n_components: Optional[int] = None,
    gamma: Any = None,
    perplexity: Optional[float] = None,
    init: Optional[str] = None,
):
    filename = file.filename
    filename_lower = filename.lower()
    contents = await file.read()

    if filename_lower.endswith(".csv"):
        df = load_csv_file(io.BytesIO(contents))
    elif filename_lower.endswith(".tsv"):
        df = load_tsv_file(io.BytesIO(contents))
    elif filename_lower.endswith((".xlsx", ".xls")):
        df = load_xlsx_file(io.BytesIO(contents))
    elif filename_lower.endswith(".txt"):
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
    else:
        raise HTTPException(status_code=400, detail="File format not supported.")

    dt_val = data_type.value if hasattr(data_type, "value") else data_type
    df = preprocess_data(df, scale, normalize, encode, data_type=dt_val)

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        data_type=dt_val,
        ylabel=ylabel,
        test_size=test_size,
        train_size=train_size,
        split_by=split_by,
        left_size=left_size,
        right_size=right_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    model_type_val = model_type.value if hasattr(model_type, "value") else model_type
    model_name_val = model_name.value if hasattr(model_name, "value") else model_name

    model, model_desc = select_model(
        model_type=model_type_val,
        model_name=model_name_val,
        alpha=alpha,
        C=C,
        l1_ratio=l1_ratio,
        dual=dual,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        selection=selection,
        degree=degree,
        kernel=kernel,
        max_depth=max_depth,
        neighbors=neighbors,
        n_estimators=n_estimators,
        weights=weights,
        algorithm=algorithm,
        criterion=criterion,
        n_neighbors=n_neighbors,
        max_features=max_features,
        priors=priors,
        loss=loss,
        learning_rate=learning_rate,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
        linkage=linkage,
        bandwidth=bandwidth,
        damping=damping,
        n_components=n_components,
        gamma=gamma,
        perplexity=perplexity,
        init=init,
    )

    trained_model = train_model(model, X_train, y_train)
    predictions = predict_data(trained_model, X_test)

    evaluation_results = evaluate_model(
        type=model_type_val,
        X=X_test,
        labels_pred=None,
        y_test=y_test,
        y_pred=predictions,
        X_original=None,
        X_embedded=None,
        y=y_test,
        n_neighbors=n_neighbors,
    )
    result = {
        "status": "Success",
        "filename": filename,
        "model_info": model_desc,
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions,
        "evaluation": evaluation_results,
    }
    for var_name in (
        "contents",
        "df",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "model",
        "trained_model",
        "predictions",
    ):
        if var_name in locals():
            del locals()[var_name]
    gc.collect()
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
