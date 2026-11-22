print("[BOOT] 1. api/main.py started", flush=True)
import gc
import os
from enum import Enum
import pickle
import tempfile
import uuid
from typing import Any, Dict, Optional
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
select_router = APIRouter(prefix="/select-model", tags=["Select"])
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


import os
import pickle
import tempfile
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Body, FastAPI, File, HTTPException, UploadFile


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


TEMP_DIR = os.path.join(tempfile.gettempdir(), "ml_pipeline_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

LATEST_STATE: Dict[str, Optional[str]] = {
    "raw_temp_id": None,
    "preprocessed_temp_id": None,
    "active_temp_id": None,
    "split_id": None,
    "model_id": None,
    "predictions_id": None,
}

MODEL_STORE: Dict[str, Any] = {}


def _save_temp(payload: Any, existing_id: Optional[str] = None) -> str:
    temp_id = existing_id or str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{temp_id}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return temp_id


def _load_temp(temp_id: Optional[str]) -> Any:
    if not temp_id:
        return None
    file_path = os.path.join(TEMP_DIR, f"{temp_id}.pkl")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Temporary data '{temp_id}' not found.")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _resolve_available_temp_id(preferred_order: list[str]) -> Optional[str]:
    for key in preferred_order:
        candidate_id = LATEST_STATE.get(key)
        if candidate_id and os.path.exists(os.path.join(TEMP_DIR, f"{candidate_id}.pkl")):
            return candidate_id
    return None


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


@app.get("/")
def read_root():
    return {"message": "ML Implementation API is running successfully."}


@upload_router.post("/")
async def upload(file: UploadFile = File(...)):
    try:
        filename_lower = file.filename.lower()
        contents = await file.read()
        df = get_data_from_file_binary(contents, filename_lower)
        df = _ensure_df(df)
        temp_id = _save_temp(df)

        LATEST_STATE["raw_temp_id"] = temp_id
        LATEST_STATE["preprocessed_temp_id"] = None
        LATEST_STATE["active_temp_id"] = temp_id
        LATEST_STATE["split_id"] = None
        LATEST_STATE["predictions_id"] = None

        return {
            "temp_id": temp_id,
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "preview": df.head(3).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload error: {type(exc).__name__}: {exc}")


@preprocess_router.post("/")
async def preprocess(
    temp_id: Optional[str] = None,
    df: Any = Body(None),
    scale: bool = True,
    normalize: bool = False,
    encode: bool = True,
    data_type: str = "ordinal",
):
    try:
        resolved_temp_id = temp_id or _resolve_available_temp_id(["raw_temp_id", "active_temp_id"])
        source_data = _load_temp(resolved_temp_id) if resolved_temp_id else df
        payload_df = (
            source_data.get("df")
            if isinstance(source_data, dict) and "df" in source_data and len(source_data) == 1
            else source_data
        )
        parsed_df = _ensure_df(payload_df)
        if parsed_df is None:
            raise HTTPException(status_code=400, detail="No available dataset found for preprocessing.")

        dt_val = data_type.value if hasattr(data_type, "value") else data_type
        result = preprocess_data(
            parsed_df,
            scale=scale,
            normalize=normalize,
            encode=encode,
            data_type=dt_val,
        )
        processed_df = _ensure_df(result)
        saved_id = _save_temp(processed_df)

        LATEST_STATE["preprocessed_temp_id"] = saved_id
        LATEST_STATE["active_temp_id"] = saved_id
        LATEST_STATE["split_id"] = None
        LATEST_STATE["predictions_id"] = None

        return {
            "temp_id": saved_id,
            "source_temp_id": resolved_temp_id,
            "rows": int(processed_df.shape[0]),
            "columns": list(processed_df.columns),
            "preview": processed_df.head(3).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preprocess error: {type(exc).__name__}: {exc}")


@split_router.post("/")
async def split(
    temp_id: Optional[str] = None,
    df: Any = Body(None),
    huggingface_dataset: Optional[str] = None,
    data_type: str = "ordinal",
    ylabel: Optional[str] = None,
    test_size: float = 0.2,
    train_size: float = 0.8,
    split_by: str = "train",
    left_size: float = 0.8,
    right_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
):
    try:
        resolved_temp_id = temp_id or _resolve_available_temp_id([
            "preprocessed_temp_id",
            "active_temp_id",
            "raw_temp_id",
        ])
        source_data = _load_temp(resolved_temp_id) if resolved_temp_id else df
        payload_df = (
            source_data.get("df")
            if isinstance(source_data, dict) and "df" in source_data and len(source_data) == 1
            else source_data
        )
        parsed_df = _ensure_df(payload_df)

        dt_val = data_type.value if hasattr(data_type, "value") else data_type
        X_train, X_test, y_train, y_test = split_data(
            df=parsed_df,
            huggingface_dataset=huggingface_dataset,
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

        split_id = _save_temp({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        })
        LATEST_STATE["split_id"] = split_id
        LATEST_STATE["predictions_id"] = None

        return {
            "split_id": split_id,
            "source_temp_id": resolved_temp_id,
            "X_train_shape": getattr(X_train, "shape", None),
            "X_test_shape": getattr(X_test, "shape", None),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Split error: {type(exc).__name__}: {exc}")


@select_router.post("/")
async def select(
    model_type: str = "regression",
    model_name: str = "RandomForestRegressor",
    alpha: float = 1.0,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    dual: bool = True,
    fit_intercept: bool = True,
    max_iter: int = 100,
    selection: str = "random",
    degree: int = 3,
    kernel: str = "rbf",
    max_depth: int = 5,
    neighbors: int = 5,
    n_estimators: int = 100,
    weights: str = "uniform",
    algorithm: str = "auto",
    criterion: str = "squared_error",
    n_neighbors: int = 10,
    max_features: Any = "sqrt",
    priors: Any = None,
    loss: str = "squared_error",
    learning_rate: float = 0.05,
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    linkage: str = "ward",
    bandwidth: Optional[float] = None,
    damping: float = 0.5,
    n_components: int = 2,
    gamma: Any = "scale",
    perplexity: float = 30.0,
    init: str = "pca",
):
    try:
        model_type_val = model_type.value if hasattr(model_type, "value") else model_type
        model_name_val = model_name.value if hasattr(model_name, "value") else model_name

        estimator, model_desc = select_model(
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
        model_id = str(uuid.uuid4())
        MODEL_STORE[model_id] = estimator
        LATEST_STATE["model_id"] = model_id
        return {
            "model_id": model_id,
            "model_name": model_name_val,
            "model_type": model_type_val,
            "model_info": model_desc,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Select error: {type(exc).__name__}: {exc}")


@train_router.post("/")
async def train(
    model_id: Optional[str] = None,
    split_id: Optional[str] = None,
):
    try:
        resolved_split_id = (
            split_id
            or LATEST_STATE.get("split_id")
        )
        if not resolved_split_id:
            raise HTTPException(
                status_code=400,
                detail="No split dataset found. Pass split_id / split_temp_id or run /split-data/ first.",
            )

        split_bundle = _load_temp(resolved_split_id)
        if not isinstance(split_bundle, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid split data in '{resolved_split_id}'.",
            )

        X_train = split_bundle.get("X_train")
        y_train = split_bundle.get("y_train")

        resolved_model_id = model_id or LATEST_STATE.get("model_id")
        if not resolved_model_id or resolved_model_id not in MODEL_STORE:
            raise HTTPException(
                status_code=400,
                detail="No model found. Call /select-data/ first or pass a valid model_id.",
            )

        target_model = MODEL_STORE[resolved_model_id]
        trained = train_model(target_model, X_train, y_train)

        MODEL_STORE[resolved_model_id] = trained
        LATEST_STATE["model_id"] = resolved_model_id
        LATEST_STATE["split_id"] = resolved_split_id

        return {
            "model_id": resolved_model_id,
            "split_id": resolved_split_id,
            "status": "trained",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Train error: {type(exc).__name__}: {exc}")

@predict_router.post("/")
async def predict(
    model_id: Optional[str] = None,
    split_id: Optional[str] = None,
):
    try:
        resolved_split_id = (
            split_id
            or LATEST_STATE.get("split_id")
        )
        if not resolved_split_id:
            raise HTTPException(
                status_code=400,
                detail="No split dataset found. Pass split_id / split_temp_id or run /split-data/ first.",
            )

        split_bundle = _load_temp(resolved_split_id)
        if not isinstance(split_bundle, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid split data in '{resolved_split_id}'.",
            )

        X_test = split_bundle.get("X_test")
        if X_test is None:
            raise HTTPException(
                status_code=400,
                detail=f"X_test not found in split '{resolved_split_id}'.",
            )

        resolved_model_id = model_id or LATEST_STATE.get("model_id")
        if not resolved_model_id or resolved_model_id not in MODEL_STORE:
            raise HTTPException(
                status_code=400,
                detail="No trained model found for prediction. Run /train-model/ first or pass a valid model_id.",
            )

        target_model = MODEL_STORE[resolved_model_id]
        predictions = predict_data(target_model, X_test)
        pred_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions
        predictions_id = _save_temp(pred_list)

        LATEST_STATE["predictions_id"] = predictions_id
        LATEST_STATE["model_id"] = resolved_model_id
        LATEST_STATE["split_id"] = resolved_split_id

        return {
            "predictions_id": predictions_id,
            "model_id": resolved_model_id,
            "split_id": resolved_split_id,
            "predictions": pred_list,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Predict error: {type(exc).__name__}: {exc}")

@evaluate_router.post("/")
async def evaluate(
    type: str = "regression",
    split_id: Optional[str] = None,
    split_temp_id: Optional[str] = None,
    predictions_id: Optional[str] = None,
    n_neighbors: int = 10,
):
    try:
        resolved_split_id = (
            split_id
            or split_temp_id
            or LATEST_STATE.get("split_id")
        )
        split_bundle = _load_temp(resolved_split_id) if resolved_split_id else {}
        if not isinstance(split_bundle, dict):
            split_bundle = {}

        X = split_bundle.get("X_test")
        y_test = split_bundle.get("y_test")
        y = split_bundle.get("y_test")
        labels_pred = None
        X_original = None
        X_embedded = None

        resolved_predictions_id = predictions_id or LATEST_STATE.get("predictions_id")
        y_pred = _load_temp(resolved_predictions_id) if resolved_predictions_id else None

        eval_type = type.value if hasattr(type, "value") else type
        if y_pred is not None and eval_type == "clustering":
            labels_pred = y_pred

        return evaluate_model(
            eval_type,
            X,
            labels_pred,
            y_test,
            y_pred,
            X_original,
            X_embedded,
            y,
            n_neighbors,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluate error: {type(exc).__name__}: {exc}")



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
    scale: bool = True,
    normalize: bool = False,
    encode: bool = True,
    data_type: DataTypeEnum = DataTypeEnum.ordinal,
    ylabel: Optional[str] = None,
    test_size: float = 0.2,
    train_size: float = 0.8,
    split_by: str = "train",
    left_size: float = 0.8,
    right_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    model_type: ModelTypeEnum = ModelTypeEnum.regression,
    model_name: ModelNameEnum = ModelNameEnum.random_forest_reg,
    alpha: float = 1.0,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    dual: bool = True,
    fit_intercept: bool = True,
    max_iter: int = 100,
    selection: str = "random",
    degree: int = 3,
    kernel: str = "rbf",
    max_depth: int = 5,
    neighbors: int = 5,
    n_estimators: int = 100,
    weights: str = "uniform",
    algorithm: str = "auto",
    criterion: str = "squared_error",
    n_neighbors: int = 10,
    max_features: Any = "sqrt",
    priors: Any = None,
    loss: str = "squared_error",
    learning_rate: float = 0.05,
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    linkage: str = "ward",
    bandwidth: Optional[float] = None,
    damping: float = 0.5,
    n_components: int = 2,
    gamma: Any = "scale",
    perplexity: float = 30.0,
    init: str = "pca",
):
    try:
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

        if hasattr(df, "toarray"):
            df = pd.DataFrame(df.toarray())
        elif not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)