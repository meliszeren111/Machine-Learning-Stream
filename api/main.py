from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from process.data_loading.load_csv import *
from process.data_loading.load_tsv import *
from process.data_loading.load_xlsx import *
from process.data_splitting.split_data import *
from process.data_preprocessing.preprocess_data import *
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
from enum import Enum
from process.get_data.get_data import *
from process.evaluation.evaluate import evaluate_model
from process.model_selection.select_model import select_model
from process.prediction.predict_data import *
from process.training.train_model import train_model

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
            "rows": df.size,
            "columns": list(df.columns),
            "preview": df.head(3).to_dict(orient="records")
        }

@preprocess_router.post("/")
async def preprocess(df, scale=True, normalize=True, encode=True, data_type="nonordinal"):
        return preprocess_data(df, scale, normalize, encode, data_type=data_type) # data_type: ordinal, nonordinal, image

@split_router.post("/")
async def split(df, huggingface_dataset, data_type="sklearn", ylabel="target", test_size=0.2, train_size=0.8, split="train", left_size=0.2, right_size=0.8, shuffle=True, random_state=42,):
    X_train, X_test, y_train, y_test = split_data(df = df, huggingface_dataset = huggingface_dataset, data_type = data_type, ylabel = ylabel, test_size = test_size, train_size = train_size, split = split, left_size = left_size, right_size = right_size, shuffle = shuffle, random_state = random_state)
    return X_train, X_test, y_train, y_test

@select_router.post("/")
async def select(
    model_type: str,
    model_name: str,
    # Lineer model argümanları
    alpha: float = 1.0,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    dual: bool = True,
    fit_intercept: bool = True,
    max_iter: int = 100,
    selection: str = "random",
    # Non-linear / Regresyon argümanları
    degree: int = 3,
    kernel: str = "rbf",
    max_depth: int = 5,
    neighbors: int = 5,
    n_estimators: int = 100,
    # Sınıflandırma argümanları
    weights: str = "uniform",
    algorithm: str = "auto",
    criterion: str = "entropy",
    n_neighbors: int = 10,
    max_features=4,
    priors=None,
    loss: str = "log_loss",
    learning_rate: float = 0.05,
    # Kümeleme argümanları
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    linkage: str = "ward",
    bandwidth=None,
    damping: float = 0.5,
    # Boyut indirgeme argümanları
    n_components: int = 2,
    gamma=None,
    perplexity: float = 30.0,
    init: str = "pca"
):
    return select_model(
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
        init=init
    )
@train_router.post("/")
async def train(model, X_train, y_train):
    model = train_model(model, X_train, y_train)
    return model

@predict_router.post("/")
async def predict(model, X_test):
    predictions = predict_data(model, X_test)
    return predictions

@evaluate_router.post("/")
async def evaluate(type, X, labels_pred, y_test, y_pred, X_original, X_embedded, y=None, n_neighbors=5):
    return evaluate_model(type, X, labels_pred, y_test, y_pred, X_original, X_embedded, y, n_neighbors)


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
        ylabel: str = "target",
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
        criterion: str = "entropy",
        n_neighbors: int = 10,
        max_features=4,
        priors=None,
        loss: str = "log_loss",
        learning_rate: float = 0.05,
        n_clusters: int = 3,
        eps: float = 0.5,
        min_samples: int = 5,
        linkage: str = "ward",
        bandwidth=None,
        damping: float = 0.5,
        n_components: int = 2,
        gamma=None,
        perplexity: float = 30.0,
        init: str = "pca"
):
    filename = file.filename
    filename_lower = filename.lower()
    contents = await file.read()

    if filename_lower.endswith('.csv'):
        df = load_csv_file(io.BytesIO(contents))
    elif filename_lower.endswith('.tsv'):
        df = load_tsv_file(io.BytesIO(contents))
    elif filename_lower.endswith(('.xlsx', '.xls')):
        df = load_xlsx_file(io.BytesIO(contents))
    elif filename_lower.endswith('.txt'):
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python')
    else:
        raise HTTPException(status_code=400, detail="File format not supported.")

    dt_val = data_type.value if hasattr(data_type, "value") else data_type
    df = preprocess_data(df, scale, normalize, encode, data_type=dt_val)

    X_train, X_test, y_train, y_test = split_data(
        df=df, data_type=dt_val,
        ylabel=ylabel, test_size=test_size, train_size=train_size, split_by=split_by,
        left_size=left_size, right_size=right_size, shuffle=shuffle,
        random_state=random_state
    )

    model_type = model_type.value if hasattr(model_type, "value") else model_type
    model_name = model_name.value if hasattr(model_name, "value") else model_name

    model, model_desc = select_model(
        model_type=model_type,
        model_name=model_name,
        alpha=alpha, C=C, l1_ratio=l1_ratio, dual=dual,
        fit_intercept=fit_intercept, max_iter=max_iter, selection=selection,
        degree=degree, kernel=kernel, max_depth=max_depth, neighbors=neighbors,
        n_estimators=n_estimators, weights=weights, algorithm=algorithm,
        criterion=criterion, n_neighbors=n_neighbors, max_features=max_features,
        priors=priors, loss=loss, learning_rate=learning_rate,
        n_clusters=n_clusters, eps=eps, min_samples=min_samples, linkage=linkage,
        bandwidth=bandwidth, damping=damping, n_components=n_components,
        gamma=gamma, perplexity=perplexity, init=init
    )

    trained_model = train_model(model, X_train, y_train)
    predictions = predict_data(trained_model, X_test)

    evaluation_results = evaluate_model(
        type=model_type,
        X=X_test,
        labels_pred=None,
        y_test=y_test,
        y_pred=predictions,
        X_original=None,
        X_embedded=None,
        y=y_test,
        n_neighbors=n_neighbors
    )

    return {
        "status": "Success",
        "filename": filename,
        "model_info": model_desc,
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions,
        "evaluation": evaluation_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)