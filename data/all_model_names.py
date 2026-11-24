regression_and_classification_models = [
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "AdaBoostRegressor",
    "AdaBoostClassifier",
    "BaggingRegressor",
    "BaggingClassifier",
    "ExtraTreesRegressor",
    "ExtraTreesClassifier",
    "GaussianProcessRegressor",
    "RBFSampler",
    "LinearRegression",
    "LogisticRegression",
    "Lasso",
    "Ridge",
    "LassoCV",
    "RidgeCV",
    "GaussianNB",
    "KNeighborsRegressor",
    "KNeighborsClassifier",
    "MLPRegressor",
    "SVR",
    "SVC",
    "PolynomialFeatures",
    "SplineTransformer",
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMRegressor",
    "LGBMClassifier",
    "CatBoostRegressor",
    "CatBoostClassifier",
    "PolynomialRegression",
    "SupportVectorRegressor",
    "MultiLayerPerceptron",
    "RadialBasisFunction",
    "KNN",
    "NaiveBayes",
    "SupportVectorClassifier",
]

dimensionality_reduction_models = [
    "PCA",
    "TruncatedSVD",
    "KernelPCA",
    "FastICA",
    "FactorAnalysis",
    "NMF",
    "DictionaryLearning",
    "LDA",  # LinearDiscriminantAnalysis
    "QDA",
    "TSNE",
    "UMAP",
]

clustering_models = [
    "KMeans",
    "DBSCAN",
    "AgglomerativeClustering",
    "SpectralClustering",
    "MeanShift",
    "AffinityPropagation",
    "OPTICS",
    "Birch",
    "HDBSCAN",
    "GaussianMixture",
]

all_models = (
    regression_and_classification_models
    + dimensionality_reduction_models
    + clustering_models
)