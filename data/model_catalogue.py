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
