from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier,  \
    AdaBoostRegressor , AdaBoostClassifier, BaggingRegressor, BaggingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.decomposition import (
    PCA, TruncatedSVD, KernelPCA, FastICA, FactorAnalysis, NMF, DictionaryLearning
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis
from sklearn.manifold import TSNE
from umap import UMAP


from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    MeanShift, AffinityPropagation, OPTICS, Birch, HDBSCAN
)
from sklearn.mixture import GaussianMixture


def select_linear_regression_model(model_name, alpha=1.0, C=1.0, l1_ratio=0.5, dual=True, fit_intercept=True,
                                   max_iter=100, selection="random"):
    descriptions = {
        'LogisticRegression': (
            "A fundamental linear model used for binary or multi-class classification tasks. "
            "It applies the logistic (sigmoid) function to map linear combinations of input features into a probability "
            "between 0 and 1. It is highly interpretable, fast to train, and works exceptionally well when classes "
            "are linearly separable."
        ),
        'Lasso': (
            "A linear regression model that incorporates L1 regularization (Absolute Value Penalty). "
            "By adding the sum of the absolute values of the coefficients to the loss function, Lasso forces some "
            "coefficients to become **exactly zero**, thereby performing **automatic feature selection** and "
            "preventing overfitting in high-dimensional datasets."
        ),
        'Ridge': (
            "A linear regression model that utilizes L2 regularization (Squared Magnitude Penalty). "
            "It penalizes large coefficients by adding the squared values of weights to the loss function. Unlike Lasso, "
            "Ridge shrinks coefficients close to zero without making them entirely zero, preserving all features while "
            "reducing multicollinearity and stabilizing predictions."
        ),
        'SVR': (
            "Support Vector Regression; a powerful non-linear or linear regression model adapted from Support Vector Machines. "
            "Instead of minimizing squared errors, SVR tries to fit the error within a predefined threshold margin (epsilon). "
            "It relies on a subset of training data (support vectors) to define the decision boundary, making it robust against outliers."
        ),
        'LassoCV': (
            "A specialized Lasso regression model integrated with K-Fold Cross-Validation. "
            "It automatically searches across a range of regularization strengths (alpha parameters), evaluates performance "
            "on validation folds, and selects the optimal alpha value that minimizes out-of-sample prediction error."
        ),
        'RidgeCV': (
            "A Ridge regression model equipped with built-in efficient Cross-Validation (Generalized Cross-Validation or K-Fold). "
            "It automatically tests multiple alpha (L2 penalty) values and instantly returns the model trained with the best "
            "hyperparameter, saving you manual grid-search steps."
        ),
        'LinearRegression': (
            "The classic, foundational linear regression model that minimizes the Residual Sum of Squares using the "
            "Ordinary Least Squares (OLS) method. It assumes a direct, linear relationship between the independent features "
            "and the continuous target variable. It is fast and interpretable but vulnerable to multicollinearity and outliers."
        )
    }
    if model_name == 'LogisticRegression':
        model = LogisticRegression(C=C, l1_ratio=l1_ratio, dual=dual, fit_intercept=fit_intercept)
    elif model_name == 'Lasso':
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, selection=selection)
    elif model_name == 'Ridge':
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
    elif model_name == 'SVR':
        model = SVR(kernel="rbf", degree=3, C=C, max_iter=max_iter)
    elif model_name == 'LassoCV':
        model = LassoCV(fit_intercept=fit_intercept, max_iter=max_iter, selection=selection)
    elif model_name == 'RidgeCV':
        model = RidgeCV(fit_intercept=fit_intercept)
    elif model_name == "LinearRegression":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, model_name +": "+ descriptions.get(model_name, "Description not found.")


def select_nonlinear_regression_model(model_name, degree=3, kernel="rbf", max_depth=5, neighbors=5, n_estimators=100):
    descriptions = {
        'PolynomialRegression': (
            "Transforms the original features into polynomial and interaction terms up to a specified degree "
            "(e.g., $x^2$, $x^3$, $x_1 x_2$). This allows a standard linear model to fit complex, curved, "
            "and non-linear relationships in the data, though it can risk overfitting if the degree is set too high."
        ),
        'DecisionTreeRegressor': (
            "A tree-based regression model that splits the dataset into hierarchical branches by evaluating feature "
            "thresholds that minimize variance. It creates intuitive rule-based partitions, can capture complex non-linear "
            "patterns easily, but is prone to overfitting if the maximum depth is left unrestricted."
        ),
        'RandomForestRegressor': (
            "A powerful ensemble learning model that constructs a forest of decision trees during training, each on a "
            "random bootstrap sample with random feature subsets. It averages their predictions to dramatically reduce "
            "variance, prevent overfitting, and handle noisy datasets effectively."
        ),
        'GradientBoostingRegressor': (
            "An iterative boosting ensemble that builds decision trees sequentially. Each new tree is trained specifically "
            "to predict and minimize the residual errors (gradients) left by the previous trees, leading to high predictive "
            "accuracy on complex datasets."
        ),
        'SupportVectorRegressor': (
            "A non-linear regression model adapted from Support Vector Machines. It projects data into higher-dimensional "
            "feature spaces using kernel functions (such as RBF) to find an optimal tube (epsilon-insensitive margin) "
            "that captures the majority of data points while penalizing predictions outside the margin."
        ),
        'KNeighborsRegressor': (
            "A non-parametric, distance-based lazy learning algorithm. To predict a target value, it identifies the "
            "'k' closest data points in the feature space using metrics like Euclidean distance and calculates their average. "
            "It requires no training phase but can become computationally expensive on large datasets."
        ),
        'GaussianProcessRegressor': (
            "A flexible, non-parametric probabilistic model based on Gaussian processes. Beyond providing point estimates "
            "for predictions, it models the full distribution, outputting both the predicted value and a measure of "
            "**uncertainty (variance)**, making it ideal for active learning and optimization tasks."
        ),
        'MultiLayerPerceptron': (
            "A feedforward artificial neural network regressor consisting of input, hidden, and output layers with non-linear "
            "activation functions. It is capable of learning highly complex, multidimensional, and non-linear patterns "
            "from large volumes of data through backpropagation."
        ),
        'SplineTransformer': (
            "A feature engineering preprocessing tool that generates piecewise polynomial (spline) bases with smoothness "
            "constraints at the knots. It allows linear models to capture highly flexible, smooth non-linear trends "
            "across different segments of the feature space."
        ),
        'RadialBasisFunction': (
            "A preprocessing transformer (RBFSampler) that uses random Fourier features to approximate kernel maps "
            "based on radial basis functions. It maps linear models into a non-linear kernel space efficiently, "
            "scaling well to large datasets."
        ),
        'XGBoostRegressor': (
            "Extreme Gradient Boosting; an elite, highly optimized gradient boosting library designed for speed, efficiency, "
            "and superior predictive performance. It includes advanced regularization (L1/L2) to control overfitting "
            "and handles sparse data and missing values natively."
        ),
        'HistGradientBoostingRegressor': (
            "A modern, ultra-fast gradient boosting tree model inspired by LightGBM. It bins continuous features into "
            "integer-valued histograms (e.g., 256 bins), significantly accelerating training times and enabling scalable "
            "performance on massive datasets with millions of rows."
        ),
        'AdaBoostRegressor': (
            "Adaptive Boosting; an iterative ensemble technique that trains a sequence of weak regressors (typically shallow trees). "
            "In each subsequent iteration, higher sample weights are assigned to instances that were predicted with larger errors, "
            "forcing the model to focus on hard-to-learn patterns."
        ),
        'BaggingRegressor': (
            "Bootstrap Aggregating; a robust ensemble meta-estimator that trains multiple base models independently in parallel "
            "on random subsets (with replacement) of the training set and aggregates their predictions (usually via averaging) "
            "to reduce variance and prevent overfitting."
        ),
        'ExtraTreesRegressor': (
            "Extremely Randomized Trees; an ensemble variation of Random Forest where decision trees are randomized even further. "
            "Instead of searching for the optimal split threshold, splits are chosen entirely at random for each feature, "
            "which speeds up training and often yields lower variance."
        ),
        'LGBMRegressor': (
            "LightGBM; a lightning-fast, highly efficient gradient boosting framework developed by Microsoft. Unlike traditional "
            "level-wise growth, it grows trees **leaf-wise** by choosing the leaf with max delta loss, achieving higher accuracy "
            "and drastically lower training times on large-scale data."
        ),
        'CatBoostRegressor': (
            "A specialized gradient boosting algorithm designed to handle **categorical features natively and automatically** "
            "without requiring extensive manual one-hot encoding. It uses ordered boosting to prevent target leakage, "
            "delivering exceptional out-of-the-box accuracy with minimal hyperparameter tuning."
        )
    }
    if model_name == "PolynomialRegression":
        model = PolynomialFeatures(degree=degree)
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(max_depth=max_depth)
    elif model_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(max_depth=max_depth)
    elif model_name == "SupportVectorRegressor":
        model = SVR(kernel=kernel, degree=degree)
    elif model_name == "KNeighborsRegressor":
        model = KNeighborsRegressor(n_neighbors=neighbors)
    elif model_name == "GaussianProcessRegressor":
        model = GaussianProcessRegressor(kernel=kernel)
    elif model_name == "MultiLayerPerceptron":
        model = MLPRegressor(hidden_layer_sizes=neighbors)
    elif model_name == "SplineTransformer":
        model = SplineTransformer(degree=degree)
    elif model_name in ["RadialBasisFunction", "RBFSampler"]:
        model = RBFSampler(n_components=neighbors)
    elif model_name == "XGBoostRegressor":
        model = XGBRegressor(n_estimators=n_estimators)
    elif model_name == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor()
    elif model_name == "AdaBoostRegressor":
        model = AdaBoostRegressor(n_estimators=n_estimators)
    elif model_name == "BaggingRegressor":
        model = BaggingRegressor(n_estimators=n_estimators)
    elif model_name == "ExtraTreesRegressor":
        model = ExtraTreesRegressor(n_estimators=n_estimators)
    elif model_name == "LGBMRegressor":
        model = LGBMRegressor(n_estimators=n_estimators, verbose=-1)
    elif model_name == "CatBoostRegressor":
        model = CatBoostRegressor(n_estimators=n_estimators, verbose=0)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, model_name +": "+ descriptions.get(model_name, "Description not found.")


def select_classification_model(model_name, C=1.0, l1_ratio=0.5, dual=True, weights="uniform", algorithm="auto",
                                fit_intercept=True, criterion="entropy", n_neighbors=10, max_depth=5, max_features=4,
                                priors=None, loss="log_loss", learning_rate=0.05, max_iter=100, n_estimators=100,
                                degree=3, kernel="rbf"):
    descriptions = {
        'LogisticRegression': (
            "A foundational linear classification model used for binary and multi-class problems. "
            "It applies the logistic (sigmoid) function to map linear combinations of input features into class probabilities "
            "between 0 and 1. It provides highly interpretable coefficient weights, trains quickly, and is ideal for linearly "
            "separable data."
        ),
        'KNN': (
            "K-Nearest Neighbors; a non-parametric, distance-based classification algorithm. "
            "To classify a new data point, it looks at the 'k' closest points in the feature space using metrics like "
            "Euclidean distance and assigns the class by majority vote. It requires no explicit training phase but can "
            "become slow when evaluating large feature spaces."
        ),
        'NaiveBayes': (
            "A fast, highly scalable probabilistic classifier based on Bayes' Theorem. "
            "It relies on the 'naive' assumption that all input features are conditionally independent given the class label. "
            "Despite this simplifying assumption, it performs exceptionally well on text classification, spam filtering, "
            "and high-dimensional sparse datasets."
        ),
        'DecisionTreeClassifier': (
            "A tree-based hierarchical classifier that recursively splits the dataset into distinct classes by evaluating "
            "feature threshold rules that maximize information gain or Gini impurity. It is intuitive and handles non-linear "
            "decision boundaries well, but is prone to overfitting if tree depth is unrestricted."
        ),
        'RandomForestClassifier': (
            "A robust ensemble classifier that builds a forest of diverse decision trees during training using bootstrap "
            "samples and random feature subsets. By aggregating their predictions via majority voting, it dramatically "
            "reduces variance, prevents overfitting, and achieves high generalization accuracy."
        ),
        'XGBoostClassifier': (
            "Extreme Gradient Boosting Classifier; an elite, highly optimized gradient boosting framework designed for "
            "maximum speed and predictive power. It features built-in regularization (L1/L2) to penalize complex models, handles "
            "sparse matrices and missing values natively, and dominates tabular data competitions."
        ),
        'GradientBoostingClassifier': (
            "An iterative ensemble classifier that trains decision trees sequentially. Each new tree is specifically "
            "optimized to predict and correct the residual classification errors (pseudo-residuals) of the previous sequence "
            "of trees, steadily boosting overall accuracy."
        ),
        'SupportVectorClassifier': (
            "Support Vector Classifier (SVC); finds the optimal separating hyperplane that maximizes the margin (distance) "
            "between different classes. Using kernel functions (like RBF, poly, or linear), it can project data into higher "
            "dimensions to carve out complex, non-linear decision boundaries safely."
        ),
        'HistGradientBoostingClassifier': (
            "A modern, histogram-based gradient boosting classifier inspired by LightGBM. It bins continuous features "
            "into integer-valued histograms (e.g., 256 bins), significantly accelerating training times and enabling scalable "
            "performance on massive datasets containing millions of rows."
        ),
        'RBFSampler': (
            "A preprocessing transformer that uses random Fourier features to approximate kernel maps based on radial "
            "basis functions. It transforms non-linear feature spaces so that linear classifiers can separate complex, "
            "non-linear data efficiently."
        ),
        'AdaBoostClassifier': (
            "Adaptive Boosting Classifier; an iterative boosting algorithm that builds a sequence of weak classifiers "
            "(typically shallow decision trees). In each successive iteration, it increases the sample weights of instances "
            "that were misclassified, forcing the next model to focus on hard-to-learn edge cases."
        ),
        'BaggingClassifier': (
            "Bootstrap Aggregating Classifier; an ensemble meta-estimator that fits multiple base classifiers independently "
            "in parallel on random subsets (with replacement) of the training set. It combines their predictions through "
            "majority voting to lower variance and stabilize predictions."
        ),
        'ExtraTreesClassifier': (
            "Extremely Randomized Trees Classifier; an ensemble variation of Random Forest where decision trees are randomized "
            "even further. Instead of computing the optimal split threshold for features, splits are chosen entirely at random, "
            "which accelerates training speed and further reduces model variance."
        ),
        'LGBMClassifier': (
            "LightGBM Classifier; a lightning-fast, high-efficiency gradient boosting framework developed by Microsoft. "
            "Unlike traditional level-wise methods, it grows trees **leaf-wise** by expanding the leaf with the maximum delta loss, "
            "delivering superior classification accuracy and minimal memory footprint."
        ),
        'CatBoostClassifier': (
            "A high-performance gradient boosting classifier specialized for handling **categorical features natively and automatically** "
            "without manual encoding. It utilizes ordered boosting to prevent target leakage, yielding outstanding out-of-the-box "
            "accuracy with minimal hyperparameter tuning."
        )
    }
    if model_name == "LogisticRegression":
        model = LogisticRegression(C=C, l1_ratio=l1_ratio, dual=dual, fit_intercept=fit_intercept)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    elif model_name == "NaiveBayes":
        model = GaussianNB(priors=priors)
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model_name == "XGBoostClassifier":
        model = XGBClassifier(n_estimators=n_estimators)
    elif model_name == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate)
    elif model_name == "SupportVectorClassifier":
        model = SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iter)
    elif model_name == "HistGradientBoostingClassifier":
        model = HistGradientBoostingClassifier(loss=loss, learning_rate=learning_rate, max_iter=max_iter)
    elif model_name == "RBFSampler":
        model = RBFSampler(n_components=n_neighbors)
    elif model_name == "AdaBoostClassifier":
        model = AdaBoostClassifier(n_estimators=n_estimators)
    elif model_name == "BaggingClassifier":
        model = BaggingClassifier(n_estimators=n_estimators)
    elif model_name == "ExtraTreesClassifier":
        model = ExtraTreesClassifier(n_estimators=n_estimators)
    elif model_name == "LGBMClassifier":
        model = LGBMClassifier(n_estimators=n_estimators)
    elif model_name == "CatBoostClassifier":
        model = CatBoostClassifier(n_estimators=n_estimators)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, model_name +": "+ descriptions.get(model_name, "Description not found.")


def select_clustering_model(model_name, n_clusters=3, eps=0.5, min_samples=5, linkage="ward", bandwidth=None, damping=0.5):
    descriptions = {
        'KMeans': (
            "A centroid-based clustering algorithm that partitions data into 'k' distinct clusters by minimizing "
            "the within-cluster sum of squares (inertia). It is fast, highly scalable, and works exceptionally well "
            "when clusters are spherical and roughly equal in size."
        ),
        'DBSCAN': (
            "Density-Based Spatial Clustering of Applications with Noise; groups together data points that are closely packed "
            "together, marking points in low-density regions as outliers (noise). It can discover arbitrarily shaped clusters "
            "and does not require specifying the number of clusters in advance."
        ),
        'AgglomerativeClustering': (
            "A hierarchical bottom-up clustering approach where each data point starts in its own cluster and pairs of "
            "clusters are successively merged based on a linkage criterion (e.g., ward, average, complete). It creates a tree-like "
            "hierarchy (dendrogram) of clusters."
        ),
        'SpectralClustering': (
            "A graph-based clustering technique that uses the spectrum (eigenvalues) of a similarity matrix of the data to perform "
            "dimensionality reduction before clustering in fewer dimensions. It excels at identifying complex, non-linear cluster "
            "boundaries and intertwined shapes."
        ),
        'MeanShift': (
            "A centroid-based non-parametric clustering algorithm that shifts data points iteratively toward the mode (region of "
            "highest density) using a sliding-window kernel. It automatically determines the number of clusters based on data density "
            "without requiring 'k' to be predefined."
        ),
        'AffinityPropagation': (
            "A message-passing clustering algorithm that finds clusters by simultaneously sending messages (responsibility and availability) "
            "between pairs of data points until convergence. It does not require specifying the number of clusters beforehand and "
            "identifies representative exemplars for each cluster."
        ),
        'OPTICS': (
            "Ordering Points To Identify the Clustering Structure; a density-based algorithm similar to DBSCAN designed to handle "
            "varying densities. It creates an augmented ordering of the dataset representing its density-based structure, making it "
            "less sensitive to hyperparameter choices."
        ),
        'Birch': (
            "Balanced Iterative Reducing and Clustering using Hierarchies; a memory-efficient, fast clustering algorithm designed "
            "for very large datasets. It builds a tree structure (CF Tree) to summarize cluster features, allowing linear scalability "
            "with dataset size."
        ),
        'GaussianMixture': (
            "A probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions "
            "with unknown parameters. Unlike KMeans, it provides soft clustering (probabilities of belonging to multiple clusters) and "
            "can model clusters with different ellipsoidal shapes."
        )
    }

    if model_name == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    elif model_name == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif model_name == 'HDBSCAN':
        model = HDBSCAN(min_samples=min_samples)
    elif model_name == 'AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    elif model_name == 'SpectralClustering':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='kmeans')
    elif model_name == 'MeanShift':
        model = MeanShift(bandwidth=bandwidth)
    elif model_name == 'AffinityPropagation':
        model = AffinityPropagation(damping=damping, random_state=42)
    elif model_name == 'OPTICS':
        model = OPTICS(min_samples=min_samples, eps=eps)
    elif model_name == 'Birch':
        model = Birch(n_clusters=n_clusters)
    elif model_name == 'GaussianMixture':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unknown clustering model: {model_name}")

    return model, model_name +": "+ descriptions.get(model_name, "Description not found.")




def select_dimensionality_reduction_model(model_name, n_components=2, kernel="linear", gamma=None, perplexity=30.0, n_neighbors=15, init="pca"):
    descriptions = {
        'PCA': (
            "Principal Component Analysis; a linear dimensionality reduction technique that orthogonally transforms the "
            "original correlated features into a smaller set of uncorrelated variables called principal components. "
            "It maximizes the preserved variance, making it ideal for data compression, noise reduction, and visualization."
        ),
        'TruncatedSVD': (
            "Dimensionality reduction using truncated singular value decomposition. Unlike PCA, it works directly on "
            "sample-feature matrices without centering the data first, making it extremely efficient for **sparse datasets** "
            "(such as TF-IDF matrices in Natural Language Processing)."
        ),
        'LDA': (
            "Linear Discriminant Analysis; a supervised dimensionality reduction and classification technique. "
            "Unlike PCA (which is unsupervised), LDA finds the feature subspace that maximizes class separability, "
            "making it exceptionally useful as a preprocessing step before classification tasks."
        ),
        'KernelPCA': (
            "An extension of PCA that uses kernel techniques (such as RBF, polynomial, or sigmoid) to perform linear PCA "
            "in a higher-dimensional kernel space. This allows it to capture and reduce complex, **non-linear structures** "
            "and curved manifolds in the data."
        ),
        'TSNE': (
            "t-Distributed Stochastic Neighbor Embedding; a non-linear technique well-suited for embedding high-dimensional "
            "data into a low-dimensional space of 2 or 3 dimensions. It preserves local neighborhood structures exceptionally well, "
            "making it the gold standard for visualizing complex clusters in scatter plots."
        ),
        'UMAP': (
            "Uniform Manifold Approximation and Projection; a modern non-linear dimensionality reduction technique. "
            "Based on Riemannian geometry and algebraic topology, UMAP often preserves both local and global data structure "
            "better than t-SNE while offering significantly faster computation times."
        ),
        'FastICA': (
            "Independent Component Analysis; a fast algorithm designed to separate a multivariate signal into additive "
            "subcomponents by maximizing statistical independence. It is widely used in signal processing, artifact removal, "
            "and blind source separation (e.g., separating mixed audio tracks)."
        ),
        'FactorAnalysis': (
            "A linear statistical method used to describe variability among observed, correlated variables in terms of a "
            "potentially lower number of unobserved random variables called factors. It models underlying latent traits "
            "while explicitly accounting for Gaussian noise in the features."
        ),
        'NMF': (
            "Non-Negative Matrix Factorization; a linear dimensionality reduction method where both the decomposed matrices "
            "and features are constrained to have non-negative elements. It is widely used in recommender systems, image processing, "
            "and topic modeling because it yields an intuitive, part-based additive representation."
        ),
        'DictionaryLearning': (
            "A matrix factorization method that finds a sparse representation of the input data (a sparse coding) over a learned "
            "dictionary of atoms. It is powerful for denoising, image reconstruction, and feature extraction in sparse domains."
        )
    }

    if model_name == 'PCA':
        model = PCA(n_components=n_components, random_state=42)
    elif model_name == 'TruncatedSVD':
        model = TruncatedSVD(n_components=n_components, random_state=42)
    elif model_name == 'LDA':
        model = LDA(n_components=n_components)
    elif model_name == "QDA":
        model = QuadraticDiscriminantAnalysis()
    elif model_name == 'KernelPCA':
        model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=42)
    elif model_name == 'TSNE':
        model = TSNE(n_components=n_components, perplexity=perplexity, init=init, random_state=42, learning_rate='auto')
    elif model_name == 'UMAP':
        if UMAP is None:
            raise ImportError("UMAP is not installed. Please install it using 'pip install umap-learn'.")
        model = UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    elif model_name == 'FastICA':
        model = FastICA(n_components=n_components, random_state=42)
    elif model_name == 'FactorAnalysis':
        model = FactorAnalysis(n_components=n_components, random_state=42)
    elif model_name == 'NMF':
        model = NMF(n_components=n_components, random_state=42)
    elif model_name == 'DictionaryLearning':
        model = DictionaryLearning(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown dimensionality reduction model: {model_name}")

    return model, model_name +": "+ descriptions.get(model_name, "Description not found.")

def select_model(
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
        model_type = model_type.lower()

        if model_type in ["linear_regression", "linear", "regression_linear"]:
            return select_linear_regression_model(
                model_name=model_name,
                alpha=alpha,
                C=C,
                l1_ratio=l1_ratio,
                dual=dual,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                selection=selection
            )

        elif model_type in ["nonlinear_regression", "nonlinear", "regression"]:
            return select_nonlinear_regression_model(
                model_name=model_name,
                degree=degree,
                kernel=kernel,
                max_depth=max_depth,
                neighbors=neighbors,
                n_estimators=n_estimators
            )

        elif model_type in ["classification", "clf"]:
            return select_classification_model(
                model_name=model_name,
                C=C,
                l1_ratio=l1_ratio,
                dual=dual,
                weights=weights,
                algorithm=algorithm,
                fit_intercept=fit_intercept,
                criterion=criterion,
                n_neighbors=n_neighbors,
                max_depth=max_depth,
                max_features=max_features,
                priors=priors,
                loss=loss,
                learning_rate=learning_rate,
                max_iter=max_iter,
                n_estimators=n_estimators,
                degree=degree,
                kernel=kernel
            )

        elif model_type in ["clustering", "cluster"]:
            return select_clustering_model(
                model_name=model_name,
                n_clusters=n_clusters,
                eps=eps,
                min_samples=min_samples,
                linkage=linkage,
                bandwidth=bandwidth,
                damping=damping
            )

        elif model_type in ["dimensionality_reduction", "dim_red", "reduction"]:
            return select_dimensionality_reduction_model(
                model_name=model_name,
                n_components=n_components,
                kernel=kernel,
                gamma=gamma,
                perplexity=perplexity,
                n_neighbors=n_neighbors,
                init=init
            )

        else:
            raise ValueError(f"Bilinmeyen model_type: {model_type}")