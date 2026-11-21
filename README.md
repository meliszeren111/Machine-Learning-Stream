# # Machine Learning Stream 

is a modular Python toolkit designed to streamline machine learning workflows. It provides centralized selection functions for a wide range of algorithms, spanning regression, classification, clustering, dimensionality reduction, and feature transformation. Each selection function automatically instantiates the chosen model with customizable hyperparameters while returning a detailed technical description of its underlying mechanics.
Below is the complete list of all 63 algorithms and preprocessing tools implemented in the toolkit, categorized by their primary machine learning tasks:

# 1. Linear and Generalized Models
- LinearRegression
- LogisticRegression
- Lasso
- Ridge
- LassoCV
- RidgeCV

# 2. Ensemble and Boosting Models
RandomForestRegressor
RandomForestClassifier
GradientBoostingRegressor
GradientBoostingClassifier
HistGradientBoostingRegressor
HistGradientBoostingClassifier
AdaBoostRegressor
AdaBoostClassifier
BaggingRegressor
BaggingClassifier
ExtraTreesRegressor
ExtraTreesClassifier
XGBRegressor
XGBClassifier
LGBMRegressor
LGBMClassifier
CatBoostRegressor
CatBoostClassifier

# 3. Support Vector, Neighbors, and Neural Networks
SVR (Support Vector Regression)
SVC (Support Vector Classification)
KNeighborsRegressor
KNeighborsClassifier
GaussianProcessRegressor
MLPRegressor (Multi-Layer Perceptron)

# 4. Trees and Probabilistic Models
DecisionTreeRegressor
DecisionTreeClassifier
GaussianNB (Naive Bayes)

# 5. Clustering and Unsupervised Models
KMeans
DBSCAN
HDBSCAN
AgglomerativeClustering
SpectralClustering
MeanShift
AffinityPropagation
OPTICS
Birch
GaussianMixture (GMM)

# 6. Dimensionality Reduction and Manifold Learning
PCA (Principal Component Analysis)
TruncatedSVD
KernelPCA
FastICA
FactorAnalysis
NMF (Non-Negative Matrix Factorization)
DictionaryLearning
LinearDiscriminantAnalysis (LDA)
TSNE (t-Distributed Stochastic Neighbor Embedding)
UMAP (Uniform Manifold Approximation and Projection)
7. Feature Transformation and Preprocessing
PolynomialFeatures
SplineTransformer
RBFSampler (Radial Basis Function Sampler)
