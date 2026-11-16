from pprint import pprint
from process.data_loading.load_csv import *
from process.data_preprocessing.scale_data import *
from process.model_selection.select_model import *
from process.training.train_model import *
from process.data_splitting.split_data import *
from process.prediction.predict_data import *
from process.evaluation.evaluate_regression import *

filepath = "/home/melis/PycharmProjects/MLImplementation/data/advertising/advertising.csv"
data = load_csv_file(filepath)
y = data["Sales"]
X = data.drop("Sales", axis=1)
X = scale_data(X)
X_train, X_test, y_train, y_test = split_sklearn_dataset(X, y, test_size=0.2, train_size=0.8)
model,  explanation = select_nonlinear_regression_model("HistGradientBoostingRegressor")
print(explanation)
trained_model = train_model(model, X_train, y_train)
pred = predict_data(trained_model, X_test)
evaluation = evaluate_regression(pred, y_test)
pprint(evaluation)


