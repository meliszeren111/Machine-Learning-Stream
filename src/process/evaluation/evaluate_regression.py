from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
    mean_absolute_percentage_error,
    r2_score
)


def evaluate_regression(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    try:
        rmsle = root_mean_squared_log_error(y_test, y_pred)
    except ValueError:
        rmsle = None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "RMSLE": rmsle,
        "MAPE": mape,
        "R2": r2
    }


