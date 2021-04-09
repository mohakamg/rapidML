from typing import Dict


def get_supported_regressors() -> Dict:
    from framework.predictors.train.regressors.sklearn.linear_regression import LinearRegression
    from framework.predictors.train.regressors.sklearn.random_forest import RandomForest
    from framework.predictors.train.regressors.xgboost import XGBoost

    SUPPORTED_MODELS = {
        "Linear Regressor": LinearRegression,
        "Random Forest": RandomForest,
        "XGBoost": XGBoost
    }
    return SUPPORTED_MODELS
