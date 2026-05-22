import os
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


RANDOM_STATE = 42


def run_grid_search(model, param_grid, X_train, y_train, scoring, cv=5):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def run_randomized_search(model, param_distributions, X_train, y_train, scoring, cv=5, n_iter=20):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=cv,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    return random_search


def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred)
    }

    return results


def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    return results


def save_best_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def save_tuning_results(search_object, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results_df = pd.DataFrame(search_object.cv_results_)
    results_df.to_csv(output_path, index=False)

    return results_df


def summarize_best_params(search_object, model_name, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = pd.DataFrame([{
        "model_name": model_name,
        "best_score": search_object.best_score_,
        "best_params": search_object.best_params_
    }])

    summary.to_csv(output_path, index=False)

    return summary