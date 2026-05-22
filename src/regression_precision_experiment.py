"""Main experiment to improve exact-time regression precision for Cesar's workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

try:
    from advanced_feature_engineering import (  # type: ignore
        GroupMeanTargetEncoder,
        add_basic_precision_features,
        build_precision_feature_lists,
        get_historical_numeric_features,
    )
except ImportError:
    from .advanced_feature_engineering import (  # type: ignore
        GroupMeanTargetEncoder,
        add_basic_precision_features,
        build_precision_feature_lists,
        get_historical_numeric_features,
    )

try:
    from cross_validation_evaluation import (  # type: ignore
        cross_validate_regression_models,
        save_cross_validation_results,
    )
except ImportError:
    from .cross_validation_evaluation import (  # type: ignore
        cross_validate_regression_models,
        save_cross_validation_results,
    )

try:
    from data_preprocessing import build_regression_preprocessor  # type: ignore
except ImportError:
    from .data_preprocessing import build_regression_preprocessor  # type: ignore

try:
    from error_analysis import (  # type: ignore
        build_prediction_error_dataframe,
        error_by_segment,
        plot_top_error_segments,
        save_error_by_segment,
    )
except ImportError:
    from .error_analysis import (  # type: ignore
        build_prediction_error_dataframe,
        error_by_segment,
        plot_top_error_segments,
        save_error_by_segment,
    )

try:
    from feature_engineering import (  # type: ignore
        add_logistics_features,
        get_enhanced_categorical_features,
        get_enhanced_numeric_features,
    )
except ImportError:
    from .feature_engineering import (  # type: ignore
        add_logistics_features,
        get_enhanced_categorical_features,
        get_enhanced_numeric_features,
    )

try:
    from feature_selection_experiment import (  # type: ignore
        save_feature_selection_results,
        select_top_features,
    )
except ImportError:
    from .feature_selection_experiment import (  # type: ignore
        save_feature_selection_results,
        select_top_features,
    )

try:
    from hyperparameter_tuning import (  # type: ignore
        evaluate_optimized_model,
        get_param_grid_for_model,
        tune_regression_model_gridsearch,
        tune_regression_model_randomizedsearch,
    )
except ImportError:
    from .hyperparameter_tuning import (  # type: ignore
        evaluate_optimized_model,
        get_param_grid_for_model,
        tune_regression_model_gridsearch,
        tune_regression_model_randomizedsearch,
    )

try:
    from model_evaluation import (  # type: ignore
        calculate_regression_metrics,
        create_predictions_dataframe,
        evaluate_regression_models,
        plot_real_vs_predicted,
        plot_residuals,
    )
except ImportError:
    from .model_evaluation import (  # type: ignore
        calculate_regression_metrics,
        create_predictions_dataframe,
        evaluate_regression_models,
        plot_real_vs_predicted,
        plot_residuals,
    )

try:
    from model_interpretability import (  # type: ignore
        calculate_permutation_importance,
        plot_feature_importance,
        save_feature_importance,
    )
except ImportError:
    from .model_interpretability import (  # type: ignore
        calculate_permutation_importance,
        plot_feature_importance,
        save_feature_importance,
    )

try:
    from model_training import split_regression_data, train_regression_models  # type: ignore
except ImportError:
    from .model_training import split_regression_data, train_regression_models  # type: ignore

try:
    from operational_metrics import (  # type: ignore
        evaluate_tolerance_accuracies,
        evaluate_tolerance_accuracies_for_models,
        save_operational_accuracy,
        tolerance_accuracy,
    )
except ImportError:
    from .operational_metrics import (  # type: ignore
        evaluate_tolerance_accuracies,
        evaluate_tolerance_accuracies_for_models,
        save_operational_accuracy,
        tolerance_accuracy,
    )

try:
    from regression_setup import (  # type: ignore
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
    )
except ImportError:
    from .regression_setup import (  # type: ignore
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
    )

try:
    from target_transform_experiment import build_log_target_models  # type: ignore
except ImportError:
    from .target_transform_experiment import build_log_target_models  # type: ignore


CURRENT_TECHNICAL = {
    "version": "current_technical_optimized",
    "MAE": 11.59,
    "RMSE": 14.58,
    "R2": 0.0290,
    "acc_5_min": 28.33,
    "acc_10_min": 49.58,
    "acc_15_min": 70.00,
}

PREVIOUS_OPERATIONAL = {
    "version": "previous_operational_improved",
    "MAE": 11.5062,
    "RMSE": 14.7391,
    "R2": 0.0082,
    "acc_5_min": 29.17,
    "acc_10_min": 55.00,
    "acc_15_min": 70.42,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_precision_estimators(random_state: int = 42) -> dict[str, Any]:
    return {
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            random_state=random_state,
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=250,
        ),
        "knn_regressor": KNeighborsRegressor(n_neighbors=7),
    }


def _build_pipelines(preprocessor, estimators: dict[str, Any]) -> dict[str, Pipeline]:
    pipelines: dict[str, Pipeline] = {}
    for model_name, estimator in estimators.items():
        pipelines[model_name] = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
    return pipelines


def _merge_technical_operational(
    technical_df: pd.DataFrame,
    operational_df: pd.DataFrame,
) -> pd.DataFrame:
    op_pivot = (
        operational_df.pivot_table(
            index="model",
            columns="tolerance_minutes",
            values="accuracy_percent",
            aggfunc="mean",
        )
        .rename(columns={5: "acc_5_min", 10: "acc_10_min", 15: "acc_15_min", 20: "acc_20_min"})
        .reset_index()
    )
    merged = technical_df.merge(op_pivot, on="model", how="left")
    return merged


def _select_model_for_precision(
    combined_df: pd.DataFrame,
) -> str:
    sorted_df = combined_df.sort_values(
        by=["MAE", "RMSE", "R2", "acc_10_min", "acc_15_min"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    return str(sorted_df.iloc[0]["model"])


def _grid_size(param_grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def _safe_get_acc(op_df: pd.DataFrame, model_name: str, tolerance: int) -> float:
    row = op_df[(op_df["model"] == model_name) & (op_df["tolerance_minutes"] == tolerance)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["accuracy_percent"])


def _build_decision_note(row: pd.Series) -> str:
    notes = []
    if row["version"] == "precision_experiment":
        if row["MAE"] < CURRENT_TECHNICAL["MAE"]:
            notes.append("MAE mejor que tecnico")
        else:
            notes.append("MAE no mejora tecnico")

        if row["RMSE"] < CURRENT_TECHNICAL["RMSE"]:
            notes.append("RMSE mejor que tecnico")
        else:
            notes.append("RMSE no mejora tecnico")

        if row["R2"] > CURRENT_TECHNICAL["R2"]:
            notes.append("R2 mejor que tecnico")
        else:
            notes.append("R2 no mejora tecnico")

        if row["acc_10_min"] >= PREVIOUS_OPERATIONAL["acc_10_min"]:
            notes.append("mantiene/mejora acc_10 operacional")
        else:
            notes.append("pierde acc_10 operacional")

    return " | ".join(notes) if notes else "baseline"


def _plot_version_metric(
    comparison_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.bar(comparison_df["version"], comparison_df[metric], color=["#4C78A8", "#F58518", "#2F4B7C"])
    plt.xlabel("Version")
    plt.ylabel(metric)
    plt.title(f"Comparacion de {metric} entre versiones")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_tolerance_versions(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    tolerance_rows = []
    for _, row in comparison_df.iterrows():
        tolerance_rows.extend(
            [
                {"version": row["version"], "tolerance_minutes": 5, "accuracy_percent": row["acc_5_min"]},
                {"version": row["version"], "tolerance_minutes": 10, "accuracy_percent": row["acc_10_min"]},
                {"version": row["version"], "tolerance_minutes": 15, "accuracy_percent": row["acc_15_min"]},
            ]
        )

    tol_df = pd.DataFrame(tolerance_rows)

    plt.figure(figsize=(10, 6))
    for version in tol_df["version"].unique():
        subset = tol_df[tol_df["version"] == version].sort_values("tolerance_minutes")
        plt.plot(subset["tolerance_minutes"], subset["accuracy_percent"], marker="o", label=version)

    plt.xlabel("Tolerancia (minutos)")
    plt.ylabel("Acierto (%)")
    plt.title("Comparacion de acierto por tolerancia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _evaluate_feature_selection(
    base_estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
    numeric_features_all: list[str],
    categorical_features_all: list[str],
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    feature_sets = {
        "all_features": list(X_train.columns),
        "top_5": select_top_features(importance_df, 5),
        "top_8": select_top_features(importance_df, 8),
        "top_10": select_top_features(importance_df, 10),
    }

    for set_name, selected in feature_sets.items():
        selected = [feature for feature in selected if feature in X_train.columns]
        if not selected:
            continue

        numeric_sel = [feature for feature in selected if feature in numeric_features_all]
        categorical_sel = [feature for feature in selected if feature in categorical_features_all]

        try:
            preprocessor_sel = build_regression_preprocessor(
                numeric_features=numeric_sel,
                categorical_features=categorical_sel,
            )

            pipeline_sel = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_sel),
                    ("model", clone(base_estimator)),
                ]
            )
            pipeline_sel.fit(X_train[selected], y_train)
            y_pred_sel = pipeline_sel.predict(X_test[selected])
            metrics_sel = calculate_regression_metrics(y_test, y_pred_sel)

            rows.append(
                {
                    "feature_set": set_name,
                    "n_features": len(selected),
                    "MAE": metrics_sel["MAE"],
                    "RMSE": metrics_sel["RMSE"],
                    "R2": metrics_sel["R2"],
                    "acc_10_min": tolerance_accuracy(y_test, y_pred_sel, tolerance=10) * 100,
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "feature_set": set_name,
                    "n_features": len(selected),
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "acc_10_min": np.nan,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def run_precision_improvement_experiment(
    processed_data_path,
    output_dir_metrics,
    output_dir_plots,
    output_dir_reports,
    random_state: int = 42,
) -> dict:
    """Run a precision-focused experiment to improve exact-time regression estimates."""
    processed_data_path = Path(processed_data_path)
    output_dir_metrics = Path(output_dir_metrics)
    output_dir_plots = Path(output_dir_plots)
    output_dir_reports = Path(output_dir_reports)

    _ensure_dir(output_dir_metrics)
    _ensure_dir(output_dir_plots)
    _ensure_dir(output_dir_reports)

    warnings: list[str] = []

    if not processed_data_path.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {processed_data_path}")

    df = pd.read_csv(processed_data_path)
    df_features = add_logistics_features(df)
    df_features = add_basic_precision_features(df_features)

    previous_numeric = get_enhanced_numeric_features()
    previous_categorical = get_enhanced_categorical_features()

    numeric_features, categorical_features, all_features = build_precision_feature_lists(
        base_numeric_features=NUMERIC_FEATURES,
        base_categorical_features=CATEGORICAL_FEATURES,
        previous_extra_numeric_features=previous_numeric,
        previous_extra_categorical_features=previous_categorical,
    )

    all_features = [feature for feature in all_features if feature in df_features.columns]
    numeric_features = [feature for feature in numeric_features if feature in all_features]
    categorical_features = [feature for feature in categorical_features if feature in all_features]

    X_train_raw, X_test_raw, y_train, y_test = split_regression_data(
        df=df_features,
        feature_columns=all_features,
        target_column=TARGET_COLUMN,
        test_size=0.2,
        random_state=random_state,
    )

    historical_encoder = GroupMeanTargetEncoder(
        group_columns=["id_bodega", "tipo_vehiculo", "trafico_nivel"],
        target_column=TARGET_COLUMN,
    )
    historical_encoder.fit(X_train_raw, y_train)
    X_train_hist = historical_encoder.transform(X_train_raw)
    X_test_hist = historical_encoder.transform(X_test_raw)

    historical_features = [
        feature
        for feature in get_historical_numeric_features()
        if feature in X_train_hist.columns and feature in X_test_hist.columns
    ]

    numeric_features_final = list(dict.fromkeys(numeric_features + historical_features))
    categorical_features_final = list(dict.fromkeys(categorical_features))
    all_features_final = numeric_features_final + categorical_features_final

    X_train = X_train_hist[all_features_final].copy()
    X_test = X_test_hist[all_features_final].copy()

    preprocessor = build_regression_preprocessor(
        numeric_features=numeric_features_final,
        categorical_features=categorical_features_final,
    )

    estimators = _build_precision_estimators(random_state=random_state)
    base_models = _build_pipelines(preprocessor=preprocessor, estimators=estimators)

    trained_models = train_regression_models(base_models, X_train, y_train)

    technical_base = evaluate_regression_models(trained_models, X_test, y_test)
    operational_base = evaluate_tolerance_accuracies_for_models(
        trained_models,
        X_test,
        y_test,
        tolerances=[5, 10, 15, 20],
    )

    log_model_templates = build_log_target_models(base_models)
    trained_log_models: dict[str, Any] = {}
    for model_name, model in log_model_templates.items():
        try:
            model.fit(X_train, y_train)
            trained_log_models[model_name] = model
        except Exception as exc:  # pragma: no cover
            warnings.append(f"No se pudo entrenar modelo log-target {model_name}: {exc}")

    if trained_log_models:
        technical_log = evaluate_regression_models(trained_log_models, X_test, y_test)
        operational_log = evaluate_tolerance_accuracies_for_models(
            trained_log_models,
            X_test,
            y_test,
            tolerances=[5, 10, 15, 20],
        )
    else:
        technical_log = pd.DataFrame(columns=["model", "MAE", "RMSE", "R2"])
        operational_log = pd.DataFrame(columns=["model", "tolerance_minutes", "accuracy", "accuracy_percent"])

    precision_metrics_df = pd.concat([technical_base, technical_log], ignore_index=True)
    operational_accuracy_all_df = pd.concat([operational_base, operational_log], ignore_index=True)

    # Select best base model for RMSE-oriented tuning.
    combined_base = _merge_technical_operational(technical_base, operational_base)
    best_base_model_name = _select_model_for_precision(combined_base)

    tuned_model = trained_models[best_base_model_name]
    tuned_model_name = f"{best_base_model_name}_rmse_tuned"
    best_params: dict[str, Any] = {}
    tuning_method = "none"

    try:
        param_grid = get_param_grid_for_model(best_base_model_name)
        grid_size = _grid_size(param_grid) if param_grid else 0

        if param_grid:
            if grid_size <= 120:
                tuning_method = "grid_search_rmse"
                search = tune_regression_model_gridsearch(
                    model=trained_models[best_base_model_name],
                    param_grid=param_grid,
                    X_train=X_train,
                    y_train=y_train,
                    cv=5,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                )
            else:
                tuning_method = "randomized_search_rmse"
                search = tune_regression_model_randomizedsearch(
                    model=trained_models[best_base_model_name],
                    param_distributions=param_grid,
                    X_train=X_train,
                    y_train=y_train,
                    n_iter=30,
                    cv=5,
                    scoring="neg_root_mean_squared_error",
                    random_state=random_state,
                    n_jobs=-1,
                )

            tuned_model = search.best_estimator_
            best_params = dict(search.best_params_)
        else:
            warnings.append(
                f"No hay grilla de parametros para {best_base_model_name}; se usa modelo base sin tuning."
            )
    except Exception as exc:  # pragma: no cover
        warnings.append(f"Fallo tuning para {best_base_model_name}: {exc}")

    tuned_metrics_df = evaluate_optimized_model(
        optimized_model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        model_name=tuned_model_name,
    )
    tuned_operational_df = evaluate_tolerance_accuracies(
        model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        tolerances=[5, 10, 15, 20],
    )
    tuned_operational_df["model"] = tuned_model_name

    precision_metrics_df = pd.concat([precision_metrics_df, tuned_metrics_df], ignore_index=True)
    operational_accuracy_all_df = pd.concat([operational_accuracy_all_df, tuned_operational_df], ignore_index=True)

    combined_all = _merge_technical_operational(precision_metrics_df, operational_accuracy_all_df)
    final_model_name = _select_model_for_precision(combined_all)

    model_registry: dict[str, Any] = {}
    model_registry.update(trained_models)
    model_registry.update(trained_log_models)
    model_registry[tuned_model_name] = tuned_model

    if final_model_name not in model_registry:
        raise RuntimeError(f"No se encontro el modelo final seleccionado: {final_model_name}")

    final_model = model_registry[final_model_name]
    final_model_metrics = precision_metrics_df[precision_metrics_df["model"] == final_model_name].copy()
    final_operational_accuracy = operational_accuracy_all_df[
        operational_accuracy_all_df["model"] == final_model_name
    ].copy()

    # Cross-validation stability on base models and tuned model.
    # For leakage safety, CV is computed without historical target-mean features.
    try:
        preprocessor_cv = build_regression_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        cv_models = _build_pipelines(
            preprocessor=preprocessor_cv,
            estimators=estimators,
        )
        if best_params:
            tuned_for_cv = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor_cv)),
                    ("model", clone(estimators[best_base_model_name])),
                ]
            )
            tuned_for_cv.set_params(**best_params)
            cv_models[tuned_model_name] = tuned_for_cv
        cv_results_df = cross_validate_regression_models(
            models=cv_models,
            X=df_features[all_features].copy(),
            y=df_features[TARGET_COLUMN].copy(),
            cv=5,
        )
    except Exception as exc:  # pragma: no cover
        warnings.append(f"Fallo validacion cruzada: {exc}")
        cv_results_df = pd.DataFrame()

    # Predictions, diagnostics and interpretability.
    predictions_df = create_predictions_dataframe(final_model, X_test, y_test)
    plot_real_vs_predicted(
        predictions_df,
        output_dir_plots / "cesar_precision_real_vs_predicho.png",
    )
    plot_residuals(
        predictions_df,
        output_dir_plots / "cesar_precision_residuals.png",
    )

    try:
        feature_importance_df = calculate_permutation_importance(
            model=final_model,
            X_test=X_test,
            y_test=y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=10,
            random_state=random_state,
        )
    except Exception as exc:  # pragma: no cover
        warnings.append(f"Fallo importance permutation: {exc}")
        feature_importance_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": np.nan,
                "importance_std": np.nan,
            }
        )

    save_feature_importance(
        feature_importance_df,
        output_dir_metrics / "cesar_precision_feature_importance.csv",
    )
    plot_feature_importance(
        feature_importance_df,
        output_dir_plots / "cesar_precision_feature_importance.png",
        top_n=15,
    )

    error_df = build_prediction_error_dataframe(X_test, y_test, predictions_df["predicho"])
    segment_columns = [
        "trafico_nivel",
        "clima",
        "tipo_vehiculo",
        "id_bodega",
        "hora_punta",
        "hora_punta_manana",
        "hora_punta_tarde",
        "hora_nocturna",
        "chofer_experto",
    ]
    segment_df = error_by_segment(error_df, segment_columns)
    save_error_by_segment(segment_df, output_dir_metrics / "cesar_precision_error_by_segment.csv")
    plot_top_error_segments(segment_df, output_dir_plots / "cesar_precision_error_by_segment.png")

    # Feature selection experiment.
    base_estimator_for_fs = clone(estimators[best_base_model_name])
    fs_results_df = _evaluate_feature_selection(
        base_estimator=base_estimator_for_fs,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        importance_df=feature_importance_df,
        numeric_features_all=numeric_features_final,
        categorical_features_all=categorical_features_final,
        random_state=random_state,
    )
    save_feature_selection_results(
        fs_results_df,
        output_dir_metrics / "cesar_precision_feature_selection_results.csv",
    )

    # Save metrics artifacts.
    precision_metrics_path = output_dir_metrics / "cesar_precision_experiment_metrics.csv"
    cv_metrics_path = output_dir_metrics / "cesar_precision_cross_validation.csv"
    operational_path = output_dir_metrics / "cesar_precision_operational_accuracy.csv"
    current_vs_path = output_dir_metrics / "cesar_precision_current_vs_previous.csv"

    precision_metrics_df.to_csv(precision_metrics_path, index=False)
    if not cv_results_df.empty:
        save_cross_validation_results(cv_results_df, cv_metrics_path)
    else:
        cv_results_df.to_csv(cv_metrics_path, index=False)

    save_operational_accuracy(operational_accuracy_all_df, operational_path)

    final_acc_5 = _safe_get_acc(operational_accuracy_all_df, final_model_name, 5)
    final_acc_10 = _safe_get_acc(operational_accuracy_all_df, final_model_name, 10)
    final_acc_15 = _safe_get_acc(operational_accuracy_all_df, final_model_name, 15)

    current_vs_precision = pd.DataFrame(
        [
            CURRENT_TECHNICAL,
            PREVIOUS_OPERATIONAL,
            {
                "version": "precision_experiment",
                "MAE": float(final_model_metrics.iloc[0]["MAE"]),
                "RMSE": float(final_model_metrics.iloc[0]["RMSE"]),
                "R2": float(final_model_metrics.iloc[0]["R2"]),
                "acc_5_min": final_acc_5,
                "acc_10_min": final_acc_10,
                "acc_15_min": final_acc_15,
            },
        ]
    )
    current_vs_precision["decision_notes"] = current_vs_precision.apply(_build_decision_note, axis=1)
    current_vs_precision.to_csv(current_vs_path, index=False)

    # Required comparison plots.
    _plot_version_metric(current_vs_precision, "MAE", output_dir_plots / "cesar_precision_mae_comparison.png")
    _plot_version_metric(current_vs_precision, "RMSE", output_dir_plots / "cesar_precision_rmse_comparison.png")
    _plot_version_metric(current_vs_precision, "R2", output_dir_plots / "cesar_precision_r2_comparison.png")
    _plot_tolerance_versions(
        current_vs_precision,
        output_dir_plots / "cesar_precision_tolerance_comparison.png",
    )

    # Decision logic summary.
    precision_row = current_vs_precision[current_vs_precision["version"] == "precision_experiment"].iloc[0]
    if precision_row["MAE"] < CURRENT_TECHNICAL["MAE"] and precision_row["RMSE"] < CURRENT_TECHNICAL["RMSE"] and precision_row["R2"] > CURRENT_TECHNICAL["R2"]:
        conclusion = "Mejoro la estimacion exacta del tiempo de entrega."
    elif precision_row["MAE"] < CURRENT_TECHNICAL["MAE"] or precision_row["RMSE"] < CURRENT_TECHNICAL["RMSE"]:
        if precision_row["acc_10_min"] < PREVIOUS_OPERATIONAL["acc_10_min"]:
            conclusion = "Mejoro tecnicamente pero perdio acierto operacional."
        else:
            conclusion = "Mejoro tecnicamente y mantuvo/mejoro acierto operacional relevante."
    elif precision_row["acc_10_min"] >= PREVIOUS_OPERATIONAL["acc_10_min"]:
        conclusion = "Mejoro operacionalmente pero no tecnicamente."
    else:
        conclusion = "Empeoro respecto a modelos anteriores para precision y operacion."

    summary_path = output_dir_reports / "cesar_precision_improvement_summary.md"
    summary_text = f"""# Resumen experimento de precision - Cesar

## Modelo final seleccionado

{final_model_name}

## Metodo de tuning aplicado

{tuning_method}

## Mejores hiperparametros

{best_params}

## Metricas del modelo final

- MAE: {float(final_model_metrics.iloc[0]['MAE']):.6f}
- RMSE: {float(final_model_metrics.iloc[0]['RMSE']):.6f}
- R2: {float(final_model_metrics.iloc[0]['R2']):.6f}
- Acierto +/-5 min: {final_acc_5:.2f}%
- Acierto +/-10 min: {final_acc_10:.2f}%
- Acierto +/-15 min: {final_acc_15:.2f}%

## Comparacion contra modelos previos

{current_vs_precision.to_string(index=False)}

## Conclusion

{conclusion}

## Warnings del experimento

{chr(10).join(['- ' + w for w in warnings]) if warnings else '- Sin warnings'}
"""
    summary_path.write_text(summary_text, encoding="utf-8")

    return {
        "precision_metrics": precision_metrics_df,
        "cross_validation_results": cv_results_df,
        "operational_accuracy": operational_accuracy_all_df,
        "current_vs_precision": current_vs_precision,
        "best_model_name": final_model_name,
        "best_model": final_model,
        "best_model_metrics": final_model_metrics,
        "feature_importance": feature_importance_df,
        "error_by_segment": segment_df,
        "best_params": best_params,
        "warnings": warnings,
        "conclusion": conclusion,
    }
