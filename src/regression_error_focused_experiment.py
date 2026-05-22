"""Experiment focused on reducing difficult regression errors for Cesar's workflow."""

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
from sklearn.pipeline import Pipeline

try:
    from regression_setup import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN  # type: ignore
except ImportError:
    from .regression_setup import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN  # type: ignore

try:
    from data_preprocessing import build_regression_preprocessor  # type: ignore
except ImportError:
    from .data_preprocessing import build_regression_preprocessor  # type: ignore

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
    from advanced_feature_engineering import (  # type: ignore
        GroupMeanTargetEncoder,
        add_basic_precision_features,
        get_historical_numeric_features,
        get_precision_categorical_features,
        get_precision_numeric_features,
    )
except ImportError:
    from .advanced_feature_engineering import (  # type: ignore
        GroupMeanTargetEncoder,
        add_basic_precision_features,
        get_historical_numeric_features,
        get_precision_categorical_features,
        get_precision_numeric_features,
    )

try:
    from error_focused_features import (  # type: ignore
        add_error_focused_features,
        get_error_focused_categorical_features,
        get_error_focused_numeric_features,
    )
except ImportError:
    from .error_focused_features import (  # type: ignore
        add_error_focused_features,
        get_error_focused_categorical_features,
        get_error_focused_numeric_features,
    )

try:
    from model_training import split_regression_data, train_regression_models  # type: ignore
except ImportError:
    from .model_training import split_regression_data, train_regression_models  # type: ignore

try:
    from model_evaluation import (  # type: ignore
        calculate_regression_metrics,
        plot_real_vs_predicted,
        plot_residuals,
    )
except ImportError:
    from .model_evaluation import (  # type: ignore
        calculate_regression_metrics,
        plot_real_vs_predicted,
        plot_residuals,
    )

try:
    from operational_metrics import tolerance_accuracy  # type: ignore
except ImportError:
    from .operational_metrics import tolerance_accuracy  # type: ignore

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
    from error_analysis import (  # type: ignore
        add_error_segment_columns,
        build_prediction_error_dataframe,
        calculate_segment_error_metrics,
        plot_segment_errors,
        save_segment_error_metrics,
    )
except ImportError:
    from .error_analysis import (  # type: ignore
        add_error_segment_columns,
        build_prediction_error_dataframe,
        calculate_segment_error_metrics,
        plot_segment_errors,
        save_segment_error_metrics,
    )

try:
    from segment_error_improvement import build_segment_comparison, save_segment_comparison  # type: ignore
except ImportError:
    from .segment_error_improvement import build_segment_comparison, save_segment_comparison  # type: ignore

try:
    from residual_correction import (  # type: ignore
        evaluate_residual_correction,
        predict_with_residual_correction,
        train_residual_correction_model,
    )
except ImportError:
    from .residual_correction import (  # type: ignore
        evaluate_residual_correction,
        predict_with_residual_correction,
        train_residual_correction_model,
    )


CURRENT_BASELINE = {
    "version": "current_balanced_model",
    "model": "GradientBoostingRegressor",
    "MAE": 11.4711,
    "RMSE": 14.5709,
    "R2": 0.0307,
    "acc_5_min": 29.17,
    "acc_10_min": 52.50,
    "acc_15_min": 70.42,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_error_focused_estimators(random_state: int = 42) -> dict[str, Any]:
    return {
        "gradient_boosting_baseline_like": GradientBoostingRegressor(random_state=random_state),
        "gradient_boosting_huber": GradientBoostingRegressor(
            loss="huber",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "extra_trees_error_focused": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest_error_focused": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting_error_focused": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "gradient_boosting_low_lr": GradientBoostingRegressor(
            loss="huber",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=2,
            random_state=random_state,
        ),
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


def _calculate_full_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    basic = calculate_regression_metrics(y_true, np.asarray(y_pred))
    error = pd.Series(y_true).reset_index(drop=True) - pd.Series(y_pred)
    abs_error = error.abs()

    return {
        "MAE": float(basic["MAE"]),
        "RMSE": float(basic["RMSE"]),
        "R2": float(basic["R2"]),
        "mean_error": float(error.mean()),
        "median_absolute_error": float(abs_error.median()),
        "p90_absolute_error": float(abs_error.quantile(0.90)),
        "acc_5_min": float(tolerance_accuracy(y_true, y_pred, tolerance=5) * 100),
        "acc_10_min": float(tolerance_accuracy(y_true, y_pred, tolerance=10) * 100),
        "acc_15_min": float(tolerance_accuracy(y_true, y_pred, tolerance=15) * 100),
        "acc_20_min": float(tolerance_accuracy(y_true, y_pred, tolerance=20) * 100),
    }


def _metrics_to_operational_rows(model_name: str, metrics: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "model": model_name,
            "tolerance_minutes": 5,
            "accuracy_percent": metrics["acc_5_min"],
            "accuracy": metrics["acc_5_min"] / 100,
        },
        {
            "model": model_name,
            "tolerance_minutes": 10,
            "accuracy_percent": metrics["acc_10_min"],
            "accuracy": metrics["acc_10_min"] / 100,
        },
        {
            "model": model_name,
            "tolerance_minutes": 15,
            "accuracy_percent": metrics["acc_15_min"],
            "accuracy": metrics["acc_15_min"] / 100,
        },
        {
            "model": model_name,
            "tolerance_minutes": 20,
            "accuracy_percent": metrics["acc_20_min"],
            "accuracy": metrics["acc_20_min"] / 100,
        },
    ]


def _select_best_balanced_model(metrics_df: pd.DataFrame) -> str:
    baseline_acc_10 = CURRENT_BASELINE["acc_10_min"]

    eligible = metrics_df[
        (metrics_df["acc_10_min"] >= baseline_acc_10 - 2.0)
        | (
            (metrics_df["MAE"] <= CURRENT_BASELINE["MAE"] - 0.2)
            & (metrics_df["RMSE"] <= CURRENT_BASELINE["RMSE"] - 0.2)
        )
    ].copy()

    if eligible.empty:
        eligible = metrics_df.copy()

    eligible = eligible.sort_values(
        by=["MAE", "RMSE", "R2", "acc_10_min", "p90_absolute_error"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)

    return str(eligible.iloc[0]["model"])


def _plot_metric_bars(metrics_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    sorted_df = metrics_df.sort_values(metric, ascending=(metric != "R2"))
    plt.bar(sorted_df["model"], sorted_df[metric], color="#3E79A8")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Comparacion de modelos - {metric}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_tolerance_comparison(operational_df: pd.DataFrame, output_path: Path) -> None:
    pivot = operational_df.pivot_table(
        index="tolerance_minutes",
        columns="model",
        values="accuracy_percent",
        aggfunc="mean",
    )

    plt.figure(figsize=(11, 6))
    for model_name in pivot.columns:
        plt.plot(pivot.index, pivot[model_name], marker="o", label=model_name)

    plt.xlabel("Tolerancia (minutos)")
    plt.ylabel("Acierto (%)")
    plt.title("Comparacion de acierto operacional por tolerancia")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_current_vs_new(best_row: pd.Series) -> pd.DataFrame:
    new_row = {
        "version": "best_error_focused_model",
        "model": str(best_row["model"]),
        "MAE": float(best_row["MAE"]),
        "RMSE": float(best_row["RMSE"]),
        "R2": float(best_row["R2"]),
        "acc_5_min": float(best_row["acc_5_min"]),
        "acc_10_min": float(best_row["acc_10_min"]),
        "acc_15_min": float(best_row["acc_15_min"]),
        "p90_absolute_error": float(best_row["p90_absolute_error"]),
    }

    baseline_row = {
        **CURRENT_BASELINE,
        "p90_absolute_error": np.nan,
    }

    comparison_df = pd.DataFrame([baseline_row, new_row])

    baseline = comparison_df.iloc[0]
    comparison_df["delta_MAE"] = comparison_df["MAE"] - float(baseline["MAE"])
    comparison_df["delta_RMSE"] = comparison_df["RMSE"] - float(baseline["RMSE"])
    comparison_df["delta_R2"] = comparison_df["R2"] - float(baseline["R2"])
    comparison_df["delta_acc_10_min"] = comparison_df["acc_10_min"] - float(baseline["acc_10_min"])

    decision = "mantener modelo actual"
    if (
        new_row["MAE"] < baseline["MAE"]
        and new_row["RMSE"] < baseline["RMSE"]
        and new_row["R2"] >= baseline["R2"]
        and new_row["acc_10_min"] >= baseline["acc_10_min"] - 1
    ):
        decision = "usar modelo nuevo"
    elif (
        new_row["MAE"] <= baseline["MAE"]
        and new_row["RMSE"] <= baseline["RMSE"]
        and new_row["acc_10_min"] >= baseline["acc_10_min"] - 2
    ):
        decision = "modelo nuevo util para analisis, evaluar antes de reemplazar"

    comparison_df["decision"] = ["baseline", decision]
    return comparison_df


def run_error_focused_improvement_experiment(
    processed_data_path,
    output_dir_metrics,
    output_dir_plots,
    output_dir_reports,
    random_state: int = 42,
) -> dict:
    """
    Ejecuta experimento local enfocado en reducir errores dificiles.
    """
    processed_data_path = Path(processed_data_path)
    output_dir_metrics = Path(output_dir_metrics)
    output_dir_plots = Path(output_dir_plots)
    output_dir_reports = Path(output_dir_reports)

    _ensure_dir(output_dir_metrics)
    _ensure_dir(output_dir_plots)
    _ensure_dir(output_dir_reports)

    if not processed_data_path.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {processed_data_path}")

    df = pd.read_csv(processed_data_path)

    # Feature engineering chain.
    df_features = add_logistics_features(df)
    df_features = add_basic_precision_features(df_features)
    df_features = add_error_focused_features(df_features)

    numeric_features = list(
        dict.fromkeys(
            NUMERIC_FEATURES
            + get_enhanced_numeric_features()
            + get_precision_numeric_features()
            + get_error_focused_numeric_features()
        )
    )
    categorical_features = list(
        dict.fromkeys(
            CATEGORICAL_FEATURES
            + get_enhanced_categorical_features()
            + get_precision_categorical_features()
            + get_error_focused_categorical_features()
        )
    )

    all_features = [feature for feature in numeric_features + categorical_features if feature in df_features.columns]
    numeric_features = [feature for feature in numeric_features if feature in all_features]
    categorical_features = [feature for feature in categorical_features if feature in all_features]

    X_train_raw, X_test_raw, y_train, y_test = split_regression_data(
        df=df_features,
        feature_columns=all_features,
        target_column=TARGET_COLUMN,
        test_size=0.2,
        random_state=random_state,
    )

    # Historical (train-only) means to improve difficult segments without leakage.
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
    selected_features = numeric_features_final + categorical_features_final

    X_train = X_train_hist[selected_features].copy()
    X_test = X_test_hist[selected_features].copy()

    preprocessor = build_regression_preprocessor(
        numeric_features=numeric_features_final,
        categorical_features=categorical_features_final,
    )

    estimators = _build_error_focused_estimators(random_state=random_state)
    model_pipelines = _build_pipelines(preprocessor, estimators)
    trained_models = train_regression_models(model_pipelines, X_train, y_train)

    metrics_rows: list[dict[str, Any]] = []
    operational_rows: list[dict[str, Any]] = []
    prediction_registry: dict[str, np.ndarray] = {}
    model_registry: dict[str, Any] = dict(trained_models)

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        prediction_registry[model_name] = np.asarray(y_pred)
        model_metrics = _calculate_full_metrics(y_test, y_pred)
        metrics_rows.append({"model": model_name, **model_metrics})
        operational_rows.extend(_metrics_to_operational_rows(model_name, model_metrics))

    # Residual correction over the best initial MAE candidate.
    base_metrics_df = pd.DataFrame(metrics_rows)
    base_candidate_name = str(
        base_metrics_df.sort_values(
            by=["MAE", "RMSE", "R2", "acc_10_min", "p90_absolute_error"],
            ascending=[True, True, False, False, True],
        ).iloc[0]["model"]
    )

    residual_model = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    fitted_base, fitted_residual = train_residual_correction_model(
        base_model=trained_models[base_candidate_name],
        residual_model=residual_model,
        X_train=X_train,
        y_train=y_train,
    )

    residual_name = f"{base_candidate_name}_residual_correction"
    residual_pred = predict_with_residual_correction(fitted_base, fitted_residual, X_test)
    prediction_registry[residual_name] = np.asarray(residual_pred)

    residual_metrics = evaluate_residual_correction(
        base_model=fitted_base,
        residual_model=fitted_residual,
        X_test=X_test,
        y_test=y_test,
    )
    metrics_rows.append({"model": residual_name, **residual_metrics})
    operational_rows.extend(_metrics_to_operational_rows(residual_name, residual_metrics))

    model_registry[residual_name] = {
        "base": fitted_base,
        "residual": fitted_residual,
    }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(
        by=["MAE", "RMSE", "R2", "acc_10_min", "p90_absolute_error"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)

    operational_df = pd.DataFrame(operational_rows)

    best_model_name = _select_best_balanced_model(metrics_df)
    best_row = metrics_df[metrics_df["model"] == best_model_name].iloc[0]
    best_pred = prediction_registry[best_model_name]

    # Segment-level analysis for best model.
    error_df_best = build_prediction_error_dataframe(X_test, y_test, best_pred)
    error_df_best = add_error_segment_columns(error_df_best)

    mandatory_segment_columns = [
        "tipo_vehiculo",
        "trafico_nivel",
        "clima",
        "id_bodega",
        "hora_punta",
        "distancia_larga",
        "carga_pesada",
        "muchas_paradas",
    ]

    segment_metrics_df = calculate_segment_error_metrics(error_df_best, mandatory_segment_columns)

    # Segment comparison vs baseline-like model.
    baseline_model_name = "gradient_boosting_baseline_like"
    if baseline_model_name not in prediction_registry:
        baseline_model_name = str(metrics_df.iloc[0]["model"])

    error_df_baseline = build_prediction_error_dataframe(
        X_test,
        y_test,
        prediction_registry[baseline_model_name],
    )
    error_df_baseline = add_error_segment_columns(error_df_baseline)
    baseline_segment_metrics_df = calculate_segment_error_metrics(error_df_baseline, mandatory_segment_columns)

    segment_comparison_df = build_segment_comparison(
        baseline_segment_metrics_df=baseline_segment_metrics_df,
        new_segment_metrics_df=segment_metrics_df,
    )

    # Feature importance of best model.
    best_model_obj = model_registry[best_model_name]
    if isinstance(best_model_obj, dict):
        # Residual-correction case: use base model for permutation importance reference.
        model_for_importance = best_model_obj["base"]
    else:
        model_for_importance = best_model_obj

    feature_importance_df = calculate_permutation_importance(
        model=model_for_importance,
        X_test=X_test,
        y_test=y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=random_state,
    )

    # Save CSV outputs.
    metrics_path = output_dir_metrics / "cesar_error_focused_metrics.csv"
    operational_path = output_dir_metrics / "cesar_error_focused_operational_accuracy.csv"
    current_vs_path = output_dir_metrics / "cesar_error_focused_current_vs_new.csv"
    segment_metrics_path = output_dir_metrics / "cesar_error_focused_segment_metrics.csv"
    segment_comparison_path = output_dir_metrics / "cesar_error_focused_segment_comparison.csv"
    feature_importance_path = output_dir_metrics / "cesar_error_focused_feature_importance.csv"

    metrics_df.to_csv(metrics_path, index=False)
    operational_df.to_csv(operational_path, index=False)
    save_segment_error_metrics(segment_metrics_df, segment_metrics_path)
    save_segment_comparison(segment_comparison_df, segment_comparison_path)
    save_feature_importance(feature_importance_df, feature_importance_path)

    current_vs_new_df = _build_current_vs_new(best_row)
    current_vs_new_df.to_csv(current_vs_path, index=False)

    # Save plots.
    _plot_metric_bars(metrics_df, "MAE", output_dir_plots / "cesar_error_focused_mae_comparison.png")
    _plot_metric_bars(metrics_df, "RMSE", output_dir_plots / "cesar_error_focused_rmse_comparison.png")
    _plot_metric_bars(metrics_df, "R2", output_dir_plots / "cesar_error_focused_r2_comparison.png")
    _plot_tolerance_comparison(
        operational_df,
        output_dir_plots / "cesar_error_focused_tolerance_comparison.png",
    )

    predictions_df = pd.DataFrame(
        {
            "real": y_test.reset_index(drop=True),
            "predicho": best_pred,
        }
    )
    predictions_df["error"] = predictions_df["real"] - predictions_df["predicho"]
    predictions_df["error_absoluto"] = predictions_df["error"].abs()

    plot_real_vs_predicted(
        predictions_df,
        output_dir_plots / "cesar_error_focused_real_vs_predicho.png",
    )
    plot_residuals(
        predictions_df,
        output_dir_plots / "cesar_error_focused_residuals.png",
    )
    plot_segment_errors(
        segment_metrics_df,
        output_dir_plots / "cesar_error_focused_segment_errors.png",
        top_n=15,
    )
    plot_feature_importance(
        feature_importance_df,
        output_dir_plots / "cesar_error_focused_feature_importance.png",
        top_n=15,
    )

    # Decision
    decision = str(current_vs_new_df.iloc[1]["decision"])

    summary_path = output_dir_reports / "cesar_error_focused_improvement_summary.md"
    summary_text = f"""# Resumen experimento enfocado en errores dificiles - Cesar

## Baseline actual

- Modelo: {CURRENT_BASELINE['model']}
- MAE: {CURRENT_BASELINE['MAE']}
- RMSE: {CURRENT_BASELINE['RMSE']}
- R2: {CURRENT_BASELINE['R2']}
- Acierto +/-10 min: {CURRENT_BASELINE['acc_10_min']}%
- Acierto +/-15 min: {CURRENT_BASELINE['acc_15_min']}%

## Mejor modelo nuevo

- Modelo: {best_model_name}
- MAE: {float(best_row['MAE']):.6f}
- RMSE: {float(best_row['RMSE']):.6f}
- R2: {float(best_row['R2']):.6f}
- Acierto +/-10 min: {float(best_row['acc_10_min']):.2f}%
- Acierto +/-15 min: {float(best_row['acc_15_min']):.2f}%
- P90 error absoluto: {float(best_row['p90_absolute_error']):.4f}

## Modelo usado para correccion de residuos

- Base para residuos: {base_candidate_name}
- Variante residual: {residual_name}

## Decision recomendada

{decision}

## Segmentos comparados

{segment_comparison_df.to_string(index=False)}
"""
    summary_path.write_text(summary_text, encoding="utf-8")

    return {
        "metrics": metrics_df,
        "operational_accuracy": operational_df,
        "current_vs_new": current_vs_new_df,
        "segment_metrics": segment_metrics_df,
        "segment_comparison": segment_comparison_df,
        "feature_importance": feature_importance_df,
        "best_model_name": best_model_name,
        "best_model": model_registry[best_model_name],
        "decision": decision,
        "residual_model_name": residual_name,
    }
