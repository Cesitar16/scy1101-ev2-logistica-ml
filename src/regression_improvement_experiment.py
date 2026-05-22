"""Experimental workflow to improve Cesar's regression precision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    from data_preprocessing import build_regression_preprocessor  # type: ignore
except ImportError:
    from .data_preprocessing import build_regression_preprocessor  # type: ignore

try:
    from feature_engineering import (  # type: ignore
        add_logistics_features,
        build_enhanced_feature_lists,
    )
except ImportError:
    from .feature_engineering import (  # type: ignore
        add_logistics_features,
        build_enhanced_feature_lists,
    )

try:
    from hyperparameter_tuning import (  # type: ignore
        evaluate_optimized_model,
        get_param_grid_for_model,
        tune_regression_model_for_tolerance_10,
    )
except ImportError:
    from .hyperparameter_tuning import (  # type: ignore
        evaluate_optimized_model,
        get_param_grid_for_model,
        tune_regression_model_for_tolerance_10,
    )

try:
    from model_evaluation import (  # type: ignore
        create_predictions_dataframe,
        evaluate_regression_models,
        plot_real_vs_predicted,
        plot_residuals,
    )
except ImportError:
    from .model_evaluation import (  # type: ignore
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
    from model_training import (  # type: ignore
        build_regression_pipelines,
        split_regression_data,
        train_regression_models,
    )
except ImportError:
    from .model_training import (  # type: ignore
        build_regression_pipelines,
        split_regression_data,
        train_regression_models,
    )

try:
    from operational_metrics import (  # type: ignore
        evaluate_tolerance_accuracies,
        evaluate_tolerance_accuracies_for_models,
        plot_tolerance_accuracy_comparison,
        save_operational_accuracy,
    )
except ImportError:
    from .operational_metrics import (  # type: ignore
        evaluate_tolerance_accuracies,
        evaluate_tolerance_accuracies_for_models,
        plot_tolerance_accuracy_comparison,
        save_operational_accuracy,
    )

CURRENT_METRICS = {
    "MAE": 11.59,
    "RMSE": 14.58,
    "R2": 0.0290,
    "acc_5_min": 28.33,
    "acc_10_min": 49.58,
    "acc_15_min": 70.00,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_accuracy_for_tolerance(
    accuracy_df: pd.DataFrame,
    model_name: str,
    tolerance_minutes: int,
) -> float:
    match = accuracy_df[
        (accuracy_df["model"] == model_name)
        & (accuracy_df["tolerance_minutes"] == tolerance_minutes)
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["accuracy_percent"])


def _select_best_model_by_operational_criteria(
    technical_metrics_df: pd.DataFrame,
    operational_accuracy_df: pd.DataFrame,
) -> str:
    tolerance_10 = operational_accuracy_df[
        operational_accuracy_df["tolerance_minutes"] == 10
    ][["model", "accuracy_percent"]].copy()

    merged = technical_metrics_df.merge(
        tolerance_10,
        how="left",
        on="model",
    )

    merged = merged.sort_values(
        by=["accuracy_percent", "RMSE", "MAE", "R2"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    return str(merged.iloc[0]["model"])


def _select_final_candidate(
    best_base_model_name: str,
    technical_metrics_df: pd.DataFrame,
    operational_accuracy_df: pd.DataFrame,
    optimized_model_name: str,
    optimized_metrics_df: pd.DataFrame,
    optimized_operational_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    base_row = technical_metrics_df[technical_metrics_df["model"] == best_base_model_name].iloc[0]
    base_acc_10 = _get_accuracy_for_tolerance(
        operational_accuracy_df,
        best_base_model_name,
        10,
    )

    opt_row = optimized_metrics_df.iloc[0]
    opt_acc_10 = _get_accuracy_for_tolerance(
        optimized_operational_df,
        optimized_model_name,
        10,
    )

    if (
        (opt_acc_10 > base_acc_10)
        or (opt_acc_10 == base_acc_10 and opt_row["RMSE"] < base_row["RMSE"])
        or (
            opt_acc_10 == base_acc_10
            and opt_row["RMSE"] == base_row["RMSE"]
            and opt_row["MAE"] < base_row["MAE"]
        )
        or (
            opt_acc_10 == base_acc_10
            and opt_row["RMSE"] == base_row["RMSE"]
            and opt_row["MAE"] == base_row["MAE"]
            and opt_row["R2"] > base_row["R2"]
        )
    ):
        return optimized_model_name, optimized_metrics_df.copy(), optimized_operational_df.copy(), "optimized"

    base_metrics_df = technical_metrics_df[
        technical_metrics_df["model"] == best_base_model_name
    ].copy()
    base_operational_df = operational_accuracy_df[
        operational_accuracy_df["model"] == best_base_model_name
    ].copy()
    return best_base_model_name, base_metrics_df, base_operational_df, "base"


def _create_current_vs_improved(
    improved_metrics_df: pd.DataFrame,
    improved_operational_df: pd.DataFrame,
) -> pd.DataFrame:
    improved_mae = float(improved_metrics_df.iloc[0]["MAE"])
    improved_rmse = float(improved_metrics_df.iloc[0]["RMSE"])
    improved_r2 = float(improved_metrics_df.iloc[0]["R2"])

    improved_acc_5 = float(improved_operational_df.loc[
        improved_operational_df["tolerance_minutes"] == 5,
        "accuracy_percent",
    ].iloc[0])
    improved_acc_10 = float(improved_operational_df.loc[
        improved_operational_df["tolerance_minutes"] == 10,
        "accuracy_percent",
    ].iloc[0])
    improved_acc_15 = float(improved_operational_df.loc[
        improved_operational_df["tolerance_minutes"] == 15,
        "accuracy_percent",
    ].iloc[0])

    current_vs_improved = pd.DataFrame(
        [
            {
                "version": "current_optimized",
                "MAE": CURRENT_METRICS["MAE"],
                "RMSE": CURRENT_METRICS["RMSE"],
                "R2": CURRENT_METRICS["R2"],
                "acc_5_min": CURRENT_METRICS["acc_5_min"],
                "acc_10_min": CURRENT_METRICS["acc_10_min"],
                "acc_15_min": CURRENT_METRICS["acc_15_min"],
            },
            {
                "version": "improved_experiment",
                "MAE": improved_mae,
                "RMSE": improved_rmse,
                "R2": improved_r2,
                "acc_5_min": improved_acc_5,
                "acc_10_min": improved_acc_10,
                "acc_15_min": improved_acc_15,
            },
        ]
    )
    return current_vs_improved


def _plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.bar(comparison_df["version"], comparison_df[metric], color=["#4C78A8", "#F58518"])
    plt.xlabel("Version")
    plt.ylabel(metric)
    plt.title(f"Current vs Improved - {metric}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_tolerance_comparison(
    improved_operational_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = [
        {
            "model": "current_optimized",
            "tolerance_minutes": 5,
            "accuracy_percent": CURRENT_METRICS["acc_5_min"],
        },
        {
            "model": "current_optimized",
            "tolerance_minutes": 10,
            "accuracy_percent": CURRENT_METRICS["acc_10_min"],
        },
        {
            "model": "current_optimized",
            "tolerance_minutes": 15,
            "accuracy_percent": CURRENT_METRICS["acc_15_min"],
        },
    ]

    improved_model_name = str(improved_operational_df.iloc[0]["model"])
    for _, row in improved_operational_df.iterrows():
        tolerance = int(row["tolerance_minutes"])
        rows.append(
            {
                "model": improved_model_name,
                "tolerance_minutes": tolerance,
                "accuracy_percent": float(row["accuracy_percent"]),
            }
        )

    tolerance_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    for model_name in tolerance_df["model"].unique():
        model_df = tolerance_df[tolerance_df["model"] == model_name].sort_values("tolerance_minutes")
        plt.plot(
            model_df["tolerance_minutes"],
            model_df["accuracy_percent"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Tolerancia en minutos")
    plt.ylabel("Acierto (%)")
    plt.title("Current vs Improved - Acierto por tolerancia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return tolerance_df


def _build_conclusion_text(current_vs_improved: pd.DataFrame) -> str:
    current = current_vs_improved[current_vs_improved["version"] == "current_optimized"].iloc[0]
    improved = current_vs_improved[current_vs_improved["version"] == "improved_experiment"].iloc[0]

    if improved["acc_10_min"] > current["acc_10_min"]:
        return (
            "El modelo mejoro operacionalmente porque aumento el acierto dentro de +/-10 minutos."
        )

    if improved["acc_10_min"] <= current["acc_10_min"] and improved["RMSE"] < current["RMSE"]:
        return (
            "El modelo mejoro tecnicamente, pero no mejoro el criterio operacional principal."
        )

    return (
        "Las mejoras probadas no superaron el modelo actual. Se recomienda mantener el modelo anterior."
    )


def run_improvement_experiment(
    processed_data_path,
    output_dir_metrics,
    output_dir_plots,
    output_dir_reports,
    random_state: int = 42,
) -> dict:
    """Run a full regression improvement experiment focused on +/-10 minute accuracy."""
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
    df_features = add_logistics_features(df)

    enhanced_numeric, enhanced_categorical, all_features = build_enhanced_feature_lists(
        base_numeric_features=NUMERIC_FEATURES,
        base_categorical_features=CATEGORICAL_FEATURES,
    )

    enhanced_numeric = [column for column in enhanced_numeric if column in df_features.columns]
    enhanced_categorical = [column for column in enhanced_categorical if column in df_features.columns]
    all_features = [column for column in all_features if column in df_features.columns]

    preprocessor = build_regression_preprocessor(
        numeric_features=enhanced_numeric,
        categorical_features=enhanced_categorical,
    )

    X_train, X_test, y_train, y_test = split_regression_data(
        df=df_features,
        feature_columns=all_features,
        target_column=TARGET_COLUMN,
        test_size=0.2,
        random_state=random_state,
    )

    pipelines = build_regression_pipelines(
        preprocessor=preprocessor,
        random_state=random_state,
    )

    trained_models = train_regression_models(
        models=pipelines,
        X_train=X_train,
        y_train=y_train,
    )

    technical_metrics_df = evaluate_regression_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
    )

    operational_accuracy_df = evaluate_tolerance_accuracies_for_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        tolerances=[5, 10, 15, 20],
    )

    best_model_name = _select_best_model_by_operational_criteria(
        technical_metrics_df=technical_metrics_df,
        operational_accuracy_df=operational_accuracy_df,
    )

    best_model = trained_models[best_model_name]
    param_grid = get_param_grid_for_model(best_model_name)

    optimized_model = best_model
    optimized_model_name = f"{best_model_name}_tolerance10_optimized"

    if param_grid:
        tuning_search = tune_regression_model_for_tolerance_10(
            model=best_model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=5,
            n_jobs=-1,
        )
        optimized_model = tuning_search.best_estimator_
        best_params = tuning_search.best_params_
        best_score = float(tuning_search.best_score_)
    else:
        tuning_search = None
        best_params = {}
        best_score = float("nan")

    optimized_metrics_df = evaluate_optimized_model(
        optimized_model=optimized_model,
        X_test=X_test,
        y_test=y_test,
        model_name=optimized_model_name,
    )

    optimized_operational_df = evaluate_tolerance_accuracies(
        model=optimized_model,
        X_test=X_test,
        y_test=y_test,
        tolerances=[5, 10, 15, 20],
    )
    optimized_operational_df["model"] = optimized_model_name

    final_model_name, improved_metrics_df, improved_operational_df, selected_from = _select_final_candidate(
        best_base_model_name=best_model_name,
        technical_metrics_df=technical_metrics_df,
        operational_accuracy_df=operational_accuracy_df,
        optimized_model_name=optimized_model_name,
        optimized_metrics_df=optimized_metrics_df,
        optimized_operational_df=optimized_operational_df,
    )

    final_model = optimized_model if selected_from == "optimized" else best_model

    improved_metrics_output_path = output_dir_metrics / "cesar_regression_improved_metrics.csv"
    improved_operational_output_path = (
        output_dir_metrics / "cesar_regression_improved_operational_accuracy.csv"
    )
    current_vs_improved_output_path = (
        output_dir_metrics / "cesar_regression_current_vs_improved.csv"
    )
    feature_importance_output_path = output_dir_metrics / "cesar_feature_importance.csv"

    improved_metrics_df.to_csv(improved_metrics_output_path, index=False)
    save_operational_accuracy(improved_operational_df, improved_operational_output_path)

    current_vs_improved = _create_current_vs_improved(
        improved_metrics_df=improved_metrics_df,
        improved_operational_df=improved_operational_df,
    )
    current_vs_improved.to_csv(current_vs_improved_output_path, index=False)

    rmse_plot_path = output_dir_plots / "cesar_current_vs_improved_rmse.png"
    mae_plot_path = output_dir_plots / "cesar_current_vs_improved_mae.png"
    tolerance_plot_path = output_dir_plots / "cesar_current_vs_improved_tolerance_accuracy.png"
    feature_importance_plot_path = output_dir_plots / "cesar_feature_importance.png"
    real_vs_pred_plot_path = output_dir_plots / "cesar_improved_real_vs_predicho.png"
    residuals_plot_path = output_dir_plots / "cesar_improved_residuals.png"

    _plot_metric_comparison(current_vs_improved, "RMSE", rmse_plot_path)
    _plot_metric_comparison(current_vs_improved, "MAE", mae_plot_path)
    tolerance_comparison_df = _plot_tolerance_comparison(
        improved_operational_df=improved_operational_df,
        output_path=tolerance_plot_path,
    )

    # Keep full operational table plot for models explored in experiment as well.
    plot_tolerance_accuracy_comparison(
        accuracy_df=operational_accuracy_df,
        output_path=output_dir_plots / "cesar_improved_models_tolerance_accuracy.png",
    )

    feature_importance_df = calculate_permutation_importance(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=random_state,
    )
    save_feature_importance(feature_importance_df, feature_importance_output_path)
    plot_feature_importance(feature_importance_df, feature_importance_plot_path, top_n=15)

    predictions_df = create_predictions_dataframe(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
    )
    plot_real_vs_predicted(predictions_df, real_vs_pred_plot_path)
    plot_residuals(predictions_df, residuals_plot_path)

    conclusion = _build_conclusion_text(current_vs_improved)

    summary_output_path = output_dir_reports / "cesar_regression_improvement_summary.md"
    summary_text = f"""# Resumen de mejora del modelo de regresion - Cesar

## Modelo base seleccionado para experimento

{best_model_name}

## Modelo optimizado por tolerancia +/-10

{optimized_model_name}

## Modelo final recomendado

{final_model_name} ({selected_from})

## Mejores hiperparametros

{best_params}

## Mejor score CV (tolerance +/-10)

{best_score}

## Metricas del modelo final recomendado

- MAE: {float(improved_metrics_df.iloc[0]['MAE']):.6f}
- RMSE: {float(improved_metrics_df.iloc[0]['RMSE']):.6f}
- R2: {float(improved_metrics_df.iloc[0]['R2']):.6f}

## Acierto operacional del modelo final recomendado

- +/-5 min: {float(improved_operational_df.loc[improved_operational_df['tolerance_minutes']==5, 'accuracy_percent'].iloc[0]):.2f}%
- +/-10 min: {float(improved_operational_df.loc[improved_operational_df['tolerance_minutes']==10, 'accuracy_percent'].iloc[0]):.2f}%
- +/-15 min: {float(improved_operational_df.loc[improved_operational_df['tolerance_minutes']==15, 'accuracy_percent'].iloc[0]):.2f}%
- +/-20 min: {float(improved_operational_df.loc[improved_operational_df['tolerance_minutes']==20, 'accuracy_percent'].iloc[0]):.2f}%

## Comparacion current vs improved

{current_vs_improved.to_string(index=False)}

## Conclusion

{conclusion}
"""
    summary_output_path.write_text(summary_text, encoding="utf-8")

    return {
        "improved_metrics": improved_metrics_df,
        "operational_accuracy": improved_operational_df,
        "best_model_name": best_model_name,
        "optimized_model": optimized_model,
        "optimized_metrics": optimized_metrics_df,
        "base_vs_improved": current_vs_improved,
        "feature_importance": feature_importance_df,
        "tolerance_comparison": tolerance_comparison_df,
        "final_model_name": final_model_name,
        "final_model_source": selected_from,
        "best_params": best_params,
        "conclusion": conclusion,
        "paths": {
            "improved_metrics": improved_metrics_output_path,
            "improved_operational_accuracy": improved_operational_output_path,
            "current_vs_improved": current_vs_improved_output_path,
            "feature_importance": feature_importance_output_path,
            "rmse_plot": rmse_plot_path,
            "mae_plot": mae_plot_path,
            "tolerance_plot": tolerance_plot_path,
            "feature_importance_plot": feature_importance_plot_path,
            "real_vs_pred_plot": real_vs_pred_plot_path,
            "residuals_plot": residuals_plot_path,
            "summary_report": summary_output_path,
        },
    }
