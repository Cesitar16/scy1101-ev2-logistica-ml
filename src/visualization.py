"""Funciones para visualizar y guardar graficos del EDA de logistica."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_output_dir(path: str | Path) -> Path:
    """Asegura que el directorio de salida exista."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _ensure_parent(path: str | Path) -> Path:
    """Asegura que exista el directorio padre de un archivo."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def plot_target_distribution(df: pd.DataFrame, target_column: str, output_path: str | Path) -> str:
    """Guarda histograma y boxplot de la variable objetivo."""
    file_path = _ensure_parent(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df[target_column], kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Distribucion de {target_column}")
    axes[0].set_xlabel(target_column)
    axes[0].set_ylabel("Frecuencia")

    sns.boxplot(x=df[target_column], ax=axes[1], color="#9ecae1")
    axes[1].set_title(f"Boxplot de {target_column}")
    axes[1].set_xlabel(target_column)

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close(fig)
    return str(file_path)


def plot_numeric_histograms(df: pd.DataFrame, numeric_columns: list[str], output_dir: str | Path) -> list[str]:
    """Guarda histogramas para cada variable numerica."""
    output = _ensure_output_dir(output_dir)
    saved_paths: list[str] = []

    for column in numeric_columns:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df[column], kde=True, ax=ax, color="#2a9d8f")
        ax.set_title(f"Histograma: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
        plt.tight_layout()
        file_path = output / f"hist_{column}.png"
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(file_path))

    return saved_paths


def plot_boxplots(df: pd.DataFrame, numeric_columns: list[str], output_dir: str | Path) -> list[str]:
    """Guarda boxplots para cada variable numerica."""
    output = _ensure_output_dir(output_dir)
    saved_paths: list[str] = []

    for column in numeric_columns:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(x=df[column], ax=ax, color="#f4a261")
        ax.set_title(f"Boxplot: {column}")
        ax.set_xlabel(column)
        plt.tight_layout()
        file_path = output / f"boxplot_{column}.png"
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(file_path))

    return saved_paths


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str | Path) -> str:
    """Guarda heatmap de correlaciones entre variables numericas."""
    file_path = _ensure_parent(output_path)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de Correlacion (Variables Numericas)")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close(fig)
    return str(file_path)


def plot_categorical_counts(
    df: pd.DataFrame, categorical_columns: list[str], output_dir: str | Path
) -> list[str]:
    """Guarda graficos de conteo para variables categoricas."""
    output = _ensure_output_dir(output_dir)
    saved_paths: list[str] = []

    for column in categorical_columns:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = df[column].value_counts(dropna=False).index
        sns.countplot(data=df, x=column, order=order, ax=ax, palette="Blues_d")
        ax.set_title(f"Conteo por categoria: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Cantidad")
        plt.xticks(rotation=20)
        plt.tight_layout()
        file_path = output / f"count_{column}.png"
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(file_path))

    return saved_paths


def plot_target_by_category(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
    output_dir: str | Path,
) -> list[str]:
    """Guarda boxplots del target segmentado por categorias."""
    output = _ensure_output_dir(output_dir)
    saved_paths: list[str] = []

    for column in categorical_columns:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.boxplot(data=df, x=column, y=target_column, ax=ax, color="#8ecae6")
        ax.set_title(f"{target_column} por {column}")
        ax.set_xlabel(column)
        ax.set_ylabel(target_column)
        plt.xticks(rotation=20)
        plt.tight_layout()
        file_path = output / f"target_by_{column}.png"
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(file_path))

    return saved_paths


def plot_scatter_numeric_vs_target(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str,
    output_dir: str | Path,
) -> list[str]:
    """Guarda scatterplots de variables numericas contra el target."""
    output = _ensure_output_dir(output_dir)
    saved_paths: list[str] = []

    for column in numeric_columns:
        if column not in df.columns or column == target_column:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.scatterplot(data=df, x=column, y=target_column, ax=ax, alpha=0.65, s=25)
        ax.set_title(f"{column} vs {target_column}")
        ax.set_xlabel(column)
        ax.set_ylabel(target_column)
        plt.tight_layout()
        file_path = output / f"scatter_{column}_vs_{target_column}.png"
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(file_path))

    return saved_paths


def plot_suspicious_times(df: pd.DataFrame, output_path: str | Path) -> str:
    """Guarda grafico de tiempos sospechosos en el target."""
    file_path = _ensure_parent(output_path)
    target_col = "target_tiempo_entrega"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    target_values = pd.to_numeric(df[target_col], errors="coerce")
    suspicious_mask = target_values < 10
    invalid_mask = target_values <= 0

    ax.scatter(df.index, target_values, alpha=0.45, s=18, label="Registros")
    ax.scatter(
        df.index[suspicious_mask],
        target_values[suspicious_mask],
        color="#e76f51",
        s=34,
        label="Tiempo < 10 min",
    )
    ax.scatter(
        df.index[invalid_mask],
        target_values[invalid_mask],
        color="#d62828",
        s=45,
        label="Tiempo <= 0 (invalido)",
    )
    ax.axhline(10, color="#f4a261", linestyle="--", linewidth=1.2, label="Umbral 10 min")
    ax.axhline(0, color="#6c757d", linestyle=":", linewidth=1.1)
    ax.set_title("Registros sospechosos en target_tiempo_entrega")
    ax.set_xlabel("Indice")
    ax.set_ylabel(target_col)
    ax.legend()

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close(fig)
    return str(file_path)
