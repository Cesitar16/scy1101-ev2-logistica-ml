import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


RANDOM_STATE = 42


def split_features_for_clustering(df, target_column="target_tiempo_entrega"):
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
    else:
        X = df.copy()

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return X, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    return preprocessor


def evaluate_kmeans_range(X_processed, k_min=2, k_max=8, random_state=RANDOM_STATE):
    results = []

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(X_processed)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(X_processed, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X_processed, labels),
            "davies_bouldin_score": davies_bouldin_score(X_processed, labels)
        })

    return pd.DataFrame(results)


def train_kmeans(X_processed, n_clusters=3, random_state=RANDOM_STATE):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(X_processed)

    return model, labels


def create_cluster_profile(df_original, labels, target_column="target_tiempo_entrega"):
    df_clustered = df_original.copy()
    df_clustered["cluster"] = labels

    numeric_columns = df_clustered.select_dtypes(include=["int64", "float64"]).columns.tolist()

    profile = df_clustered.groupby("cluster")[numeric_columns].mean().round(2)
    profile["cantidad_registros"] = df_clustered.groupby("cluster").size()

    if target_column in df_clustered.columns:
        profile["tiempo_entrega_promedio"] = (
            df_clustered.groupby("cluster")[target_column].mean().round(2)
        )

    return df_clustered, profile


def plot_elbow(metrics_df, output_path="results/plots/elbow_kmeans.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k"], metrics_df["inertia"], marker="o")
    plt.title("Método del codo - K-Means")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_silhouette(metrics_df, output_path="results/plots/silhouette_kmeans.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k"], metrics_df["silhouette_score"], marker="o")
    plt.title("Silhouette Score por número de clusters")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_clusters_pca(X_processed, labels, output_path="results/plots/clusters_pca.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_processed)

    plot_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(8, 6))

    for cluster in sorted(plot_df["cluster"].unique()):
        subset = plot_df[plot_df["cluster"] == cluster]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.7)

    plt.title("Visualización de clusters con PCA")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_clustering_outputs(metrics_df, profile, model, output_metrics_path, output_profile_path, output_model_path):
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_profile_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    metrics_df.to_csv(output_metrics_path, index=False)
    profile.to_csv(output_profile_path)
    joblib.dump(model, output_model_path)