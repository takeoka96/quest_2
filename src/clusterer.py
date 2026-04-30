"""
clusterer.py
クラスタリングモジュール
最終採用手法：UMAP + HDBSCAN
"""
import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# 最終採用パラメータ（Round5で確定）
UMAP_PARAMS = {
    "n_components": 10,
    "n_neighbors" : 15,
    "min_dist"    : 0.0,
    "random_state": 42
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 50,
    "metric"          : "euclidean"
}


def reduce_dimensions(embeddings, params=None):
    """
    UMAPで次元削減する関数

    Parameters:
        embeddings: 埋め込みベクトル
        params    : UMAPパラメータ（Noneの場合はデフォルト使用）

    Returns:
        np.ndarray: 次元削減済みベクトル
    """
    if params is None:
        params = UMAP_PARAMS

    print(f"UMAPで次元削減中... {embeddings.shape} → {params['n_components']}次元")
    reducer = umap.UMAP(**params)
    embeddings_reduced = reducer.fit_transform(normalize(embeddings))
    print("次元削減完了")
    return embeddings_reduced


def cluster(embeddings_reduced, params=None):
    """
    HDBSCANでクラスタリングする関数

    Parameters:
        embeddings_reduced: 次元削減済みベクトル
        params            : HDBSCANパラメータ（Noneの場合はデフォルト使用）

    Returns:
        np.ndarray: クラスタラベル（-1はノイズ）
    """
    if params is None:
        params = HDBSCAN_PARAMS

    print("HDBSCANでクラスタリング中...")
    clusterer = hdbscan.HDBSCAN(**params)
    labels = clusterer.fit_predict(embeddings_reduced)

    # 結果サマリー
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    mask = labels != -1

    if mask.sum() > 1 and n_clusters > 1:
        score = silhouette_score(
            embeddings_reduced[mask],
            labels[mask],
            sample_size=1000,
            random_state=42
        )
    else:
        score = -1

    print(f"クラスタ数: {n_clusters}")
    print(f"ノイズ件数: {n_noise}件")
    print(f"シルエットスコア: {score:.4f}")

    return labels