"""
embedder.py
テキスト埋め込みモジュール
最終採用モデル：multilingual-e5-large-instruct
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


# 最終採用モデル
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# instructモデル用プレフィックス
INSTRUCTION = "Instruct: Retrieve semantically similar political speeches about DEI\nQuery: "


def load_model():
    """
    埋め込みモデルをロードする関数

    Returns:
        SentenceTransformer: ロード済みモデル
    """
    print(f"モデルをロード中: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("ロード完了")
    return model


def embed_texts(texts, model, batch_size=16):
    """
    テキストリストを埋め込みベクトルに変換する関数

    Parameters:
        texts     : テキストのリスト
        model     : SentenceTransformerモデル
        batch_size: バッチサイズ

    Returns:
        np.ndarray: 正規化済み埋め込みベクトル (n_samples, 1024)
    """
    # instructモデル用プレフィックスを付与
    texts_with_prefix = [INSTRUCTION + t for t in texts]

    print(f"埋め込みを計算中... ({len(texts)}件)")
    embeddings = model.encode(
        texts_with_prefix,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(f"埋め込み完了: {embeddings.shape}")
    return embeddings