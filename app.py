# ============================================================
# app.py
# DEI国民の声マップ - RAGアプリケーション
# 国会議事録のDEI関連発言をベースにした質問応答システム
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# データの読み込みとChromaDBへの登録（初回のみ）
# ============================================================
@st.cache_resource
def setup_chromadb():
    """ChromaDBにデータを登録する（初回のみ実行）"""
    df = pd.read_csv("data/speeches_clustered.csv")
    embeddings = np.load("data/embeddings_round5.npy")
    
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="dei_speeches",
        metadata={"hnsw:space": "cosine"}
    )
    
    if collection.count() == 0:
        batch_size = 500
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_emb = embeddings[i:i+batch_size]
            
            collection.add(
                ids=[str(idx) for idx in batch_df.index],
                embeddings=batch_emb.tolist(),
                documents=batch_df["speech_clean"].tolist(),
                metadatas=[{
                    "speaker"     : str(row["speaker"]),
                    "speakerGroup": str(row["speakerGroup"]),
                    "date"        : str(row["date"]),
                    "keyword"     : str(row["keyword"]),
                    "cluster"     : int(row["cluster_r5"]) if row["cluster_r5"] != -1 else -1
                } for _, row in batch_df.iterrows()]
            )
    
    return collection

@st.cache_resource
def load_model():
    """埋め込みモデルをロード"""
    return SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="DEI国民の声マップ",
    page_icon="🗾",
    layout="wide"
)

st.title("🗾 DEI国民の声マップ")
st.markdown("""
国会議事録のDEI関連発言（6,301件）をAIで分析したシステムです。
DEIに関する質問を入力すると、関連する国会議員の発言と考察を表示します。
""")

# 初期化
with st.spinner("データを読み込み中..."):
    collection = setup_chromadb()
    model = load_model()

st.success(f"✅ {collection.count()}件の発言データを読み込みました")

# 質問入力
st.markdown("---")
query = st.text_input(
    "DEIに関する質問を入力してください",
    placeholder="例：女性の理工系進出について国会ではどんな議論がされていますか？"
)

col1, col2 = st.columns([1, 4])
with col1:
    search_btn = st.button("🔍 検索", type="primary")
with col2:
    n_results = st.slider("表示件数", 3, 10, 5)

if search_btn and query:
    with st.spinner("関連発言を検索中..."):
        instruction = "Instruct: Retrieve semantically similar political speeches about DEI\nQuery: "
        query_embedding = model.encode(
            instruction + query,
            normalize_embeddings=True
        ).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    
    st.markdown("---")
    st.subheader("📋 関連する国会発言")
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        cluster = meta["cluster"]
        is_minority = cluster != -1 and cluster in [1,2,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]
        
        with st.expander(
            f"{'🔵 少数派' if is_minority else '🔴 多数派'} | "
            f"{meta['speaker']} ({meta['speakerGroup']}) | "
            f"{meta['date']} | キーワード：{meta['keyword']}"
        ):
            st.write(doc[:500])
    
    # Geminiによる考察
    st.markdown("---")
    st.subheader("🤖 AIによる考察")
    
    with st.spinner("考察を生成中..."):
        context = "\n\n".join([
            f"発言者：{meta['speaker']}（{meta['speakerGroup']}）\n{doc[:300]}"
            for doc, meta in zip(docs[:3], metas[:3])
        ])
        
        prompt = f"""
以下は日本の国会でのDEI（多様性・公平性・包括性）に関する発言です。

{context}

ユーザーの質問：{query}

上記の発言を踏まえて、日本のDEIに関する国会での議論の傾向と課題を
200字程度で簡潔に日本語で説明してください。
"""
        response = client_gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        st.write(response.text)

# フッター
st.markdown("---")
st.markdown("""
**データソース**: 国会会議録検索システム（国立国会図書館）  
**分析手法**: multilingual-e5-large-instruct + UMAP + HDBSCAN  
**シルエットスコア**: 0.7498
""")