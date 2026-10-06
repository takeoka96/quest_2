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
#from sentence_transformers import SentenceTransformer # ← この1行削除(GC)
from google.cloud import bigquery #(GC)
from google import genai
from dotenv import load_dotenv
import streamlit.components.v1 as components #(GC GA4)

# 環境変数の読み込み（Hugging Face SpacesではSecretsから自動で環境変数に設定される）
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# データの読み込みとChromaDBへの登録（初回のみ）
# ============================================================
@st.cache_resource
def setup_chromadb():
    """ChromaDBにデータを登録する（初回のみ実行）"""
    # GC用に追記：外部ChromaDBに接続してreturn
    CHROMA_HOST = os.environ.get("CHROMA_HOST", None)
    if CHROMA_HOST:
        # GCP：外部ChromaDB（データ投入済み、起動時のCSV読み込み不要）
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=443,
            ssl=True,
            headers={"Authorization": f"Bearer {os.environ.get('CHROMA_API_TOKEN', '')}"}
        )
        return client.get_or_create_collection(
            name="dei_speeches",
            metadata={"hnsw:space": "cosine"}
        )

    # Hugging Face：ローカルChromaDB（CSVから読み込み）
    # ※GCP移行前の元実装。Hugging Face Spacesでは引き続きこちらを使用
    # データ読み込み
    # df = pd.read_csv("speeches_clustered.csv")  # 元のパス（Hugging Face用）
    # embeddings = np.load("embeddings_round5.npy")

    # 変更後(GC)
    df = pd.read_csv("data/speeches_clustered.csv")
    embeddings = np.load("data/embeddings_round5.npy")
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="dei_speeches",
        metadata={"hnsw:space": "cosine"}
    )
    # 既に登録済みの場合はスキップ
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

    # 既に登録済みの場合はスキップ(学習時残骸)
    # if collection.count() == 0:
    #     batch_size = 500
    #     for i in range(0, len(df), batch_size):
    #         batch_df = df.iloc[i:i+batch_size]
    #         batch_emb = embeddings[i:i+batch_size]
            
    #         collection.add(
    #             ids=[str(idx) for idx in batch_df.index],
    #             embeddings=batch_emb.tolist(),
    #             documents=batch_df["speech_clean"].tolist(),
    #             metadatas=[{
    #                 "speaker"     : str(row["speaker"]),
    #                 "speakerGroup": str(row["speakerGroup"]),
    #                 "date"        : str(row["date"]),
    #                 "keyword"     : str(row["keyword"]),
    #                 "cluster"     : int(row["cluster_r5"]) if row["cluster_r5"] != -1 else -1
    #             } for _, row in batch_df.iterrows()]
    #         )
    
    # return collection

@st.cache_resource
def load_model():
    """埋め込みモデルをロード"""
    from sentence_transformers import SentenceTransformer  # ← この1行を追加(GC)
    return SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="DEI国民の声マップ",
    page_icon="🗾",
    layout="wide"
)

# GA4トラッキングコード（GC追記）
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-RR85JMV8J8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-RR85JMV8J8');
</script>
""", unsafe_allow_html=True)

st.title("🗾 DEI国民の声マップ")

# 初期化
with st.spinner("データを読み込み中..."):
    collection = setup_chromadb()
    #model = load_model() # 削除(GC)

st.markdown(f"""
国会議事録のDEI関連発言（{collection.count()}件）をAIで分析したシステムです。
DEIに関する質問を入力すると、関連する国会議員の発言と考察を表示します。
""")

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
        model = load_model()  # 追加(GC)
        # model = load_model() を起動時ではなく検索時に呼ぶ（@st.cache_resourceなので2回目以降はキャッシュ）
        instruction = "Instruct: Retrieve semantically similar political speeches about DEI\nQuery: "
        query_embedding = model.encode(
            instruction + query,
            normalize_embeddings=True
        ).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # クエリログをBigQueryに保存(GC)
        bq_client = bigquery.Client()
        table_id = "dei-voice-map.dei_dataset.query_logs"
        rows = [{"query": query, "timestamp": pd.Timestamp.now().isoformat()}]
        bq_client.insert_rows_json(table_id, rows)
    
    st.markdown("---")
    st.subheader("📋 関連する国会発言")
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        cluster = meta["cluster"]
        is_minority = cluster != -1 and cluster in [1, 6, 8, 9, 10, 11, 12, 14, 17, 18, 19, 21]
        
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