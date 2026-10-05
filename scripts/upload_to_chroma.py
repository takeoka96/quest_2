import os
import numpy as np
import pandas as pd
import chromadb

CHROMA_HOST = "chroma-117244440892.asia-northeast1.run.app"
CHROMA_TOKEN = "3dcc7d8949be4f4d86a6c4d15da39ea9bae3fbb3dcc4b3b3b72c86067800eae2"

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=443,
    ssl=True,
    headers={"Authorization": f"Bearer {CHROMA_TOKEN}"}
)

collection = client.get_or_create_collection(
    name="dei_speeches",
    metadata={"hnsw:space": "cosine"}
)

if collection.count() == 0:
    df = pd.read_csv("data/speeches_clustered.csv").fillna("")
    embeddings = np.load("data/embeddings_round5.npy")
    
    batch_size = 500
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_emb = embeddings[i:i+batch_size]
        collection.add(
            ids=[str(idx) for idx in batch_df.index],
            embeddings=batch_emb.tolist(),
            documents=batch_df["speech_clean"].tolist(),
            metadatas=[{
                "speaker": str(row["speaker"]),
                "speakerGroup": str(row["speakerGroup"]),
                "date": str(row["date"]),
                "keyword": str(row["keyword"]),
                "cluster": int(row["cluster_r5"]) if row["cluster_r5"] != "" and row["cluster_r5"] != -1 else -1
            } for _, row in batch_df.iterrows()]
        )
        print(f"{i+len(batch_df)}/{len(df)}件投入完了")
    
    print(f"完了：{collection.count()}件")
else:
    print(f"既に{collection.count()}件入っています")
