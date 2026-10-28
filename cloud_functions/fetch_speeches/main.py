import os
import requests
import functions_framework
from google.cloud import bigquery
from datetime import datetime, timedelta

KEYWORDS = ["多様性", "ジェンダー", "女性活躍", "障害者雇用", "外国人労働者", "ダイバーシティ", "LGBT"]

def fetch_speeches(keyword, from_date, until_date):
    url = "https://kokkai.ndl.go.jp/api/speech"
    speeches = []
    start_record = 1

    while True:
        params = {
            "any": keyword,
            "from": from_date,
            "until": until_date,
            "maximumRecords": 100,
            "startRecord": start_record,
            "recordPacking": "json"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            break
        data = response.json()
        records = data.get("speechRecord", [])
        if not records:
            break
        for r in records:
            speeches.append({
                "keyword": keyword,
                "date": r.get("date"),
                "nameOfHouse": r.get("nameOfHouse"),
                "nameOfMeeting": r.get("nameOfMeeting"),
                "speaker": r.get("speaker"),
                "speakerGroup": r.get("speakerGroup"),
                "speech": r.get("speech"),
                "speech_clean": r.get("speech", "")[:1000],
                "cluster_r5": -1
            })
        start_record += 100

    return speeches

@functions_framework.http
def main(request):
    bq = bigquery.Client()
    table_id = "dei-voice-map.dei_dataset.speeches"
    
    # （GCテスト用）
    # from_date = "2024-01-01"
    # until_date = "2024-03-31"
    # 本番用
    from_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    until_date = datetime.today().strftime("%Y-%m-%d")
    
    all_speeches = []
    for keyword in KEYWORDS:
        all_speeches.extend(fetch_speeches(keyword, from_date, until_date))
    
    if all_speeches:
        # BigQueryに追記(GC)
        errors = bq.insert_rows_json(table_id, all_speeches)
        if errors:
            return f"errors: {errors}", 500
        
        # ChromaDBにも追加（GC）
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        CHROMA_HOST = "chroma-117244440892.asia-northeast1.run.app"
        CHROMA_TOKEN = os.environ.get("CHROMA_API_TOKEN", "")
        
        chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=443,
            ssl=True,
            headers={"Authorization": f"Bearer {CHROMA_TOKEN}"}
        )
        collection = chroma_client.get_or_create_collection(
            name="dei_speeches",
            metadata={"hnsw:space": "cosine"}
        )
        
        model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        instruction = "Instruct: Retrieve semantically similar political speeches about DEI\nQuery: "
        
        ids = [f"new_{i}_{datetime.today().strftime('%Y%m%d')}" for i in range(len(all_speeches))]
        embeddings = model.encode(
            [instruction + s["speech_clean"] for s in all_speeches],
            normalize_embeddings=True
        ).tolist()
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[s["speech_clean"] for s in all_speeches],
            metadatas=[{
                "speaker": str(s.get("speaker", "")),
                "speakerGroup": str(s.get("speakerGroup", "")),
                "date": str(s.get("date", "")),
                "keyword": str(s.get("keyword", "")),
                "cluster": -1
            } for s in all_speeches]
        )
    
    return f"inserted {len(all_speeches)} rows", 200
