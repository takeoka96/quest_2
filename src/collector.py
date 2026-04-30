"""
collector.py
国会会議録APIからDEI関連発言を収集するモジュール
"""
import requests
import time
import pandas as pd


# DEI関連キーワード一覧
KEYWORDS = [
    "ダイバーシティ",
    "多様性",
    "女性活躍",
    "ジェンダー",
    "障害者雇用",
    "外国人労働者",
    "DEI"
]

# 取得期間（半年ごとに分割）
PERIODS = [
    ("2020-01-01", "2020-06-30"),
    ("2020-07-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
]


def fetch_speeches(keyword, from_date, until_date, max_records=100):
    """
    国会会議録APIから発言データを取得する関数

    Parameters:
        keyword    : 検索キーワード
        from_date  : 取得開始日 (例: "2020-01-01")
        until_date : 取得終了日 (例: "2020-06-30")
        max_records: 1回のリクエストで取得する最大件数（上限100）

    Returns:
        list: 発言データのリスト
    """
    url = "https://kokkai.ndl.go.jp/api/speech"
    speeches = []
    start_record = 1

    while True:
        params = {
            "any": keyword,
            "from": from_date,
            "until": until_date,
            "maximumRecords": max_records,
            "startRecord": start_record,
            "recordPacking": "json"
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"エラー: {keyword} {from_date}〜{until_date} status={response.status_code}")
            break

        data = response.json()
        records = data.get("speechRecord", [])

        if not records:
            break

        for r in records:
            speeches.append({
                "keyword"      : keyword,
                "date"         : r.get("date"),
                "nameOfHouse"  : r.get("nameOfHouse"),
                "nameOfMeeting": r.get("nameOfMeeting"),
                "speaker"      : r.get("speaker"),
                "speakerGroup" : r.get("speakerGroup"),
                "speech"       : r.get("speech"),
            })

        next_pos = data.get("nextRecordPosition")
        if not next_pos:
            break

        start_record = next_pos
        time.sleep(0.5)

    return speeches


def collect_all_speeches():
    """
    全キーワード×全期間でデータを収集してDataFrameを返す関数
    """
    all_speeches = []

    for keyword in KEYWORDS:
        for from_date, until_date in PERIODS:
            speeches = fetch_speeches(keyword, from_date, until_date)
            all_speeches.extend(speeches)
            print(f"[取得完了] {keyword} | {from_date}〜{until_date} | {len(speeches)}件")
            time.sleep(0.5)

    df = pd.DataFrame(all_speeches)

    # 同一発言テキストをグループ化しキーワードを結合
    df = df.groupby("speech", as_index=False).agg({
        "keyword"      : lambda x: "|".join(sorted(set(x))),
        "date"         : "first",
        "nameOfHouse"  : "first",
        "nameOfMeeting": "first",
        "speaker"      : "first",
        "speakerGroup" : "first",
    })

    print(f"\n総取得件数: {len(df)}件")
    return df