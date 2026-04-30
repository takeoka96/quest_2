"""
preprocessor.py
テキストの前処理モジュール
"""
import re
import pandas as pd


# 除外キーワード（DEI文脈以外の発言が多いため）
EXCLUDE_KEYWORDS = ["公平", "包括"]


def clean_text(text):
    """
    発言テキストをクリーニングする関数

    処理内容：
    1. 発言者名を除去（例：「○山田太郎君　」）
    2. 改行・空白を正規化
    3. 5,000文字以上は冒頭1,000文字に切り詰め

    Parameters:
        text: 発言テキスト

    Returns:
        str: クリーニング済みテキスト
    """
    # 発言者名を除去（○で始まり全角スペースで終わるパターン）
    text = re.sub(r'○[^　]+　', '', text)

    # 改行・連続空白を半角スペースに統一
    text = re.sub(r'[\r\n\s]+', ' ', text)

    # 前後の空白を除去
    text = text.strip()

    # 5,000文字以上は冒頭1,000文字に切り詰め
    if len(text) > 5000:
        text = text[:1000]

    return text


def preprocess(df):
    """
    DataFrameに前処理を適用する関数

    Parameters:
        df: speeches_raw.csvを読み込んだDataFrame

    Returns:
        DataFrame: 前処理済みDataFrame
    """
    # 除外キーワードのフィルタリング
    df = df[~df["keyword"].str.split("|").str[0].isin(EXCLUDE_KEYWORDS)].copy()
    df = df.reset_index(drop=True)

    # テキストクリーニング
    df["speech_clean"] = df["speech"].apply(clean_text)

    print(f"前処理後のデータ数: {len(df)}件")
    return df