# 課題2：DEIに関する国民の声を集めて可視化するアルゴリズムを実装せよ

## プロジェクト概要

国会会議録検索システムAPIからDEI関連発言を収集し、機械学習によるクラスタリングで
意見グループを自動分類・可視化する。多数決的な判断軸ではなく、少数派の意見も
公平に認識できるアルゴリズムを実装した。

## 成果物

- `notebooks/01_collect_data.ipynb` - データ収集
- `notebooks/02_eda.ipynb` - EDA・前処理方針決定
- `notebooks/03_modeling.ipynb` - 埋め込み・クラスタリング（Round1〜6）
- `notebooks/04_visualization_shap.ipynb` - 可視化・SHAP/LIME分析
- `app.py` - RAGアプリ（Streamlit）
- `docs/要件定義.md` - 要件定義書
- `docs/仕様書.md` - 仕様書
- `docs/presentation2.pdf` - プレゼンテーション資料

## 分析の流れ

1. 国会会議録APIからDEI関連発言を収集（6,301件）
2. EDAによるデータ品質確認・前処理方針決定
3. テキスト埋め込み（Round1〜6で試行錯誤・最終スコア0.7498）
4. UMAP+HDBSCANでクラスタリング（23クラスタ）
5. 可視化・SHAP/LIME分析
6. RAGアプリ実装・デプロイ

## 環境構築

```bash
# リポジトリのクローン
git clone [リポジトリURL]
cd quest_2

# 仮想環境の作成（Python 3.11推奨）
python -m venv .venv311
source .venv311/bin/activate

# ライブラリのインストール
pip install -r requirements.txt
```

## デモアプリ

URLを開くだけで動作します（環境構築不要）：
https://huggingface.co/spaces/t-okada/dei-voice-map

- ホスティング: Hugging Face Spaces（無料）
- フレームワーク: Streamlit

## 実験管理
Weights & Biases（W&B）で全Roundの実験結果を記録・可視化しています。
https://wandb.ai/omegatribe369-free/dei-clustering

## 使用データ

- 国会会議録検索システム（国立国会図書館）
  - 取得期間：2020年〜2025年
  - 取得件数：6,301件（前処理後）

## 主な分析結果

- DEI関連発言を23クラスタに自動分類（シルエットスコア0.7498）
- 少数派クラスタにはジェンダーギャップ・LGBT・国際連携など専門的トピックが集中
- 多数派では「多様性」が農業・環境文脈で使われるケースが多くDEI本来の議論が埋もれている
- HDBSCANにより少数派の声を独立したグループとして検出することに成功

## 注意事項

- `data/`フォルダはGitの管理外（`.gitignore`参照）
- RAGアプリ実行時はGemini APIキーが必要（`.env`に設定）
- データ収集セルは初回のみ実行（約10分）
- 埋め込み計算セルは初回のみ実行（約20分）