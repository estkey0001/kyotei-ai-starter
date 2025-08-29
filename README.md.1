# 🚤 競艇AI予想システム - KYOTEI AI PREDICTION

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 概要

競艇AI予想システムは、機械学習アルゴリズムを活用した高精度な競艇レース予想システムです。複数のアルゴリズム（XGBoost、Random Forest、Gradient Boosting）を組み合わせることで、高い予想精度を実現しています。

## ✨ 主要機能

### 🤖 AI予想エンジン
- **複合AI予想**：XGBoost、Random Forest、Gradient Boostingの3つの高精度機械学習モデルを組み合わせ
- **リアルタイム予想**：選手データ、会場情報、気象条件を統合した総合的な予想
- **確率ベース予想**：各艇の勝率を百分率で表示

### 📊 データ分析機能
- **選手成績分析**：勝率、平均スタートタイム、モーター・ボート成績
- **会場特性分析**：24の全国競艇場の特徴を考慮
- **気象条件対応**：天候、風向き、風速、気温、湿度の影響を分析

### 🎨 ユーザーインターフェース
- **美しいUI**：Streamlitベースの直感的で使いやすいWebインターフェース
- **リアルタイム更新**：入力データの変更に応じて即座に予想を更新
- **レスポンシブデザイン**：PC・タブレット・スマートフォンに対応

### 📈 予想精度・実績
- **高精度AI予想**：複数モデルの組み合わせにより優秀な予想精度を実現
- **継続的学習**：新しいデータに基づくモデルの継続的な改善
- **統計的検証**：過去データに基づく予想精度の統計的検証

## 🛠️ 技術仕様

### 使用技術スタック
```
フロントエンド  : Streamlit
バックエンド    : Python 3.7+
機械学習        : Scikit-Learn, XGBoost
データ処理      : Pandas, NumPy
UI/UX          : Custom CSS + Streamlit Components
```

### 依存関係
```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

### システム要件
- **Python**: 3.7以上
- **メモリ**: 4GB以上推奨
- **ディスク**: 1GB以上の空き容量
- **ネットワーク**: インターネット接続（初回セットアップ時）

## 🚀 インストール・セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/estkey0001/kyotei-ai-starter.git
cd kyotei-ai-starter
```

### 2. 仮想環境の作成（推奨）
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

### 4. アプリケーションの起動
```bash
streamlit run kyotei_ai_complete_v14.py
```

### 5. ブラウザでアクセス
```
http://localhost:8501
```

## 💡 使用方法

### 基本的な予想手順

#### 1. 会場・気象情報の入力
- **会場選択**：24の競艇場から選択
- **天候情報**：晴れ、曇り、雨から選択
- **風向き・風速**：風向き（8方向）と風速（m/s）
- **気温・湿度**：現在の気象条件

#### 2. 選手情報の入力（1～6号艇）
各選手について以下を入力：
- **基本情報**：年齢、体重、支部
- **成績データ**：勝率、連対率、3連対率
- **スタート成績**：平均スタートタイム、出遅れ率
- **モーター・ボート成績**：2連対率

#### 3. AI予想の実行
「🚀 AI予想実行」ボタンをクリック

#### 4. 予想結果の確認
- **各艇の勝率**：百分率表示
- **予想順位**：1位から6位までの予想
- **総合評価**：モデル間の一致度

### 高度な使用方法

#### データの品質管理
- 入力データの妥当性チェック
- 欠損データの自動補完
- 異常値の検出と処理

#### 予想精度の向上
- 複数レースのデータを蓄積
- 継続的なモデル改善
- 統計的な検証と調整

## 🎯 データ構造

### 入力データ仕様

#### 選手データ
```python
racer_data = {
    'age': int,              # 年齢
    'weight': float,         # 体重 (kg)
    'branch': str,           # 支部
    'win_rate': float,       # 勝率 (%)
    'quinella_rate': float,  # 連対率 (%)
    'trio_rate': float,      # 3連対率 (%)
    'avg_start_time': float, # 平均ST (秒)
    'late_start_rate': float,# 出遅れ率 (%)
    'motor_quinella': float, # モーター2連対率 (%)
    'boat_quinella': float   # ボート2連対率 (%)
}
```

#### 会場・気象データ
```python
venue_weather = {
    'venue': str,            # 会場名
    'weather': str,          # 天候 (晴れ/曇り/雨)
    'wind_direction': str,   # 風向き (8方向)
    'wind_speed': float,     # 風速 (m/s)
    'temperature': float,    # 気温 (℃)
    'humidity': float        # 湿度 (%)
}
```

### 出力データ仕様

#### 予想結果
```python
prediction_result = {
    'boat_1_probability': float,  # 1号艇勝率 (%)
    'boat_2_probability': float,  # 2号艇勝率 (%)
    'boat_3_probability': float,  # 3号艇勝率 (%)
    'boat_4_probability': float,  # 4号艇勝率 (%)
    'boat_5_probability': float,  # 5号艇勝率 (%)
    'boat_6_probability': float,  # 6号艇勝率 (%)
    'predicted_ranking': list,    # 予想順位 [1-6]
    'model_confidence': float     # モデル信頼度 (%)
}
```

## 🔧 システム詳細仕様

### AI予想アルゴリズム

#### 1. XGBoost予想モデル
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

#### 2. Random Forest予想モデル
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    random_state=42
)
```

#### 3. Gradient Boosting予想モデル
```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

### 特徴量エンジニアリング

#### 数値特徴量
- 選手の年齢、体重
- 各種成績率（勝率、連対率、3連対率）
- スタート関連指標
- モーター・ボート成績

#### カテゴリ特徴量
- 会場（24競艇場）
- 支部（全国の競艇選手支部）
- 天候（晴れ、曇り、雨）
- 風向き（8方向）

### データ前処理
```python
# 数値データの標準化
scaler = StandardScaler()

# カテゴリデータのエンコーディング
label_encoder = LabelEncoder()
```

## 🎨 UI/UXデザイン

### カラーパレット
```css
Primary Blue    : #1E88E5
Secondary Purple: #667eea - #764ba2 (グラデーション)
Success Green   : #4CAF50
Warning Orange  : #FF9800
Error Red       : #F44336
Background Gray : #f8f9fa
```

### レスポンシブデザイン
- **デスクトップ**: フルレイアウト
- **タブレット**: 2カラム → 1カラム
- **スマートフォン**: 縦スクロール最適化

### アクセシビリティ
- コントラスト比 4.5:1 以上
- キーボードナビゲーション対応
- スクリーンリーダー対応

## 📊 予想精度・実績

### 検証結果（※過去データによる検証）
- **的中率**: 高精度な予想を実現
- **回収率**: 継続的な改善を実施
- **安定性**: 複数モデルにより安定した予想

### 統計的指標
```python
評価指標 = {
    'Mean Squared Error': float,
    'Mean Absolute Error': float,
    'R² Score': float,
    'Cross Validation Score': float
}
```

## 🔮 今後の拡張計画

### Phase 1: 短期改善（1-3ヶ月）
- [ ] 過去レースデータの蓄積機能
- [ ] 予想結果の履歴管理
- [ ] 的中率の統計表示
- [ ] CSV/Excelデータエクスポート

### Phase 2: 中期拡張（3-6ヶ月）
- [ ] リアルタイムデータ取得API連携
- [ ] 自動データ更新機能
- [ ] 高度な統計分析ダッシュボード
- [ ] 複数レース一括予想機能

### Phase 3: 長期発展（6-12ヶ月）
- [ ] ディープラーニングモデルの導入
- [ ] 画像解析による選手・ボート状態認識
- [ ] モバイルアプリ版の開発
- [ ] クラウドデプロイメント（AWS/GCP）

### 技術的拡張
- [ ] Docker化
- [ ] Kubernetes対応
- [ ] CI/CD パイプライン
- [ ] 自動テスト環境
- [ ] パフォーマンス最適化

## 🛡️ トラブルシューティング

### よくある問題と解決方法

#### 1. アプリケーションが起動しない
```bash
# 依存関係の再インストール
pip install --upgrade -r requirements.txt

# Streamlitの再インストール
pip uninstall streamlit
pip install streamlit
```

#### 2. 予想が表示されない
- 入力データの妥当性を確認
- 数値範囲が適切かチェック
- エラーメッセージを確認

#### 3. パフォーマンスが遅い
```python
# キャッシュのクリア
st.cache_resource.clear()

# メモリ使用量の最適化
import gc
gc.collect()
```

#### 4. データの異常値
- 勝率: 0-100%の範囲
- 年齢: 15-70歳の範囲
- 体重: 40-70kgの範囲

### エラーコードと対処法

| エラーコード | 説明 | 対処法 |
|------------|------|--------|
| ML001 | モデル学習エラー | データ形式を確認 |
| UI002 | 表示エラー | ブラウザキャッシュをクリア |
| DATA003 | データ異常値 | 入力範囲を確認 |

## 👥 開発者情報

### 開発者向けセットアップ

#### 開発環境
```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt

# プリコミットフックの設定
pre-commit install
```

#### テスト実行
```bash
# 単体テスト
pytest tests/

# カバレッジ測定
pytest --cov=.
```

#### コード品質
```bash
# コードフォーマット
black .

# リンティング
flake8 .

# 型チェック
mypy .
```

### 貢献方法
1. Issues で問題を報告
2. Feature Request で機能提案
3. Pull Request で改善提案
4. コードレビューに参加

### ライセンス
```
MIT License
Copyright (c) 2024 estkey0001
```

## 📞 サポート・お問い合わせ

- **Issues**: [GitHub Issues](https://github.com/estkey0001/kyotei-ai-starter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/estkey0001/kyotei-ai-starter/discussions)
- **Email**: estkey0001@example.com

## 🙏 謝辞

このプロジェクトは以下のオープンソースプロジェクトを使用しています：

- [Streamlit](https://streamlit.io/) - Webアプリケーションフレームワーク
- [Scikit-Learn](https://scikit-learn.org/) - 機械学習ライブラリ
- [XGBoost](https://xgboost.readthedocs.io/) - 勾配ブースティング
- [Pandas](https://pandas.pydata.org/) - データ分析ライブラリ
- [NumPy](https://numpy.org/) - 数値計算ライブラリ

---

**⚠️ 免責事項**: この予想システムは娯楽・研究目的で開発されました。実際の競艇投票は自己責任で行ってください。
**🎯 最終更新**: 2024年8月29日
