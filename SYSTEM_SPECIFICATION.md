# 競艇AI予想システム - システム仕様書

## バージョン: v14.0
## 作成日: 2024年8月29日
## 文書タイプ: 技術仕様書

---

## 📋 システム概要

### プロジェクト名
**競艇AI予想システム (KYOTEI AI PREDICTION)**

### システム目的
機械学習アルゴリズムを活用した高精度な競艇レース結果予想システムの構築

### 対象ユーザー
- 競艇ファン・愛好家
- データサイエンティスト
- 研究者・学生

---

## 🏗️ システムアーキテクチャ

### アーキテクチャパターン
```
Web Application (Streamlit)
├── Presentation Layer (UI/UX)
├── Business Logic Layer (AI Models)
├── Data Access Layer (Pandas/NumPy)
└── Infrastructure Layer (Python Runtime)
```

### 技術スタック詳細

#### フロントエンド
- **Framework**: Streamlit 1.0+
- **UI Components**: Custom CSS + Streamlit Widgets
- **Responsive**: CSS Media Queries
- **Browser Support**: Chrome, Firefox, Safari, Edge

#### バックエンド
- **Language**: Python 3.7+
- **Runtime**: CPython
- **Web Server**: Streamlit Server
- **Session Management**: Streamlit Session State

#### 機械学習・データ処理
```python
# Core ML Libraries
XGBoost >= 1.5.0          # Gradient Boosting
Scikit-Learn >= 1.0.0     # ML Framework
Pandas >= 1.3.0           # Data Processing  
NumPy >= 1.21.0           # Numerical Computing

# ML Models Used
- XGBRegressor
- RandomForestRegressor
- GradientBoostingRegressor
```

---

## 🔧 詳細技術仕様

### データフロー
```
1. User Input (UI) → 2. Data Validation → 3. Feature Engineering 
      ↓                      ↓                    ↓
8. UI Display ← 7. Results ← 6. Model Ensemble ← 5. Preprocessing ← 4. ML Pipeline
```

### 機械学習パイプライン

#### 1. データ前処理
```python
# 数値特徴量の標準化
StandardScaler().fit_transform(numerical_features)

# カテゴリ特徴量のエンコーディング
LabelEncoder().fit_transform(categorical_features)

# 欠損値処理
fillna(method='forward')  # 前方補完
```

#### 2. 特徴量設計
```python
# 入力特徴量 (計38次元)
features = [
    # 選手基本情報 (6×3 = 18次元)
    'racer_age', 'racer_weight', 'racer_branch',

    # 選手成績 (6×4 = 24次元) 
    'win_rate', 'quinella_rate', 'trio_rate', 'avg_start_time',

    # 機材成績 (6×2 = 12次元)
    'motor_quinella', 'boat_quinella',

    # 環境条件 (5次元)
    'venue', 'weather', 'wind_direction', 'wind_speed', 'temperature'
]
```

#### 3. モデルアンサンブル
```python
# アンサンブル手法: 平均化
def ensemble_predict(X):
    pred_xgb = xgb_model.predict(X)
    pred_rf = rf_model.predict(X) 
    pred_gb = gb_model.predict(X)

    # 重み付き平均 (等重み)
    return (pred_xgb + pred_rf + pred_gb) / 3
```

---

## 📊 データ仕様

### 入力データスキーマ

#### 選手データ (racer_data)
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| age | int | 15-70 | 選手年齢 |
| weight | float | 40.0-70.0 | 選手体重(kg) |
| branch | string | enum | 所属支部 |
| win_rate | float | 0.0-100.0 | 勝率(%) |
| quinella_rate | float | 0.0-100.0 | 連対率(%) |
| trio_rate | float | 0.0-100.0 | 3連対率(%) |
| avg_start_time | float | 0.00-2.00 | 平均ST(秒) |
| late_start_rate | float | 0.0-100.0 | 出遅れ率(%) |
| motor_quinella | float | 0.0-100.0 | モーター2連対率(%) |
| boat_quinella | float | 0.0-100.0 | ボート2連対率(%) |

#### 環境データ (venue_weather)
| Field | Type | Options | Description |
|-------|------|---------|-------------|
| venue | string | 24会場 | 競艇場名 |
| weather | string | 晴れ/曇り/雨 | 天候 |
| wind_direction | string | 8方向 | 風向き |
| wind_speed | float | 0.0-15.0 | 風速(m/s) |
| temperature | float | -10.0-40.0 | 気温(℃) |
| humidity | float | 0.0-100.0 | 湿度(%) |

### 出力データスキーマ

#### 予想結果 (prediction_result)
```python
{
    "boat_1_probability": 25.4,  # 1号艇勝率(%)
    "boat_2_probability": 18.7,  # 2号艇勝率(%)  
    "boat_3_probability": 16.2,  # 3号艇勝率(%)
    "boat_4_probability": 14.8,  # 4号艇勝率(%)
    "boat_5_probability": 12.6,  # 5号艇勝率(%)
    "boat_6_probability": 12.3,  # 6号艇勝率(%)
    "predicted_ranking": [1,2,3,4,5,6],  # 予想順位
    "model_confidence": 87.2,    # モデル信頼度(%)
    "ensemble_variance": 0.034   # アンサンブル分散
}
```

---

## 🎨 UI/UX仕様

### デザインシステム

#### カラーパレット
```css
/* Primary Colors */
--primary-blue: #1E88E5;
--primary-dark: #1565C0;

/* Secondary Colors */  
--secondary-purple: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
--secondary-light: #E3F2FD;

/* Status Colors */
--success: #4CAF50;
--warning: #FF9800; 
--error: #F44336;
--info: #2196F3;

/* Neutral Colors */
--background: #f8f9fa;
--surface: #ffffff;
--text-primary: #212121;
--text-secondary: #757575;
```

#### タイポグラフィ
```css
/* Font Hierarchy */
.main-header { font-size: 3rem; font-weight: bold; }
.sub-header { font-size: 1.5rem; font-weight: 500; }
.section-title { font-size: 1.25rem; font-weight: 600; }
.body-text { font-size: 1rem; font-weight: 400; }
.caption { font-size: 0.875rem; font-weight: 400; }
```

#### コンポーネント設計
```python
# Streamlit Components Mapping
st.set_page_config()     # Page Configuration
st.markdown()            # Custom CSS Injection
st.sidebar               # Navigation Sidebar
st.columns()             # Grid Layout System
st.form()                # Form Container
st.button()              # Action Buttons
st.selectbox()           # Dropdown Selection
st.number_input()        # Numeric Input
st.text_input()          # Text Input
st.metric()              # KPI Display
st.balloons()            # Success Animation
```

---

## 🔒 セキュリティ仕様

### データ保護
```python
# 入力データ検証
def validate_input(data):
    # 型チェック
    assert isinstance(data['age'], int)
    assert 15 <= data['age'] <= 70

    # 範囲チェック  
    assert 0.0 <= data['win_rate'] <= 100.0

    # 不正文字列チェック
    assert not any(char in data['branch'] for char in ['<', '>', '&'])

# XSS対策
st.markdown(html.escape(user_input))

# SQLインジェクション対策 (該当なし - DBアクセスなし)
```

### プライバシー保護
- 個人情報の収集なし
- セッション情報の暗号化
- ログの匿名化

---

## ⚡ パフォーマンス仕様

### レスポンス時間要件
```python
# Target Performance Metrics
Page_Load_Time      <= 2秒    # 初回ページロード
Prediction_Time     <= 3秒    # AI予想実行時間
Input_Response_Time <= 0.5秒  # 入力レスポンス
Model_Load_Time     <= 5秒    # モデル初期化時間
```

### リソース使用量
```python
# Resource Requirements  
CPU_Usage     <= 70%    # CPU使用率上限
Memory_Usage  <= 2GB    # メモリ使用量上限
Disk_IO       <= 50MB/s # ディスクIO上限
Network_IO    <= 10MB/s # ネットワークIO上限
```

### スケーラビリティ
```python
# Concurrent Users Support
Max_Concurrent_Users = 100      # 最大同時ユーザー数
Session_Timeout     = 1800      # セッションタイムアウト(秒)
Cache_Size          = 100MB     # キャッシュサイズ
```

---

## 🧪 テスト仕様

### テスト戦略
```python
# Test Pyramid
Unit_Tests        # 関数・メソッド単位
Integration_Tests # コンポーネント間連携  
E2E_Tests        # ユーザーシナリオ全体
Performance_Tests # 負荷・ストレステスト
Security_Tests   # セキュリティテスト
```

### テストカバレッジ目標
```python
Unit_Test_Coverage      >= 80%   # 単体テストカバレッジ
Integration_Coverage    >= 70%   # 結合テストカバレッジ  
E2E_Coverage           >= 90%   # E2Eテストカバレッジ
Critical_Path_Coverage  = 100%   # 重要経路カバレッジ
```

### テスト環境
```python
# Test Environment Setup
Test_Data_Size     = 1000    # テストデータ件数
Mock_API_Response  = True    # APIモック使用
Test_Database     = SQLite   # テスト用DB
CI_Pipeline       = GitHub_Actions  # CI/CDパイプライン
```

---

## 📈 監視・ログ仕様

### ログレベル
```python
# Logging Configuration
import logging

logging_config = {
    'ERROR':   'システムエラー、例外',
    'WARNING': 'パフォーマンス警告、異常値検出', 
    'INFO':    'ユーザーアクション、予想実行',
    'DEBUG':   '詳細デバッグ情報（開発時のみ）'
}
```

### メトリクス収集
```python
# Key Metrics to Monitor
User_Metrics = {
    'daily_active_users': int,
    'prediction_requests': int, 
    'average_session_duration': float,
    'bounce_rate': float
}

System_Metrics = {
    'response_time_95th': float,
    'error_rate': float,
    'cpu_utilization': float,
    'memory_utilization': float
}

Business_Metrics = {
    'prediction_accuracy': float,
    'model_confidence_avg': float,
    'feature_importance_drift': float
}
```

---

## 🔄 デプロイメント仕様

### 環境構成
```yaml
# Environment Configuration
environments:
  development:
    url: "http://localhost:8501"
    debug: true
    log_level: "DEBUG"

  staging:
    url: "https://staging-kyotei-ai.com"
    debug: false  
    log_level: "INFO"

  production:
    url: "https://kyotei-ai.com"
    debug: false
    log_level: "WARNING"
    ssl: true
```

### CI/CD パイプライン
```yaml
# GitHub Actions Workflow
name: Deploy Kyotei AI
on: [push, pull_request]

jobs:
  test:
    - run: pytest tests/
    - run: flake8 .
    - run: black --check .

  deploy:
    - run: docker build .
    - run: docker push registry/kyotei-ai
    - run: kubectl apply -f k8s/
```

---

## 📚 参考資料

### 技術文書
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

### 競艇関連資料
- 日本モーターボート競走会公式サイト
- 競艇統計データベース
- 各競艇場公式データ

---

**文書管理情報**
- 作成者: システム開発チーム
- 最終更新: 2024年8月29日
- バージョン: 1.0
- 承認者: プロジェクトマネージャー
