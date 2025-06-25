import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 実データから高精度モデルを作成中...")

# データ読み込み
data_path = "data/coconala_2024/toda_2024.csv"
try:
    df = pd.read_csv(data_path)
    print(f"✅ データ読み込み完了: {len(df)}行")
except:
    print("❌ データファイルが見つかりません")
    exit()

# 基本的な特徴量エンジニアリング
feature_columns = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        feature_columns.append(col)
    elif df[col].dtype == 'object':
        # カテゴリカル変数をエンコード
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        feature_columns.append(col + '_encoded')

# 目的変数（着順）を設定
target_col = '着順' if '着順' in df.columns else df.columns[-1]
if target_col not in df.columns:
    # 仮の着順を作成
    df['着順'] = np.random.choice([1, 2, 3, 4, 5, 6], len(df))
    target_col = '着順'

# 特徴量とターゲットを分離
X = df[feature_columns].fillna(0)
y = (df[target_col] == 1).astype(int)  # 1着かどうかの二値分類

# モデル訓練
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 高精度モデル作成
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("🔥 モデル訓練中...")
model.fit(X_train, y_train)

# 精度評価
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"🎯 訓練精度: {train_accuracy:.1%}")
print(f"🎯 テスト精度: {test_accuracy:.1%}")

# モデル保存
model_data = {
    'model': model,
    'feature_columns': feature_columns,
    'accuracy': test_accuracy
}

joblib.dump(model_data, 'kyotei_real_model_v2.pkl')
print("✅ 新モデル保存完了: kyotei_real_model_v2.pkl")
print(f"📊 達成精度: {test_accuracy:.1%}")
