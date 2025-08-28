# train_toda_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# CSV読み込み
df = pd.read_csv('~/toda_2024.csv')

# 使用する特徴量
features = [
    'actual_course_1', 'actual_course_2', 'actual_course_3',
    'exhibition_time_1', 'exhibition_time_2', 'exhibition_time_3',
    'motor_advantage_1', 'motor_advantage_2', 'motor_advantage_3',
    'wind_speed', 'temperature'
]

# ターゲット（1号艇の着順）
target = 'finish_position_1'

# 欠損値除去
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 着順を「1着=1」「2着=2」... に変換（int型）
y = y.astype(int)

# 学習
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 保存
joblib.dump(model, 'model_toda.pkl')
print("✅ モデル保存完了: model_toda.pkl")
