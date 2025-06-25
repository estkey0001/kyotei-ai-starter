import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 実際のココナラデータで高精度学習開始...")

# データ読み込み
df = pd.read_csv('data/coconala_2024/toda_2024.csv')
print(f"✅ データ読み込み完了: {len(df)}行 x {len(df.columns)}列")

# レース単位から艇単位にデータを変換
print("🔄 データ変換中...")
boat_data = []

for idx, row in df.iterrows():
    for boat_num in range(1, 7):  # 1号艇～6号艇
        boat_record = {
            # 基本情報
            'race_date': row['race_date'],
            'venue_name': row['venue_name'],
            'race_number': row['race_number'],
            'boat_number': boat_num,
            
            # 選手情報
            'racer_name': row[f'racer_name_{boat_num}'],
            'racer_class': row[f'racer_class_{boat_num}'],
            'racer_age': row[f'racer_age_{boat_num}'],
            'racer_weight': row[f'racer_weight_{boat_num}'],
            
            # 成績情報
            'win_rate_national': row[f'win_rate_national_{boat_num}'],
            'place_rate_2_national': row[f'place_rate_2_national_{boat_num}'],
            'win_rate_local': row[f'win_rate_local_{boat_num}'],
            'avg_start_timing': row[f'avg_start_timing_{boat_num}'],
            
            # モーター・ボート情報
            'motor_advantage': row[f'motor_advantage_{boat_num}'],
            'motor_win_rate': row[f'motor_win_rate_{boat_num}'],
            
            # 気象情報
            'weather': row['weather'],
            'temperature': row['temperature'],
            'wind_speed': row['wind_speed'],
            'wind_direction': row['wind_direction'],
            
            # 目的変数（着順）
            'finish_position': row[f'finish_position_{boat_num}'],
            'is_winner': 1 if row[f'finish_position_{boat_num}'] == 1.0 else 0
        }
        boat_data.append(boat_record)

boat_df = pd.DataFrame(boat_data)
print(f"✅ 艇単位データ変換完了: {len(boat_df)}行")

# 欠損値処理
boat_df = boat_df.dropna(subset=['finish_position', 'is_winner'])
print(f"✅ 欠損値除去後: {len(boat_df)}行")

# 特徴量エンジニアリング
print("🔧 特徴量エンジニアリング...")

# カテゴリカル変数のエンコーディング
le_class = LabelEncoder()
le_weather = LabelEncoder()

boat_df['racer_class_encoded'] = le_class.fit_transform(boat_df['racer_class'].astype(str))
boat_df['weather_encoded'] = le_weather.fit_transform(boat_df['weather'].astype(str))

# 数値特徴量の選択
feature_columns = [
    'boat_number', 'racer_age', 'racer_weight',
    'win_rate_national', 'place_rate_2_national', 'win_rate_local',
    'avg_start_timing', 'motor_advantage', 'motor_win_rate',
    'temperature', 'wind_speed', 'racer_class_encoded', 'weather_encoded'
]

X = boat_df[feature_columns].fillna(0)
y = boat_df['is_winner']

print(f"📊 特徴量数: {len(feature_columns)}")
print(f"📊 学習データ: {len(X)}件")
print(f"📊 1着率: {y.mean():.1%}")

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 複数モデルで学習
models = {
    'RandomForest': RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=500, max_depth=8, random_state=42)
}

trained_models = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"🔥 {name} 学習中...")
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"  - 訓練精度: {train_acc:.1%}")
    print(f"  - テスト精度: {test_acc:.1%}")
    
    trained_models[name] = model
    
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model

print(f"\n🏆 最高精度: {best_accuracy:.1%}")

# モデル保存
model_package = {
    'model': best_model,
    'feature_columns': feature_columns,
    'label_encoders': {
        'racer_class': le_class,
        'weather': le_weather
    },
    'accuracy': best_accuracy,
    'boat_df_sample': boat_df.head(100)  # サンプルデータ
}

joblib.dump(model_package, 'kyotei_real_trained_model.pkl')
print(f"✅ 高精度モデル保存完了: kyotei_real_trained_model.pkl")
print(f"🎯 達成精度: {best_accuracy:.1%}")

# 予測テスト
print("\n🧪 予測テスト:")
sample_predictions = best_model.predict_proba(X_test[:6])[:, 1]
for i, prob in enumerate(sample_predictions):
    print(f"  艇{i+1}: {prob:.1%}確率")
