import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 戸田競艇2024年実データ学習システム開始")

# データ読み込み
print("📊 データ読み込み中...")
df = pd.read_csv('data/coconala_2024/toda_2024.csv')
print(f"データ形状: {df.shape}")
print(f"データ期間: {df['race_date'].min()} - {df['race_date'].max()}")

# データ前処理
print("🔧 前処理実行中...")
# 予測ターゲット: 1着予測
target_cols = ['finish_position_1', 'finish_position_2', 'finish_position_3', 
               'finish_position_4', 'finish_position_5', 'finish_position_6']

# 各レースで1着を特定
y = []
features = []

for idx, row in df.iterrows():
    positions = [row[col] for col in target_cols if pd.notna(row[col])]
    if 1.0 in positions:
        winner = positions.index(1.0)  # 0-5の艇番
        y.append(winner)
        
        # 特徴量選択
        feature_row = []
        
        # 気象データ
        feature_row.extend([
            row.get('temperature', 0),
            row.get('wind_speed', 0),
            row.get('wave_height', 0)
        ])
        
        # 各艇の特徴量
        for i in range(1, 7):
            feature_row.extend([
                row.get(f'win_rate_national_{i}', 0),
                row.get(f'place_rate_2_national_{i}', 0),
                row.get(f'racer_age_{i}', 0),
                row.get(f'motor_win_rate_{i}', 0),
                row.get(f'avg_start_timing_{i}', 0)
            ])
        
        features.append(feature_row)

X = np.array(features)
y = np.array(y)

print(f"学習データ: {len(X)}レース, {X.shape[1]}特徴量")
print(f"クラス分布: {np.bincount(y)}")

# 学習実行
print("🤖 機械学習モデル訓練中...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# アンサンブル学習
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    # 交差検証
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = cv_scores.mean()
    
    print(f"{name}: CV精度 = {mean_score:.4f} (±{cv_scores.std()*2:.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model

# 最良モデルで学習
print(f"🏆 最良モデル選択: {best_model.__class__.__name__}")
best_model.fit(X_train, y_train)

# テスト精度評価
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"✅ テスト精度: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print(f"🎯 精度向上: {((test_accuracy - 0.823) / 0.823 * 100):+.1f}%")

# モデル保存
model_path = 'toda_2024_high_accuracy_model.pkl'
joblib.dump(best_model, model_path)
print(f"💾 モデル保存完了: {model_path}")

print("🎉 学習完了！")
