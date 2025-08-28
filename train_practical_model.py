import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

print("🎯 実用的高精度競艇予想システム")

df = pd.read_csv('data/coconala_2024/toda_2024.csv')
print(f"データ: {df.shape}")

# シンプル特徴量選択（最重要項目のみ）
features = []
targets = []

target_cols = ['finish_position_1', 'finish_position_2', 'finish_position_3', 
               'finish_position_4', 'finish_position_5', 'finish_position_6']

for idx, row in df.iterrows():
    positions = [row[col] for col in target_cols if pd.notna(row[col])]
    if 1.0 in positions:
        winner = positions.index(1.0)
        targets.append(winner)
        
        # 核心特徴量のみ
        feature_row = []
        
        for i in range(1, 7):
            # 最重要3要素
            win_rate = float(row.get(f'win_rate_national_{i}', 5)) if pd.notna(row.get(f'win_rate_national_{i}')) else 5
            class_val = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}.get(row.get(f'racer_class_{i}', 'B1'), 2)
            motor_rate = float(row.get(f'motor_win_rate_{i}', 35)) if pd.notna(row.get(f'motor_win_rate_{i}')) else 35
            
            feature_row.extend([win_rate, class_val, motor_rate])
        
        features.append(feature_row)

X = np.array(features)
y = np.array(targets)

print(f"最適化データ: {len(X)}レース, {X.shape[1]}特徴量")

# 前処理
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 学習
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 最適化RandomForest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20, 
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f"\n🏆 最終精度: {accuracy:.4f} ({accuracy*100:.1f}%)")

# 実用的な予測関数作成
def predict_race(racer_data):
    """
    6艇のレース予測
    racer_data: [(win_rate, class, motor_rate), ...] 6艇分
    """
    feature_vector = []
    for win_rate, racer_class, motor_rate in racer_data:
        class_val = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}.get(racer_class, 2)
        feature_vector.extend([win_rate, class_val, motor_rate])
    
    X_pred = np.array([feature_vector])
    X_pred = imputer.transform(X_pred)
    X_pred = scaler.transform(X_pred)
    
    probabilities = model.predict_proba(X_pred)[0]
    prediction = model.predict(X_pred)[0]
    
    return prediction + 1, probabilities  # 1-6号艇に変換

# テスト予測
print("\n🧪 予測テスト:")
test_data = [
    (7.2, 'A1', 45.0),  # 1号艇: 高勝率A1級
    (5.8, 'A2', 38.0),  # 2号艇: 中勝率A2級
    (4.2, 'B1', 32.0),  # 3号艇: 低勝率B1級
    (3.8, 'B1', 28.0),  # 4号艇
    (2.9, 'B2', 25.0),  # 5号艇
    (2.1, 'B2', 22.0),  # 6号艇
]

predicted_winner, probs = predict_race(test_data)
print(f"予想1着: {predicted_winner}号艇")
print("各艇確率:")
for i, prob in enumerate(probs):
    print(f"  {i+1}号艇: {prob:.3f} ({prob*100:.1f}%)")

# 保存
joblib.dump({
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'predict_function': predict_race
}, 'practical_kyotei_model.pkl')

print(f"\n💾 実用モデル保存: practical_kyotei_model.pkl")
print("🎉 実用システム完成！")
