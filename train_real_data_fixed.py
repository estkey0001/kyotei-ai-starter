import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 戸田競艇2024年実データ学習システム開始（修正版）")

# データ読み込み
print("📊 データ読み込み中...")
df = pd.read_csv('data/coconala_2024/toda_2024.csv')
print(f"データ形状: {df.shape}")
print(f"データ期間: {df['race_date'].min()} - {df['race_date'].max()}")

# データ前処理
print("🔧 前処理実行中...")
target_cols = ['finish_position_1', 'finish_position_2', 'finish_position_3', 
               'finish_position_4', 'finish_position_5', 'finish_position_6']

y = []
features = []

for idx, row in df.iterrows():
    positions = [row[col] for col in target_cols if pd.notna(row[col])]
    if 1.0 in positions:
        winner = positions.index(1.0)
        y.append(winner)
        
        # 特徴量選択（欠損値を0で埋める）
        feature_row = []
        
        # 気象データ
        feature_row.extend([
            float(row.get('temperature', 0)) if pd.notna(row.get('temperature', 0)) else 0,
            float(row.get('wind_speed', 0)) if pd.notna(row.get('wind_speed', 0)) else 0,
            float(row.get('wave_height', 0)) if pd.notna(row.get('wave_height', 0)) else 0
        ])
        
        # 各艇の特徴量（欠損値処理）
        for i in range(1, 7):
            win_rate = row.get(f'win_rate_national_{i}', 0)
            place_rate = row.get(f'place_rate_2_national_{i}', 0)
            age = row.get(f'racer_age_{i}', 0)
            motor_rate = row.get(f'motor_win_rate_{i}', 0)
            start_timing = row.get(f'avg_start_timing_{i}', 0)
            
            feature_row.extend([
                float(win_rate) if pd.notna(win_rate) else 0,
                float(place_rate) if pd.notna(place_rate) else 0,
                float(age) if pd.notna(age) else 0,
                float(motor_rate) if pd.notna(motor_rate) else 0,
                float(start_timing) if pd.notna(start_timing) else 0
            ])
        
        features.append(feature_row)

X = np.array(features, dtype=float)
y = np.array(y)

print(f"学習データ: {len(X)}レース, {X.shape[1]}特徴量")
print(f"クラス分布: {np.bincount(y)}")

# NaN値最終チェック・処理
print("🔍 欠損値処理中...")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
print(f"欠損値処理後: NaN数 = {np.isnan(X).sum()}")

# 学習実行
print("🤖 機械学習モデル訓練中...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 欠損値対応モデル使用
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():
    try:
        # 交差検証
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_score = cv_scores.mean()
        
        print(f"{name}: CV精度 = {mean_score:.4f} (±{cv_scores.std()*2:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
    except Exception as e:
        print(f"{name}: エラー - {str(e)}")

# 最良モデルで学習
print(f"🏆 最良モデル選択: {best_model.__class__.__name__}")
best_model.fit(X_train, y_train)

# テスト精度評価
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"✅ テスト精度: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

# 基準精度との比較
base_accuracy = 0.823
if test_accuracy > base_accuracy:
    improvement = ((test_accuracy - base_accuracy) / base_accuracy * 100)
    print(f"🎯 精度向上: +{improvement:.1f}%")
else:
    decline = ((base_accuracy - test_accuracy) / base_accuracy * 100)
    print(f"⚠️ 精度低下: -{decline:.1f}%")

# 詳細レポート
print("\n📋 詳細分類レポート:")
print(classification_report(y_test, test_pred, target_names=[f'{i+1}号艇' for i in range(6)]))

# モデル保存
model_path = 'toda_2024_high_accuracy_model_fixed.pkl'
joblib.dump({
    'model': best_model,
    'imputer': imputer,
    'feature_names': ['temperature', 'wind_speed', 'wave_height'] + 
                    [f'{feat}_{i}' for i in range(1, 7) 
                     for feat in ['win_rate', 'place_rate', 'age', 'motor_rate', 'start_timing']]
}, model_path)
print(f"💾 モデル保存完了: {model_path}")

print("🎉 学習完了！")
