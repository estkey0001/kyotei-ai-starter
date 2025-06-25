import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 戸田競艇2024年ウルトラ高精度学習システム")

# データ読み込み
print("📊 データ読み込み中...")
df = pd.read_csv('data/coconala_2024/toda_2024.csv')
print(f"データ形状: {df.shape}")

# 高度な特徴量エンジニアリング
print("🔧 高度な特徴量エンジニアリング中...")

target_cols = ['finish_position_1', 'finish_position_2', 'finish_position_3', 
               'finish_position_4', 'finish_position_5', 'finish_position_6']

enhanced_features = []
y = []

for idx, row in df.iterrows():
    positions = [row[col] for col in target_cols if pd.notna(row[col])]
    if 1.0 in positions:
        winner = positions.index(1.0)
        y.append(winner)
        
        feature_row = []
        
        # 基本気象データ
        temp = float(row.get('temperature', 15)) if pd.notna(row.get('temperature')) else 15
        wind = float(row.get('wind_speed', 2)) if pd.notna(row.get('wind_speed')) else 2
        wave = float(row.get('wave_height', 0)) if pd.notna(row.get('wave_height')) else 0
        
        feature_row.extend([temp, wind, wave])
        
        # 各艇の詳細特徴量
        racer_features = []
        for i in range(1, 7):
            # 基本成績
            win_rate = float(row.get(f'win_rate_national_{i}', 5)) if pd.notna(row.get(f'win_rate_national_{i}')) else 5
            place_rate = float(row.get(f'place_rate_2_national_{i}', 30)) if pd.notna(row.get(f'place_rate_2_national_{i}')) else 30
            place_rate3 = float(row.get(f'place_rate_3_national_{i}', 50)) if pd.notna(row.get(f'place_rate_3_national_{i}')) else 50
            
            # 選手データ
            age = float(row.get(f'racer_age_{i}', 35)) if pd.notna(row.get(f'racer_age_{i}')) else 35
            weight = float(row.get(f'racer_weight_{i}', 52)) if pd.notna(row.get(f'racer_weight_{i}')) else 52
            
            # モーター・ボート
            motor_rate = float(row.get(f'motor_win_rate_{i}', 35)) if pd.notna(row.get(f'motor_win_rate_{i}')) else 35
            motor_place = float(row.get(f'motor_place_rate_3_{i}', 55)) if pd.notna(row.get(f'motor_place_rate_3_{i}')) else 55
            
            # スタート・展示
            start_timing = float(row.get(f'avg_start_timing_{i}', 0.15)) if pd.notna(row.get(f'avg_start_timing_{i}')) else 0.15
            exhibition_time = float(row.get(f'exhibition_time_{i}', 6.7)) if pd.notna(row.get(f'exhibition_time_{i}')) else 6.7
            
            # クラス効果
            racer_class = row.get(f'racer_class_{i}', 'B1')
            class_bonus = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}.get(racer_class, 2)
            
            # 特徴量統合
            racer_features.extend([
                win_rate, place_rate, place_rate3, age, weight,
                motor_rate, motor_place, start_timing, exhibition_time, class_bonus
            ])
        
        feature_row.extend(racer_features)
        
        # 相対特徴量（1号艇基準）
        if len(racer_features) >= 60:  # 6艇×10特徴量
            boat1_win = racer_features[0]  # 1号艇勝率
            for i in range(1, 6):  # 2-6号艇
                other_win = racer_features[i*10]  # 各艇勝率
                feature_row.append(boat1_win - other_win)  # 勝率差
        
        # オッズ情報（あれば）
        odds = float(row.get('win_odds', 2.5)) if pd.notna(row.get('win_odds')) else 2.5
        feature_row.append(odds)
        
        enhanced_features.append(feature_row)

X = np.array(enhanced_features, dtype=float)
y = np.array(y)

print(f"強化学習データ: {len(X)}レース, {X.shape[1]}特徴量")
print(f"クラス分布: {np.bincount(y)}")

# 欠損値最終処理
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# 特徴量標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データ分割（層化サンプリング）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("🤖 高精度アンサンブル学習実行中...")

# クラス重み計算（不均衡対応）
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 高性能モデル群
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000
    )
}

model_scores = {}
trained_models = {}

for name, model in models.items():
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        mean_score = cv_scores.mean()
        
        print(f"{name}: CV精度 = {mean_score:.4f} (±{cv_scores.std()*2:.4f})")
        
        model.fit(X_train, y_train)
        model_scores[name] = mean_score
        trained_models[name] = model
        
    except Exception as e:
        print(f"{name}: エラー - {str(e)}")

# アンサンブル予測
print("🎯 アンサンブル予測実行中...")
ensemble_preds = []
for model in trained_models.values():
    pred = model.predict_proba(X_test)
    ensemble_preds.append(pred)

# 重み付きアンサンブル
ensemble_avg = np.mean(ensemble_preds, axis=0)
final_pred = np.argmax(ensemble_avg, axis=1)

# 最終精度評価
final_accuracy = accuracy_score(y_test, final_pred)

print(f"\n🏆 最終アンサンブル精度: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")

# 基準との比較
base_accuracy = 0.823
if final_accuracy > base_accuracy:
    improvement = ((final_accuracy - base_accuracy) / base_accuracy * 100)
    print(f"🎯 精度向上: +{improvement:.1f}%")
else:
    decline = ((base_accuracy - final_accuracy) / base_accuracy * 100)
    print(f"⚠️ 精度: ベース比-{decline:.1f}% (改善要)")

print(f"\n📊 期待値分析:")
print(f"ランダム予測精度: {1/6:.3f} (16.7%)")
print(f"1号艇固定予測: {np.sum(y_test==0)/len(y_test):.3f}")
print(f"アンサンブル予測: {final_accuracy:.3f}")

# 詳細レポート
print("\n📋 詳細分類レポート:")
print(classification_report(y_test, final_pred, target_names=[f'{i+1}号艇' for i in range(6)]))

# ウルトラモデル保存
ultra_model_path = 'toda_2024_ultra_high_accuracy_ensemble.pkl'
joblib.dump({
    'models': trained_models,
    'imputer': imputer,
    'scaler': scaler,
    'ensemble_weights': [1/len(trained_models)] * len(trained_models)
}, ultra_model_path)

print(f"💾 ウルトラモデル保存完了: {ultra_model_path}")
print("🎉 ウルトラ高精度学習完了！")
