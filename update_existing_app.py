import shutil
import os

print("🔄 既存app.pyを新モデル対応に更新中...")

# バックアップ確認
if os.path.exists('app_original_backup.py'):
    print("✅ バックアップ作成済み: app_original_backup.py")
else:
    shutil.copy('app.py', 'app_original_backup.py')
    print("✅ バックアップ作成: app_original_backup.py")

# app.pyの先頭に新モデル統合
new_model_integration = '''
# =================== 新モデル統合 (2024年実データ学習済み) ===================
import joblib
import numpy as np
import pandas as pd
import os

@st.cache_resource
def load_practical_model():
    """2024年実データ学習済みモデル読み込み"""
    if os.path.exists('practical_kyotei_model.pkl'):
        return joblib.load('practical_kyotei_model.pkl')
    return None

def predict_with_practical_model(racer_data):
    """実用モデルでの予測
    racer_data: [(win_rate, class, motor_rate), ...] 6艇分
    """
    model_data = load_practical_model()
    if not model_data:
        return None, None
    
    feature_vector = []
    for win_rate, racer_class, motor_rate in racer_data:
        class_val = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}.get(racer_class, 2)
        feature_vector.extend([win_rate, class_val, motor_rate])
    
    X_pred = np.array([feature_vector])
    X_pred = model_data['imputer'].transform(X_pred)
    X_pred = model_data['scaler'].transform(X_pred)
    
    probabilities = model_data['model'].predict_proba(X_pred)[0]
    prediction = model_data['model'].predict(X_pred)[0]
    
    return prediction + 1, probabilities

# =================== ここまで新モデル統合 ===================

'''

# 既存app.pyを読み込み
with open('app.py', 'r', encoding='utf-8') as f:
    original_content = f.read()

# import部分の後に新モデル統合を挿入
import_section = "import streamlit as st"
if import_section in original_content:
    # import行の後に挿入
    lines = original_content.split('\n')
    new_lines = []
    import_added = False
    
    for line in lines:
        new_lines.append(line)
        if import_section in line and not import_added:
            new_lines.extend(new_model_integration.split('\n'))
            import_added = True
    
    updated_content = '\n'.join(new_lines)
    
    # 更新されたapp.pyを保存
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ app.py更新完了")
    print("📝 変更内容:")
    print("  - 2024年実データ学習済みモデル統合")
    print("  - predict_with_practical_model関数追加")
    print("  - load_practical_model関数追加")
    print("  - 既存機能は保持")
    
else:
    print("❌ app.py形式が想定と異なります")

print("🎉 既存app.py更新完了！")
