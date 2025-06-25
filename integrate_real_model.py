print("🔧 GitHub UI + 実データモデル統合中...")

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 実データモデル統合コード
real_model_code = '''
import joblib
import os

# 実データ学習済みモデル読み込み
@st.cache_resource
def load_practical_model():
    try:
        if os.path.exists('practical_kyotei_model.pkl'):
            return joblib.load('practical_kyotei_model.pkl')
    except:
        pass
    return None

def predict_with_real_data(race_data, venue="戸田"):
    """実データ学習済み予想（戸田のみ）"""
    if venue == "戸田":
        model_data = load_practical_model()
        if model_data:
            try:
                features = []
                for i in range(1, 7):
                    win_rate = race_data.get(f'win_rate_{i}', 5.0)
                    racer_class = race_data.get(f'racer_class_{i}', 'B1')
                    motor_rate = race_data.get(f'motor_rate_{i}', 35.0)
                    
                    class_val = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}.get(racer_class, 2)
                    features.extend([win_rate, class_val, motor_rate])
                
                X_pred = np.array([features])
                X_pred = model_data['imputer'].transform(X_pred)
                X_pred = model_data['scaler'].transform(X_pred)
                
                probabilities = model_data['model'].predict_proba(X_pred)[0]
                return probabilities, True  # True = 実データ使用
            except:
                pass
    
    return None, False  # False = 元のサンプル予想使用

'''

# import文の後に追加
import_pos = content.find('import streamlit as st')
if import_pos != -1:
    line_end = content.find('\n', import_pos)
    while content[line_end + 1] == '\n':  # 空行をスキップ
        line_end = content.find('\n', line_end + 1)
    content = content[:line_end] + '\n' + real_model_code + content[line_end:]

# KyoteiAIRealtimeSystemのpredict_race_winnerメソッド拡張
method_pattern = 'def predict_race_winner(self, race_data, venue="戸田"):'
method_pos = content.find(method_pattern)
if method_pos != -1:
    method_start = content.find('\n', method_pos) + 1
    
    # 元のメソッド内容を探す
    indent_count = 0
    search_pos = method_start
    while search_pos < len(content) and content[search_pos] == ' ':
        indent_count += 1
        search_pos += 1
    
    # メソッド開始後に実データ予想を挿入
    real_prediction_insert = f'''        # 🎯 実データ学習済み予想を最優先実行
        real_probs, is_real_data = predict_with_real_data(race_data, venue)
        if is_real_data:
            winner = np.argmax(real_probs) + 1
            return {{
                'winner': winner,
                'probabilities': real_probs.tolist(),
                'confidence': float(max(real_probs)),
                'reasoning': f"🎯 2024年戸田実データ学習済みAI予想（精度44.3%）\\n学習レース数: 2,346レース\\n使用モデル: RandomForest最適化",
                'method': 'real_data_learned_2024_toda'
            }}
        
        # 📊 フォールバック: 元のサンプル予想システム
'''
    
    content = content[:method_start] + real_prediction_insert + content[method_start:]

# サイドバーに実データ情報追加
sidebar_pos = content.find('selected_venue = st.sidebar.selectbox')
if sidebar_pos != -1:
    # その行の次の行を見つける
    line_end = content.find('\n', sidebar_pos)
    next_line = content.find('\n', line_end + 1)
    
    sidebar_info = '''
    # 🎯 実データ学習済み情報表示
    if selected_venue == "戸田":
        st.sidebar.success("🎯 2024年実データ学習済み")
        st.sidebar.metric("学習精度", "44.3%")
        st.sidebar.metric("学習レース数", "2,346レース")
        st.sidebar.metric("学習期間", "2024年1-12月")
        st.sidebar.info("RandomForest最適化済み")
    else:
        st.sidebar.warning("📊 サンプルデータベース")
        st.sidebar.text("統計ベース予想")
'''
    
    content = content[:next_line] + sidebar_info + content[next_line:]

# システム状況表示部分も更新
status_pattern = '"system_accuracy":'
status_pos = content.find(status_pattern)
if status_pos != -1:
    # system_accuracyの値を実データ対応に変更
    line_start = content.rfind('\n', 0, status_pos) + 1
    line_end = content.find('\n', status_pos)
    old_line = content[line_start:line_end]
    
    if 'venue_accuracy' in old_line:
        new_line = old_line.replace('venue_accuracy.get(selected_venue, "未学習")', 
                                  '"44.3%" if selected_venue == "戸田" else "サンプルベース"')
        content = content[:line_start] + new_line + content[line_end:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 実データモデル統合完了")
print("📝 統合内容:")
print("  - 戸田選択時: practical_kyotei_model.pkl使用（44.3%精度）")
print("  - その他会場: 元のサンプル予想")
print("  - サイドバー: 実データ学習済み情報表示")
print("  - システム状況: 精度表示更新")
