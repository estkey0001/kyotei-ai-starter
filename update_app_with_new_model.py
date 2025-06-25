import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# 新モデル読み込み
@st.cache_resource
def load_new_model():
    if os.path.exists('practical_kyotei_model.pkl'):
        return joblib.load('practical_kyotei_model.pkl')
    return None

def predict_with_new_model(racer_data):
    """新モデルでの予測"""
    model_data = load_new_model()
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

def main():
    st.title("🏁 戸田競艇AI予想システム（2024年実データ学習版）")
    
    st.sidebar.header("📊 システム情報")
    st.sidebar.metric("学習データ", "2,346レース")
    st.sidebar.metric("学習期間", "2024年1-12月")
    st.sidebar.metric("現在精度", "44.3%")
    st.sidebar.metric("特徴量", "18個（最適化済み）")
    
    st.header("🎯 レース予想入力")
    
    # 入力フォーム
    racer_inputs = []
    
    for i in range(1, 7):
        st.subheader(f"🚤 {i}号艇")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            win_rate = st.number_input(
                f"勝率 ({i}号艇)", 
                min_value=0.0, max_value=100.0, value=5.0, step=0.1,
                key=f"win_rate_{i}"
            )
        
        with col2:
            racer_class = st.selectbox(
                f"級別 ({i}号艇)",
                ["A1", "A2", "B1", "B2"],
                index=2,
                key=f"class_{i}"
            )
        
        with col3:
            motor_rate = st.number_input(
                f"モーター勝率 ({i}号艇)",
                min_value=0.0, max_value=100.0, value=35.0, step=0.1,
                key=f"motor_{i}"
            )
        
        racer_inputs.append((win_rate, racer_class, motor_rate))
    
    if st.button("🔮 AI予想実行", type="primary"):
        st.header("📈 予想結果")
        
        # 新モデル予測
        predicted_winner, probabilities = predict_with_new_model(racer_inputs)
        
        if predicted_winner and probabilities is not None:
            # 予想結果表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🏆 予想1着", f"{predicted_winner}号艇")
                
                # 上位3艇表示
                sorted_indices = np.argsort(probabilities)[::-1]
                st.write("**上位予想:**")
                for idx, boat_idx in enumerate(sorted_indices[:3]):
                    boat_num = boat_idx + 1
                    prob = probabilities[boat_idx]
                    rank_emoji = ["🥇", "🥈", "🥉"][idx]
                    st.write(f"{rank_emoji} {boat_num}号艇: {prob:.3f} ({prob*100:.1f}%)")
            
            with col2:
                # 全艇確率グラフ
                chart_data = pd.DataFrame({
                    '艇番': [f"{i+1}号艇" for i in range(6)],
                    '勝率予測': probabilities
                })
                st.bar_chart(chart_data.set_index('艇番'))
            
            # 投資判定
            st.subheader("💰 投資判定")
            max_prob = max(probabilities)
            confidence = max_prob
            
            if confidence > 0.6:
                st.success(f"🔥 高信頼度 ({confidence*100:.1f}%) - 投資推奨")
            elif confidence > 0.4:
                st.warning(f"⚠️ 中信頼度 ({confidence*100:.1f}%) - 慎重投資")
            else:
                st.error(f"❌ 低信頼度 ({confidence*100:.1f}%) - 投資非推奨")
            
            # note記事生成
            st.subheader("📝 note記事生成")
            
            article = f"""# 🏁 戸田競艇AI予想 - {datetime.now().strftime('%Y年%m月%d日')}

## 🎯 予想結果
- **本命**: {predicted_winner}号艇
- **信頼度**: {max_prob*100:.1f}%

## 📊 各艇評価

"""
            for i, prob in enumerate(probabilities):
                boat_num = i + 1
                win_rate = racer_inputs[i][0]
                racer_class = racer_inputs[i][1]
                motor_rate = racer_inputs[i][2]
                
                article += f"""### {boat_num}号艇 ({prob*100:.1f}%)
- 勝率: {win_rate}%
- 級別: {racer_class}
- モーター: {motor_rate}%

"""
            
            article += f"""## 🤖 AI分析コメント
戸田競艇場2024年実データ学習済みモデル（精度44.3%）による予想です。
{predicted_winner}号艇が最有力で、信頼度は{max_prob*100:.1f}%です。

---
*このAI予想は参考情報です。投資は自己責任でお願いします。*
"""
            
            st.text_area("生成記事", article, height=300)
        
        else:
            st.error("モデル読み込みエラー")

if __name__ == "__main__":
    main()
