
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# 実データ学習済みモデル読み込み
@st.cache_resource
def load_practical_model():
    """2024年戸田実データ学習済みモデル"""
    if os.path.exists('practical_kyotei_model.pkl'):
        return joblib.load('practical_kyotei_model.pkl')
    return None

def predict_race(racer_data):
    """実データ学習済みモデルで予想"""
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

def main():
    st.title("🏁 戸田競艇AI予想システム（2024年実データ学習版）")
    
    # サイドバー情報
    st.sidebar.header("📊 システム情報")
    st.sidebar.metric("学習データ", "2,346レース（実データ）")
    st.sidebar.metric("学習期間", "2024年1-12月戸田全レース")
    st.sidebar.metric("予想精度", "44.3%")
    st.sidebar.metric("モデル", "RandomForest最適化済み")
    
    # モデル読み込み確認
    model_data = load_practical_model()
    if model_data:
        st.sidebar.success("✅ 実データ学習済みモデル読み込み完了")
    else:
        st.sidebar.error("❌ モデル読み込み失敗")
        st.error("実データ学習済みモデルが見つかりません")
        return
    
    st.header("🎯 レース予想入力")
    
    # 入力フォーム
    racer_inputs = []
    
    for i in range(1, 7):
        st.subheader(f"🚤 {i}号艇")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            win_rate = st.number_input(
                f"勝率", min_value=0.0, max_value=100.0, 
                value=5.0, step=0.1, key=f"win_{i}"
            )
        
        with col2:
            racer_class = st.selectbox(
                f"級別", ["A1", "A2", "B1", "B2"], 
                index=2, key=f"class_{i}"
            )
        
        with col3:
            motor_rate = st.number_input(
                f"モーター勝率", min_value=0.0, max_value=100.0, 
                value=35.0, step=0.1, key=f"motor_{i}"
            )
        
        racer_inputs.append((win_rate, racer_class, motor_rate))
    
    if st.button("🔮 実データAI予想実行", type="primary"):
        st.header("📈 予想結果（2024年実データ学習済み）")
        
        predicted_winner, probabilities = predict_race(racer_inputs)
        
        if predicted_winner and probabilities is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🏆 予想1着", f"{predicted_winner}号艇")
                
                # 確率表示
                sorted_indices = np.argsort(probabilities)[::-1]
                st.write("**予想順位:**")
                for idx, boat_idx in enumerate(sorted_indices):
                    boat_num = boat_idx + 1
                    prob = probabilities[boat_idx]
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"][idx]
                    st.write(f"{rank_emoji} {boat_num}号艇: {prob*100:.1f}%")
            
            with col2:
                # グラフ表示
                chart_data = pd.DataFrame({
                    '艇番': [f"{i+1}号艇" for i in range(6)],
                    '勝率予測': probabilities
                })
                st.bar_chart(chart_data.set_index('艇番'))
            
            # note記事生成
            st.subheader("📝 note配信記事生成")
            
            max_prob = max(probabilities)
            confidence_level = "高" if max_prob > 0.5 else "中" if max_prob > 0.3 else "低"
            
            article = f"""# 🏁 戸田競艇AI予想 - {datetime.now().strftime('%Y年%m月%d日')}

## 🎯 本日の予想
**本命**: {predicted_winner}号艇 ({max_prob*100:.1f}%)
**信頼度**: {confidence_level}

## 📊 各艇分析（実データ学習済みAI）

"""
            for i, (prob, (win_rate, racer_class, motor_rate)) in enumerate(zip(probabilities, racer_inputs)):
                boat_num = i + 1
                analysis = "有力" if prob > 0.3 else "注意" if prob > 0.15 else "厳しい"
                
                article += f"""### {boat_num}号艇 - {analysis} ({prob*100:.1f}%)
- 勝率: {win_rate}%（級別: {racer_class}）
- モーター: {motor_rate}%
- AI評価: {prob*100:.1f}%

"""
            
            article += f"""## 🤖 AI分析総評
2024年戸田競艇場全レース（2,346レース）で学習したAIモデルによる予想です。
{predicted_winner}号艇を本命として、信頼度{confidence_level}で推奨します。

**投資判定**: {"推奨" if max_prob > 0.5 else "慎重" if max_prob > 0.3 else "見送り"}

---
*この予想は過去データに基づく参考情報です。投資は自己責任でお願いします。*
"""
            
            st.text_area("生成されたnote記事", article, height=400)
            
            # ダウンロードボタン
            st.download_button(
                label="📥 記事をダウンロード",
                data=article,
                file_name=f"kyotei_prediction_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
