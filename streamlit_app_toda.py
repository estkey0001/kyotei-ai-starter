# streamlit_app_toda.py
import streamlit as st
import pandas as pd
import joblib

# モデル読み込み
model = joblib.load('/home/estkeyieldz_ltd/kyotei-ai-starter/model_toda.pkl')

st.title("🚤 戸田競艇 AI予想システム（1着予測）")
st.markdown("2024年の実データに基づき、1号艇の着順をAIが予測します。")

# 入力フォーム
st.header("🔧 レース条件を入力してください")

actual_course_1 = st.selectbox("1号艇 進入コース", [1, 2, 3, 4, 5, 6])
actual_course_2 = st.selectbox("2号艇 進入コース", [1, 2, 3, 4, 5, 6])
actual_course_3 = st.selectbox("3号艇 進入コース", [1, 2, 3, 4, 5, 6])

exhibition_time_1 = st.number_input("1号艇 展示タイム", value=6.80, step=0.01)
exhibition_time_2 = st.number_input("2号艇 展示タイム", value=6.85, step=0.01)
exhibition_time_3 = st.number_input("3号艇 展示タイム", value=6.90, step=0.01)

motor_advantage_1 = st.number_input("1号艇 モーター評価", value=0.0, step=0.1)
motor_advantage_2 = st.number_input("2号艇 モーター評価", value=0.0, step=0.1)
motor_advantage_3 = st.number_input("3号艇 モーター評価", value=0.0, step=0.1)

wind_speed = st.number_input("風速（m/s）", value=2.0, step=0.1)
temperature = st.number_input("気温（℃）", value=10.0, step=0.5)

if st.button("🎯 1着を予測する"):
    input_df = pd.DataFrame([{
        'actual_course_1': actual_course_1,
        'actual_course_2': actual_course_2,
        'actual_course_3': actual_course_3,
        'exhibition_time_1': exhibition_time_1,
        'exhibition_time_2': exhibition_time_2,
        'exhibition_time_3': exhibition_time_3,
        'motor_advantage_1': motor_advantage_1,
        'motor_advantage_2': motor_advantage_2,
        'motor_advantage_3': motor_advantage_3,
        'wind_speed': wind_speed,
        'temperature': temperature,
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"🏆 AI予想：**{prediction}着** になる確率が最も高いです！")
