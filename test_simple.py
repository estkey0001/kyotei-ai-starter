import streamlit as st

st.title("🏁 シンプルテスト")
st.write("このメッセージが見えればOK")
st.success("正常動作中")

venue = st.selectbox("テスト選択", ["戸田", "江戸川"])
if venue == "戸田":
    st.info("実データ学習済み")

if st.button("テストボタン"):
    st.write("ボタンが動作しています")
