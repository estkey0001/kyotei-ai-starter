# 🔧 トラブルシューティングガイド

## 🚨 よくある問題と解決方法

### 1. アプリケーション起動エラー

#### 問題: `ModuleNotFoundError`
```bash
ModuleNotFoundError: No module named 'streamlit'
```

**解決方法:**
```bash
pip install -r requirements.txt
```

#### 問題: `ImportError: cannot import name`
```bash
ImportError: cannot import name 'XGBRegressor'
```

**解決方法:**
```bash
pip install --upgrade xgboost scikit-learn
```

### 2. 予想結果が表示されない

#### 症状
- ボタンを押しても予想が表示されない
- エラーメッセージも出ない

**確認項目:**
- [ ] 全選手データが入力済み
- [ ] 数値が適切な範囲内（勝率: 0-100%等）
- [ ] ブラウザのコンソールエラー確認

**解決方法:**
```bash
# ブラウザキャッシュクリア
Ctrl + Shift + R (Windows/Linux)
Cmd + Shift + R (Mac)

# アプリ再起動
streamlit run kyotei_ai_complete_v14.py
```

### 3. パフォーマンス問題

#### 症状: 動作が重い
**解決方法:**
```python
# Streamlitキャッシュクリア
st.cache_resource.clear()
```

**システムリソース確認:**
```bash
# メモリ使用量確認
free -m

# CPU使用率確認  
top
```

### 4. データ入力エラー

#### 問題: 数値入力で異常値
**有効範囲:**
- 年齢: 15-70歳
- 体重: 40.0-70.0kg  
- 勝率: 0.0-100.0%
- 風速: 0.0-15.0m/s

### 5. 環境固有の問題

#### Windows環境
```bash
# 文字化け対策
set PYTHONIOENCODING=utf-8
streamlit run kyotei_ai_complete_v14.py
```

#### Mac環境  
```bash
# Python version確認
python3 --version
pip3 install -r requirements.txt
```

---

## 📞 サポート連絡先

**GitHub Issues**: [kyotei-ai-starter/issues](https://github.com/estkey0001/kyotei-ai-starter/issues)

**緊急時**: estkey0001@example.com
