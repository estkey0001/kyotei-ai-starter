print("🔧 変数エラー修正中...")

# app.py読み込み
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 問題のある統合部分を削除
lines = content.split('\n')
fixed_lines = []
skip_integration = False

for line in lines:
    # 問題の統合部分をスキップ
    if 'if selected_venue == "戸田":' in line and 'st.sidebar.success' in content[content.find(line):content.find(line)+200]:
        skip_integration = True
        continue
    elif skip_integration and ('st.sidebar.info' in line or line.strip() == ''):
        if 'st.sidebar.info' in line:
            skip_integration = False
        continue
    
    fixed_lines.append(line)

content = '\n'.join(fixed_lines)

# 正しい位置に統合コード追加
# selected_venue定義後に追加
if 'selected_venue = st.sidebar.selectbox' in content:
    pattern = 'selected_venue = st.sidebar.selectbox'
    pos = content.find(pattern)
    if pos != -1:
        # その行の終わりを見つける
        line_end = content.find('\n', pos)
        if line_end != -1:
            # 次の行に統合コード追加
            integration_code = '''
        
        # 実データ学習済み情報表示
        if selected_venue == "戸田":
            st.sidebar.success("🎯 2024年実データ学習済み")
            st.sidebar.metric("学習精度", "44.3%")
            st.sidebar.metric("学習レース数", "2,346レース")
        else:
            st.sidebar.info("📊 統計ベース予想")
'''
            content = content[:line_end] + integration_code + content[line_end:]

# 修正版保存
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 変数エラー修正完了")
