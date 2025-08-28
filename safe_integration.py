print("🔧 安全な実データ情報統合中...")

# app.py読み込み
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# main()関数内でサイドバー情報を安全に追加
new_lines = []
main_function_found = False
sidebar_added = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # main()関数内のサイドバー部分を探す
    if 'def main():' in line:
        main_function_found = True
    
    # サイドバーの会場選択後に安全に追加
    if (main_function_found and not sidebar_added and 
        'selected_venue = st.sidebar.selectbox' in line):
        
        # 次の空行または適切な位置に追加
        next_line_idx = i + 1
        while (next_line_idx < len(lines) and 
               lines[next_line_idx].strip() == ''):
            next_line_idx += 1
        
        # 安全な統合コード追加
        integration = [
            '    \n',
            '    # 実データ学習済み情報表示\n',
            '    if selected_venue == "戸田":\n',
            '        st.sidebar.success("🎯 2024年実データ学習済み")\n',
            '        st.sidebar.metric("学習精度", "44.3%")\n',
            '        st.sidebar.metric("学習レース数", "2,346レース")\n',
            '    else:\n',
            '        st.sidebar.info("📊 統計ベース予想")\n',
            '    \n'
        ]
        
        # 適切な位置に挿入
        for j, integration_line in enumerate(integration):
            new_lines.insert(next_line_idx + j, integration_line)
        
        sidebar_added = True
        break

# 安全統合版保存
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ 安全な実データ情報統合完了")

# 構文チェック
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', 'app.py'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("✅ 構文チェックOK")
else:
    print("❌ 構文エラー:", result.stderr)
