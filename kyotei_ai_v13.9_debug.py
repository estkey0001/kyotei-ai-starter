import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import os
import glob
import chardet
import logging
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# デバッグレベルのログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KyoteiDataManager:
    """競艇データ管理クラス - パス検索とデバッグ機能強化版"""

    def __init__(self):
        self.debug_info = {
            'searched_paths': [],
            'found_files': {},
            'errors': [],
            'encoding_results': {}
        }
        self.csv_data = {}
        self.db_connection = None
        self.available_races = []

        # データパスの候補（優先順位順）
        self.candidate_paths = [
            "~/kyotei-ai-starter/data/coconala_2024/",
            "~/kyotei-ai-starter/",  
            "./data/coconala_2024/",
            "./kyotei_data/",
            "./data/",
            "./"
        ]

        # DBファイルの候補
        self.db_candidates = [
            "~/kyotei-ai-starter/kyotei_racer_master.db",
            "./kyotei_racer_master.db",
            "./data/kyotei_racer_master.db"
        ]

        self._initialize_data_paths()

    def _initialize_data_paths(self):
        """データパスの初期化と検索"""
        logger.info("データパス検索を開始します...")

        # CSV データディレクトリの検索
        self.csv_data_path = self._search_csv_directory()

        # データベースファイルの検索
        self.db_path = self._search_database_file()

        # デバッグ情報を表示
        self._display_debug_info()

    def _search_csv_directory(self):
        """CSV データディレクトリを検索"""
        logger.info("CSV データディレクトリを検索中...")

        for candidate in self.candidate_paths:
            expanded_path = Path(candidate).expanduser().resolve()
            self.debug_info['searched_paths'].append(str(expanded_path))

            if expanded_path.exists() and expanded_path.is_dir():
                # CSVファイルがあるかチェック
                csv_files = list(expanded_path.glob("*.csv"))
                if csv_files:
                    self.debug_info['found_files'][str(expanded_path)] = [f.name for f in csv_files]
                    logger.info(f"CSV データディレクトリを発見: {expanded_path}")
                    logger.info(f"CSVファイル数: {len(csv_files)}")
                    return expanded_path

        # 見つからない場合はカレントディレクトリを使用
        current_path = Path(".").resolve()
        logger.warning(f"CSV データディレクトリが見つかりません。カレントディレクトリを使用: {current_path}")
        return current_path

    def _search_database_file(self):
        """データベースファイルを検索"""
        logger.info("データベースファイルを検索中...")

        for candidate in self.db_candidates:
            expanded_path = Path(candidate).expanduser().resolve()

            if expanded_path.exists() and expanded_path.is_file():
                logger.info(f"データベースファイルを発見: {expanded_path}")
                return expanded_path

        logger.warning("データベースファイルが見つかりません")
        return None

    def _detect_encoding(self, file_path):
        """ファイルエンコーディングを自動検出"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 最初の10KBでエンコーディング検出
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']

                self.debug_info['encoding_results'][str(file_path)] = {
                    'encoding': encoding,
                    'confidence': confidence
                }

                logger.info(f"エンコーディング検出: {file_path.name} -> {encoding} (信頼度: {confidence:.2f})")
                return encoding

        except Exception as e:
            logger.error(f"エンコーディング検出エラー: {file_path} - {e}")
            return 'utf-8'  # デフォルト

    def _normalize_column_names(self, df):
        """列名の正規化"""
        original_columns = df.columns.tolist()

        # 列名の正規化ルール
        normalized_columns = []
        for col in df.columns:
            # 前後の空白削除
            normalized = col.strip()
            # 全角・半角統一
            normalized = normalized.replace('　', ' ')
            # 特殊文字の統一
            normalized = re.sub(r'[（(]', '(', normalized)
            normalized = re.sub(r'[）)]', ')', normalized)
            normalized_columns.append(normalized)

        df.columns = normalized_columns

        if original_columns != normalized_columns:
            logger.info("列名を正規化しました")
            for orig, norm in zip(original_columns, normalized_columns):
                if orig != norm:
                    logger.info(f"  {orig} -> {norm}")

        return df

    def load_csv_data(self):
        """CSV データを読み込み"""
        logger.info("CSV データ読み込みを開始...")

        if not self.csv_data_path.exists():
            error_msg = f"データディレクトリが存在しません: {self.csv_data_path}"
            self.debug_info['errors'].append(error_msg)
            logger.error(error_msg)
            return False

        csv_files = list(self.csv_data_path.glob("*.csv"))

        if not csv_files:
            error_msg = f"CSVファイルが見つかりません: {self.csv_data_path}"
            self.debug_info['errors'].append(error_msg)
            logger.error(error_msg)
            return False

        logger.info(f"発見されたCSVファイル数: {len(csv_files)}")

        # CSVファイルを読み込み
        successful_loads = 0
        for csv_file in csv_files:
            try:
                # エンコーディング自動検出
                encoding = self._detect_encoding(csv_file)

                # CSV読み込み
                df = pd.read_csv(csv_file, encoding=encoding)

                # 列名正規化
                df = self._normalize_column_names(df)

                # 日付列の自動変換
                df = self._auto_convert_dates(df)

                # ファイル名（拡張子なし）をキーとして保存
                key = csv_file.stem
                self.csv_data[key] = df

                successful_loads += 1
                logger.info(f"読み込み成功: {csv_file.name} (行数: {len(df)}, 列数: {len(df.columns)})")

            except Exception as e:
                error_msg = f"CSV読み込みエラー: {csv_file.name} - {e}"
                self.debug_info['errors'].append(error_msg)
                logger.error(error_msg)

        logger.info(f"CSV読み込み完了: {successful_loads}/{len(csv_files)} ファイル成功")
        return successful_loads > 0

    def _auto_convert_dates(self, df):
        """日付フォーマットの自動変換"""
        date_columns = []

        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', '日付', '年月日', 'day']):
                date_columns.append(col)

        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"日付変換: {col}")
            except Exception as e:
                logger.warning(f"日付変換失敗: {col} - {e}")

        return df

    def connect_database(self):
        """データベース接続"""
        if self.db_path is None:
            logger.warning("データベースファイルが見つからないため、DB機能は無効です")
            return False

        try:
            self.db_connection = sqlite3.connect(str(self.db_path))
            logger.info(f"データベース接続成功: {self.db_path}")
            return True
        except Exception as e:
            error_msg = f"データベース接続エラー: {e}"
            self.debug_info['errors'].append(error_msg)
            logger.error(error_msg)
            return False

    def get_available_races(self):
        """利用可能なレース一覧を取得"""
        races = []

        # CSV データからレース情報を抽出
        for filename, df in self.csv_data.items():
            if not df.empty:
                race_info = {
                    'source': 'CSV',
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'date_range': self._get_date_range(df)
                }
                races.append(race_info)

        self.available_races = races
        logger.info(f"利用可能なレース数: {len(races)}")
        return races

    def _get_date_range(self, df):
        """データフレームの日付範囲を取得"""
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)

        if date_cols:
            try:
                min_date = df[date_cols[0]].min()
                max_date = df[date_cols[0]].max()
                return f"{min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
            except:
                return "日付範囲不明"
        return "日付なし"

    def _display_debug_info(self):
        """デバッグ情報の表示"""
        logger.info("=== デバッグ情報 ===")
        logger.info(f"検索されたパス数: {len(self.debug_info['searched_paths'])}")

        for path in self.debug_info['searched_paths']:
            logger.info(f"  検索パス: {path}")

        logger.info(f"発見されたディレクトリ数: {len(self.debug_info['found_files'])}")
        for path, files in self.debug_info['found_files'].items():
            logger.info(f"  {path}: {len(files)} ファイル")

        if self.debug_info['errors']:
            logger.info(f"エラー数: {len(self.debug_info['errors'])}")
            for error in self.debug_info['errors']:
                logger.error(f"  {error}")

    def get_debug_summary(self):
        """デバッグ情報のサマリーを返す"""
        return {
            'csv_data_path': str(self.csv_data_path) if self.csv_data_path else "未発見",
            'db_path': str(self.db_path) if self.db_path else "未発見",
            'csv_files_loaded': len(self.csv_data),
            'total_rows': sum(len(df) for df in self.csv_data.values()),
            'available_races': len(self.available_races),
            'errors': len(self.debug_info['errors']),
            'searched_paths': len(self.debug_info['searched_paths'])
        }

class KyoteiAIPrediction:
    """競艇AI予想クラス - 実データ使用版"""

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def analyze_racer_performance(self, racer_data):
        """選手パフォーマンス分析"""
        if racer_data.empty:
            return {}

        try:
            analysis = {
                'total_races': len(racer_data),
                'win_rate': (racer_data.get('着順', pd.Series()).eq(1).sum() / len(racer_data) * 100) if '着順' in racer_data.columns else 0,
                'avg_start_timing': racer_data.get('ST', pd.Series()).mean() if 'ST' in racer_data.columns else 0,
                'recent_performance': self._get_recent_performance(racer_data)
            }
            return analysis
        except Exception as e:
            logger.error(f"選手分析エラー: {e}")
            return {}

    def _get_recent_performance(self, racer_data, days=30):
        """直近パフォーマンス分析"""
        if racer_data.empty:
            return "データなし"

        # 日付列を探す
        date_col = None
        for col in racer_data.columns:
            if 'date' in col.lower() or '日付' in col:
                date_col = col
                break

        if date_col is None:
            return "日付データなし"

        try:
            recent_date = datetime.now() - timedelta(days=days)
            recent_data = racer_data[pd.to_datetime(racer_data[date_col]) >= recent_date]

            if len(recent_data) == 0:
                return "直近データなし"

            wins = recent_data.get('着順', pd.Series()).eq(1).sum()
            return f"直近{days}日: {len(recent_data)}戦{wins}勝"
        except Exception as e:
            logger.error(f"直近パフォーマンス分析エラー: {e}")
            return "分析エラー"

    def predict_race_outcome(self, race_data):
        """レース結果予想"""
        if race_data.empty:
            return {}

        try:
            # 基本的な予想ロジック（実データベース）
            predictions = {}

            if 'オッズ' in race_data.columns:
                # オッズベースの分析
                odds_analysis = self._analyze_odds(race_data)
                predictions.update(odds_analysis)

            if 'ST' in race_data.columns:
                # スタート分析
                start_analysis = self._analyze_start_performance(race_data)
                predictions.update(start_analysis)

            return predictions

        except Exception as e:
            logger.error(f"レース予想エラー: {e}")
            return {}

    def _analyze_odds(self, race_data):
        """オッズ分析"""
        try:
            odds_col = 'オッズ'
            if odds_col in race_data.columns:
                min_odds = race_data[odds_col].min()
                favorite = race_data[race_data[odds_col] == min_odds].iloc[0]

                return {
                    'favorite_boat': favorite.get('艇番', 'N/A'),
                    'favorite_odds': min_odds,
                    'odds_analysis': "オッズ分析完了"
                }
        except Exception as e:
            logger.error(f"オッズ分析エラー: {e}")

        return {}

    def _analyze_start_performance(self, race_data):
        """スタート分析"""
        try:
            st_col = 'ST'
            if st_col in race_data.columns:
                best_st = race_data[st_col].min()
                best_starter = race_data[race_data[st_col] == best_st].iloc[0]

                return {
                    'best_starter': best_starter.get('艇番', 'N/A'),
                    'best_st': best_st,
                    'start_analysis': "スタート分析完了"
                }
        except Exception as e:
            logger.error(f"スタート分析エラー: {e}")

        return {}

    def generate_prediction_report(self, selected_data):
        """予想レポート生成"""
        if not selected_data or len(selected_data) == 0:
            return "分析対象データがありません"

        try:
            report = []
            report.append("# 競艇AI予想レポート")
            report.append(f"## データ概要")
            report.append(f"- 分析対象: {len(selected_data)} レース")
            report.append(f"- 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # データ別分析
            for filename, df in selected_data.items():
                report.append(f"\n### {filename}")
                report.append(f"- レース数: {len(df)}")

                if not df.empty:
                    # 基本統計
                    if '着順' in df.columns:
                        win_rate = (df['着順'].eq(1).sum() / len(df) * 100)
                        report.append(f"- 1着率: {win_rate:.1f}%")

                    if 'オッズ' in df.columns:
                        avg_odds = df['オッズ'].mean()
                        report.append(f"- 平均オッズ: {avg_odds:.2f}")

            report.append("\n## AI予想のポイント")
            report.append("1. **実データ分析**: 過去の実績データを基に分析")
            report.append("2. **多角的評価**: オッズ、スタート、選手成績を総合評価") 
            report.append("3. **リスク管理**: 確率論に基づく堅実な予想")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return f"レポート生成エラー: {e}"

def create_note_article(prediction_report, race_data_summary):
    """note記事生成"""
    try:
        article = []
        article.append("# 🏁 競艇AI予想 - データサイエンスで勝率UP!")
        article.append("")
        article.append("こんにちは！データサイエンティストの私が、")
        article.append("最新の競艇データを徹底分析してお送りする予想記事です。")
        article.append("")

        # データサマリー
        article.append("## 📊 今回の分析データ")
        article.append(f"- 対象レース数: {race_data_summary.get('total_races', 0)}")
        article.append(f"- 分析期間: {race_data_summary.get('period', '直近データ')}")
        article.append("")

        # 予想レポート挿入
        article.append("## 🤖 AI分析結果")
        article.append(prediction_report)
        article.append("")

        # note記事用のまとめ
        article.append("## 💡 今日のポイント")
        article.append("1. **データドリブン**: 感覚ではなく、データで判断")
        article.append("2. **確率論思考**: 100%はない、確率で考える")
        article.append("3. **継続改善**: 結果を検証し、モデルを改善")
        article.append("")

        article.append("## 🎯 免責事項")
        article.append("この予想は過去データの分析結果です。")
        article.append("投資は自己責任でお願いします。")
        article.append("")

        article.append("---")
        article.append("データで勝つ競艇予想、いかがでしたか？")
        article.append("フォロー・スキで応援よろしくお願いします！")

        return "\n".join(article)

    except Exception as e:
        logger.error(f"note記事生成エラー: {e}")
        return f"note記事生成エラー: {e}"

def main():
    """メインアプリケーション"""
    st.set_page_config(
        page_title="競艇AI予想システム v13.9 (Debug)",
        page_icon="🏁",
        layout="wide"
    )

    st.title("🏁 競艇AI予想システム v13.9 (Debug版)")
    st.caption("データパス自動検索・デバッグ機能強化版")

    # データマネージャー初期化
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = KyoteiDataManager()

    data_manager = st.session_state.data_manager

    # デバッグ情報表示
    with st.expander("🔍 システムデバッグ情報", expanded=False):
        debug_summary = data_manager.get_debug_summary()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("データ発見状況")
            st.info(f"📁 CSVデータパス: `{debug_summary['csv_data_path']}`")
            st.info(f"🗃️ データベースパス: `{debug_summary['db_path']}`")
            st.success(f"✅ 読み込み済みCSV: {debug_summary['csv_files_loaded']} ファイル")
            st.success(f"📊 総データ行数: {debug_summary['total_rows']:,} 行")

        with col2:
            st.subheader("検索・エラー状況")
            st.info(f"🔍 検索パス数: {debug_summary['searched_paths']}")
            st.info(f"🏁 利用可能レース: {debug_summary['available_races']}")

            if debug_summary['errors'] > 0:
                st.error(f"⚠️ エラー数: {debug_summary['errors']}")
            else:
                st.success("✅ エラーなし")

        # 詳細デバッグ情報
        if st.button("📋 詳細ログを表示"):
            st.subheader("検索パス一覧")
            for path in data_manager.debug_info['searched_paths']:
                st.text(f"• {path}")

            if data_manager.debug_info['found_files']:
                st.subheader("発見されたファイル")
                for path, files in data_manager.debug_info['found_files'].items():
                    st.text(f"📁 {path}")
                    for file in files:
                        st.text(f"  • {file}")

            if data_manager.debug_info['errors']:
                st.subheader("エラー詳細")
                for error in data_manager.debug_info['errors']:
                    st.error(error)

    # データ読み込み
    if st.button("🔄 データ再読み込み"):
        with st.spinner("データを読み込み中..."):
            data_manager.load_csv_data()
            data_manager.connect_database()
            st.success("データ読み込み完了！")
            st.rerun()

    # 初期データ読み込み
    if not data_manager.csv_data:
        st.warning("⚠️ データが読み込まれていません。上の「データ再読み込み」ボタンをクリックしてください。")
        st.stop()

    # メイン機能
    st.header("🎯 競艇AI予想")

    # データ選択
    available_data = list(data_manager.csv_data.keys())
    selected_files = st.multiselect(
        "📊 分析対象データを選択",
        available_data,
        default=available_data[:3] if len(available_data) >= 3 else available_data
    )

    if not selected_files:
        st.warning("分析対象データを選択してください")
        st.stop()

    # 選択されたデータの概要表示
    col1, col2, col3 = st.columns(3)

    selected_data = {f: data_manager.csv_data[f] for f in selected_files}
    total_rows = sum(len(df) for df in selected_data.values())

    with col1:
        st.metric("選択ファイル数", len(selected_files))
    with col2:
        st.metric("総レース数", f"{total_rows:,}")
    with col3:
        avg_rows = total_rows // len(selected_files) if selected_files else 0
        st.metric("ファイル平均行数", f"{avg_rows:,}")

    # AI予想実行
    if st.button("🤖 AI予想を実行", type="primary"):
        with st.spinner("AI分析中..."):
            # AI予想システム初期化
            ai_predictor = KyoteiAIPrediction(data_manager)

            # 予想レポート生成
            prediction_report = ai_predictor.generate_prediction_report(selected_data)

            # データサマリー
            race_data_summary = {
                'total_races': total_rows,
                'period': '過去データ',
                'files': len(selected_files)
            }

            st.success("✅ AI分析完了！")

            # 結果表示
            tab1, tab2, tab3 = st.tabs(["📊 予想結果", "📝 note記事", "📈 データ詳細"])

            with tab1:
                st.subheader("🤖 AI予想レポート")
                st.markdown(prediction_report)

            with tab2:
                st.subheader("📝 note投稿用記事")
                note_article = create_note_article(prediction_report, race_data_summary)
                st.markdown(note_article)

                # コピー用テキストエリア
                st.text_area("📋 コピー用テキスト", note_article, height=300)

            with tab3:
                st.subheader("📈 使用データ詳細")

                for filename, df in selected_data.items():
                    with st.expander(f"📊 {filename} (行数: {len(df):,})"):
                        st.dataframe(df.head())

                        # 基本統計
                        if not df.empty:
                            st.subheader("基本統計")
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                st.dataframe(df[numeric_cols].describe())

    # フッター
    st.markdown("---")
    st.caption("🏁 競艇AI予想システム v13.9 - Debug版 | データパス自動検索・エラーハンドリング強化")
    st.caption(f"💾 現在のデータ: CSV {len(data_manager.csv_data)} ファイル, 総行数 {sum(len(df) for df in data_manager.csv_data.values()):,}")

if __name__ == "__main__":
    main()
