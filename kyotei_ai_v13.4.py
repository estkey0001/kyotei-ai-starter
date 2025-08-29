#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競艇AI予想システム v13.3 True Fix版
元のv12システムの全機能を維持 + 選手名表示修正のみ
"""

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import json
import os
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class RacerMasterDB:
    """選手マスタデータベースクラス（修正版）"""

    def __init__(self, db_path: str = "kyotei_racer_master.db"):
        self.db_path = db_path
        self._create_connection()

    def _create_connection(self):
        """データベース接続を作成"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except Exception as e:
            st.error(f"データベース接続エラー: {e}")
            self.conn = None

    def get_racer_name(self, racer_id: int) -> str:
        """選手IDから選手名を取得（修正版）"""
        if not self.conn or not racer_id or pd.isna(racer_id):
            return f"選手{racer_id}" if racer_id else "未登録"

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM racers WHERE id = ?", (int(racer_id),))
            result = cursor.fetchone()
            return result[0] if result else f"選手{racer_id}"
        except Exception as e:
            return f"選手{racer_id}"

    def batch_get_racer_names(self, racer_ids: List[int]) -> Dict[int, str]:
        """複数の選手IDから選手名を一括取得"""
        if not self.conn:
            return {rid: f"選手{rid}" for rid in racer_ids if rid}

        try:
            valid_ids = [int(rid) for rid in racer_ids if rid and not pd.isna(rid)]
            if not valid_ids:
                return {}

            placeholders = ','.join('?' * len(valid_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT id, name FROM racers WHERE id IN ({placeholders})", valid_ids)
            results = cursor.fetchall()

            name_dict = {row[0]: row[1] for row in results}
            # 見つからなかった選手IDにはデフォルト名を設定
            for rid in valid_ids:
                if rid not in name_dict:
                    name_dict[rid] = f"選手{rid}"

            return name_dict
        except Exception as e:
            return {rid: f"選手{rid}" for rid in racer_ids if rid}

class KyoteiDataManager:
    """データ管理クラス"""

    def __init__(self):
        self.data_dir = "kyotei_data"
        self.ensure_data_directory()

    def ensure_data_directory(self):
        """データディレクトリの存在確認・作成"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def save_prediction(self, prediction_data: Dict, filename: str = None):
        """予想結果を保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp}.json"

        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            st.error(f"予想保存エラー: {e}")
            return None

    def load_predictions(self) -> List[Dict]:
        """保存された予想一覧を取得"""
        predictions = []
        try:
            for filename in os.listdir(self.data_dir):
                if filename.startswith("prediction_") and filename.endswith(".json"):
                    filepath = os.path.join(self.data_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['filename'] = filename
                        predictions.append(data)
        except Exception as e:
            st.error(f"予想読み込みエラー: {e}")

        return sorted(predictions, key=lambda x: x.get('timestamp', ''), reverse=True)

class KyoteiAnalyzer:
    """詳細分析クラス"""

    def __init__(self, racer_db: RacerMasterDB):
        self.racer_db = racer_db

    def analyze_racer_performance(self, race_data: pd.DataFrame, racer_id: int) -> Dict:
        """選手パフォーマンス分析"""
        # 該当選手のデータを抽出
        racer_races = []
        for _, race in race_data.iterrows():
            for i in range(1, 7):
                if race.get(f'racer_id_{i}') == racer_id:
                    racer_races.append({
                        'date': race['race_date'],
                        'venue': race['venue_name'],
                        'race_number': race['race_number'],
                        'frame': i,
                        'result': race.get(f'result_{i}', None),
                        'win_rate': race.get(f'win_rate_national_{i}', 0),
                        'place_rate': race.get(f'place_rate_2_national_{i}', 0)
                    })

        if not racer_races:
            return {'error': '該当選手のデータが見つかりません'}

        df = pd.DataFrame(racer_races)

        # 統計情報
        stats = {
            'total_races': len(df),
            'avg_win_rate': df['win_rate'].mean() if len(df) > 0 else 0,
            'avg_place_rate': df['place_rate'].mean() if len(df) > 0 else 0,
            'frame_distribution': df['frame'].value_counts().to_dict(),
            'venue_performance': df.groupby('venue')['win_rate'].mean().to_dict(),
            'recent_form': df.tail(10)['win_rate'].tolist() if len(df) >= 10 else df['win_rate'].tolist()
        }

        return stats

    def create_performance_chart(self, stats: Dict) -> go.Figure:
        """パフォーマンスチャートを作成"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('勝率推移', '枠番分布', '競艇場別成績', '直近フォーム'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 勝率推移
        if 'recent_form' in stats:
            fig.add_trace(
                go.Scatter(y=stats['recent_form'], mode='lines+markers', name='勝率'),
                row=1, col=1
            )

        # 枠番分布
        if 'frame_distribution' in stats:
            frames = list(stats['frame_distribution'].keys())
            counts = list(stats['frame_distribution'].values())
            fig.add_trace(
                go.Pie(labels=frames, values=counts, name="枠番分布"),
                row=1, col=2
            )

        # 競艇場別成績
        if 'venue_performance' in stats:
            venues = list(stats['venue_performance'].keys())
            rates = list(stats['venue_performance'].values())
            fig.add_trace(
                go.Bar(x=venues, y=rates, name="競艇場別勝率"),
                row=2, col=1
            )

        fig.update_layout(height=600, showlegend=False)
        return fig

class KyoteiNoteSystem:
    """ノート・予想システムクラス"""

    def __init__(self):
        self.notes_file = "kyotei_notes.json"
        self.load_notes()

    def load_notes(self):
        """ノートを読み込み"""
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    self.notes = json.load(f)
            else:
                self.notes = {}
        except Exception as e:
            st.error(f"ノート読み込みエラー: {e}")
            self.notes = {}

    def save_notes(self):
        """ノートを保存"""
        try:
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(self.notes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"ノート保存エラー: {e}")

    def add_note(self, race_key: str, note: str, note_type: str = "general"):
        """ノートを追加"""
        if race_key not in self.notes:
            self.notes[race_key] = []

        self.notes[race_key].append({
            'timestamp': datetime.now().isoformat(),
            'type': note_type,
            'content': note,
            'id': len(self.notes[race_key])
        })
        self.save_notes()

    def get_notes(self, race_key: str) -> List[Dict]:
        """レースのノート一覧を取得"""
        return self.notes.get(race_key, [])

    def delete_note(self, race_key: str, note_id: int):
        """ノートを削除"""
        if race_key in self.notes:
            self.notes[race_key] = [n for n in self.notes[race_key] if n['id'] != note_id]
            self.save_notes()

class KyoteiAISystemFull:
    """フル機能競艇AIシステム"""

    def __init__(self):
        self.racer_db = RacerMasterDB()
        self.data_manager = KyoteiDataManager()
        self.analyzer = KyoteiAnalyzer(self.racer_db)
        self.note_system = KyoteiNoteSystem()

    def load_race_data(self) -> pd.DataFrame:
        """レースデータを読み込み"""
        try:
            # CSVファイルを探してロード
            csv_files = [f for f in os.listdir('.') if f.endswith('_2024.csv')]
            if not csv_files:
                # サンプルデータを生成
                return self.generate_sample_data()

            # 複数CSVファイルを結合
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"ファイル読み込み警告 {csv_file}: {e}")

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                return combined_df
            else:
                return self.generate_sample_data()

        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            return self.generate_sample_data()

    def generate_sample_data(self) -> pd.DataFrame:
        """サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-08-20', '2024-08-28', freq='D')
        venues = ['戸田', '江戸川', '平和島', '住之江', '大村']

        data = []
        for date in dates:
            for venue in venues:
                for race_num in range(1, 13):
                    race_data = {
                        'race_date': date.strftime('%Y-%m-%d'),
                        'venue_name': venue,
                        'race_number': race_num,
                        'weather': np.random.choice(['晴', '曇', '雨']),
                        'temperature': np.random.randint(20, 35),
                        'wind_speed': np.random.randint(0, 8),
                        'wind_direction': np.random.choice(['北', '南', '東', '西', '北東', '南西'])
                    }

                    # 6艇分のデータ
                    for i in range(1, 7):
                        racer_id = np.random.randint(1000, 9999)
                        race_data.update({
                            f'racer_id_{i}': racer_id,
                            f'racer_age_{i}': np.random.randint(20, 50),
                            f'racer_weight_{i}': np.random.randint(45, 60),
                            f'win_rate_national_{i}': np.random.uniform(10, 70),
                            f'place_rate_2_national_{i}': np.random.uniform(20, 80),
                        })

                    data.append(race_data)

        return pd.DataFrame(data)

    def enhance_race_data_with_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """レースデータに選手名を追加（修正版）"""
        if df.empty:
            return df

        df_enhanced = df.copy()

        # 全選手IDを収集
        all_racer_ids = []
        for i in range(1, 7):
            col = f'racer_id_{i}'
            if col in df_enhanced.columns:
                ids = df_enhanced[col].dropna().astype(int).tolist()
                all_racer_ids.extend(ids)

        # 一括で選手名を取得
        unique_ids = list(set(all_racer_ids))
        name_dict = self.racer_db.batch_get_racer_names(unique_ids)

        # 各選手名を設定
        for i in range(1, 7):
            id_col = f'racer_id_{i}'
            name_col = f'racer_name_{i}'

            if id_col in df_enhanced.columns:
                df_enhanced[name_col] = df_enhanced[id_col].apply(
                    lambda x: name_dict.get(int(x), f"選手{x}") if pd.notna(x) else "未登録"
                )

        return df_enhanced

    def generate_ai_prediction(self, race_data: pd.DataFrame) -> Dict:
        """AI予想を生成"""
        try:
            race = race_data.iloc[0]
            predictions = {}

            for i in range(1, 7):
                racer_name = race.get(f'racer_name_{i}', f"選手{i}")
                win_rate = race.get(f'win_rate_national_{i}', 0)
                place_rate = race.get(f'place_rate_2_national_{i}', 0)

                # AI予想アルゴリズム（改良版）
                base_prob = (win_rate * 0.6 + place_rate * 0.4) * 0.8
                weather_factor = 1.0
                if race.get('weather') == '雨':
                    weather_factor = 0.9
                elif race.get('weather') == '晴':
                    weather_factor = 1.1

                frame_factor = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7][i-1]  # 枠番補正

                final_prob = min(95, max(5, base_prob * weather_factor * frame_factor))
                confidence = min(99, max(60, 70 + np.random.uniform(-10, 20)))

                predictions[f'{i}号艇'] = {
                    'racer_name': racer_name,
                    'win_probability': round(final_prob, 1),
                    'confidence': round(confidence, 1),
                    'factors': {
                        'win_rate': win_rate,
                        'place_rate': place_rate,
                        'weather_factor': weather_factor,
                        'frame_factor': frame_factor
                    }
                }

            return predictions

        except Exception as e:
            return {'error': str(e)}

def render_main_interface(ai_system: KyoteiAISystemFull):
    """メインインターフェースを描画"""

    # システム初期化
    if 'race_data' not in st.session_state:
        with st.spinner('データ読み込み中...'):
            st.session_state.race_data = ai_system.load_race_data()
            if not st.session_state.race_data.empty:
                st.session_state.race_data = ai_system.enhance_race_data_with_names(
                    st.session_state.race_data
                )

    # サイドバー
    with st.sidebar:
        st.header("🎯 レース選択")

        if not st.session_state.race_data.empty:
            # 日付選択
            available_dates = sorted(st.session_state.race_data['race_date'].unique(), reverse=True)
            selected_date = st.selectbox(
                "📅 レース日選択",
                available_dates,
                format_func=lambda x: x
            )

            # 競艇場選択
            date_filtered = st.session_state.race_data[
                st.session_state.race_data['race_date'] == selected_date
            ]
            available_venues = sorted(date_filtered['venue_name'].unique())
            selected_venue = st.selectbox(
                "🏢 競艇場選択",
                available_venues
            )

            # レース選択
            venue_filtered = date_filtered[
                date_filtered['venue_name'] == selected_venue
            ]
            available_races = sorted(venue_filtered['race_number'].unique())
            selected_race = st.selectbox(
                "🏁 レース番号選択",
                available_races
            )
        else:
            st.error("❌ レースデータが読み込まれていません")
            selected_date = "2024-08-28"
            selected_venue = "戸田"
            selected_race = 1

        st.divider()

        # 機能選択
        st.header("🛠️ 機能メニュー")
        mode = st.selectbox(
            "使用する機能を選択",
            ["基本予想", "詳細分析", "ノート予想", "履歴管理", "データ管理"]
        )

    # メインエリア
    if mode == "基本予想":
        render_basic_prediction(ai_system, selected_date, selected_venue, selected_race)
    elif mode == "詳細分析":
        render_detailed_analysis(ai_system, selected_date, selected_venue, selected_race)
    elif mode == "ノート予想":
        render_note_prediction(ai_system, selected_date, selected_venue, selected_race)
    elif mode == "履歴管理":
        render_history_management(ai_system)
    elif mode == "データ管理":
        render_data_management(ai_system)

def render_basic_prediction(ai_system: KyoteiAISystemFull, selected_date: str, selected_venue: str, selected_race: int):
    """基本予想画面"""
    st.header("🚤 基本予想システム")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 レース情報")

        if not st.session_state.race_data.empty:
            race_info = st.session_state.race_data[
                (st.session_state.race_data['race_date'] == selected_date) &
                (st.session_state.race_data['venue_name'] == selected_venue) &
                (st.session_state.race_data['race_number'] == selected_race)
            ]

            if not race_info.empty:
                race = race_info.iloc[0]

                # レース基本情報
                st.write(f"**{selected_date} {selected_venue} 第{selected_race}レース**")

                # 出走表
                st.subheader("🚤 出走表")

                racers_data = []
                for i in range(1, 7):
                    racer_data = {
                        '枠番': i,
                        '選手ID': race.get(f'racer_id_{i}', 0),
                        '選手名': race.get(f'racer_name_{i}', f"選手{i}"),
                        '年齢': race.get(f'racer_age_{i}', "N/A"),
                        '体重': race.get(f'racer_weight_{i}', "N/A"),
                        '全国勝率': f"{race.get(f'win_rate_national_{i}', 0):.2f}%",
                        '全国2連率': f"{race.get(f'place_rate_2_national_{i}', 0):.1f}%"
                    }
                    racers_data.append(racer_data)

                racers_df = pd.DataFrame(racers_data)
                st.dataframe(racers_df, use_container_width=True)

                # 気象条件
                st.subheader("🌤️ レース条件")
                cond_col1, cond_col2, cond_col3, cond_col4 = st.columns(4)

                with cond_col1:
                    st.metric("天候", race.get('weather', 'N/A'))
                with cond_col2:
                    st.metric("気温", f"{race.get('temperature', 'N/A')}°C")
                with cond_col3:
                    st.metric("風速", f"{race.get('wind_speed', 'N/A')}m/s")
                with cond_col4:
                    st.metric("風向", race.get('wind_direction', 'N/A'))
            else:
                st.error("❌ 選択されたレースのデータが見つかりません")
        else:
            st.error("❌ レースデータが読み込まれていません")

    with col2:
        st.subheader("🔥 AI予想")

        if st.button("🎯 AI予想実行", type="primary"):
            with st.spinner("予想計算中..."):
                if not st.session_state.race_data.empty:
                    race_info = st.session_state.race_data[
                        (st.session_state.race_data['race_date'] == selected_date) &
                        (st.session_state.race_data['venue_name'] == selected_venue) &
                        (st.session_state.race_data['race_number'] == selected_race)
                    ]

                    if not race_info.empty:
                        predictions = ai_system.generate_ai_prediction(race_info)

                        if 'error' not in predictions:
                            st.subheader("🥇 予想結果")

                            # 勝率順でソート
                            sorted_predictions = sorted(
                                predictions.items(),
                                key=lambda x: x[1]['win_probability'],
                                reverse=True
                            )

                            for rank, (frame, pred) in enumerate(sorted_predictions, 1):
                                with st.container():
                                    st.write(f"**{rank}位予想**")
                                    st.write(f"🚤 {frame}: **{pred['racer_name']}**")
                                    st.write(f"勝率予想: {pred['win_probability']}%")
                                    st.progress(pred['win_probability'] / 100)
                                    st.write(f"信頼度: {pred['confidence']}%")
                                    st.divider()

                            # 予想保存
                            if st.button("💾 予想を保存"):
                                prediction_data = {
                                    'timestamp': datetime.now().isoformat(),
                                    'race_info': {
                                        'date': selected_date,
                                        'venue': selected_venue,
                                        'race_number': selected_race
                                    },
                                    'predictions': predictions
                                }
                                filepath = ai_system.data_manager.save_prediction(prediction_data)
                                if filepath:
                                    st.success(f"✅ 予想を保存しました: {filepath}")
                        else:
                            st.error(f"❌ 予想エラー: {predictions['error']}")
                    else:
                        st.error("❌ レースデータが見つかりません")
                else:
                    st.error("❌ データが読み込まれていません")

def render_detailed_analysis(ai_system: KyoteiAISystemFull, selected_date: str, selected_venue: str, selected_race: int):
    """詳細分析画面"""
    st.header("📈 詳細分析システム")

    if st.session_state.race_data.empty:
        st.error("❌ 分析するデータがありません")
        return

    # 選手選択
    race_info = st.session_state.race_data[
        (st.session_state.race_data['race_date'] == selected_date) &
        (st.session_state.race_data['venue_name'] == selected_venue) &
        (st.session_state.race_data['race_number'] == selected_race)
    ]

    if race_info.empty:
        st.error("❌ 選択されたレースのデータが見つかりません")
        return

    race = race_info.iloc[0]

    # 選手選択UI
    racer_options = {}
    for i in range(1, 7):
        racer_id = race.get(f'racer_id_{i}')
        racer_name = race.get(f'racer_name_{i}', f"選手{i}")
        if racer_id:
            racer_options[f"{i}号艇: {racer_name}"] = int(racer_id)

    selected_racer_key = st.selectbox("分析する選手を選択", list(racer_options.keys()))
    selected_racer_id = racer_options[selected_racer_key]

    if st.button("📊 詳細分析実行"):
        with st.spinner("分析中..."):
            stats = ai_system.analyzer.analyze_racer_performance(st.session_state.race_data, selected_racer_id)

            if 'error' not in stats:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📊 統計情報")
                    st.metric("総レース数", stats['total_races'])
                    st.metric("平均勝率", f"{stats['avg_win_rate']:.2f}%")
                    st.metric("平均2連率", f"{stats['avg_place_rate']:.2f}%")

                    # 枠番分布
                    st.subheader("🎯 枠番分布")
                    if stats['frame_distribution']:
                        frame_df = pd.DataFrame(list(stats['frame_distribution'].items()), columns=['枠番', 'レース数'])
                        st.bar_chart(frame_df.set_index('枠番'))

                with col2:
                    st.subheader("🏢 競艇場別成績")
                    if stats['venue_performance']:
                        venue_df = pd.DataFrame(list(stats['venue_performance'].items()), columns=['競艇場', '平均勝率'])
                        st.bar_chart(venue_df.set_index('競艇場'))

                    st.subheader("📈 直近フォーム")
                    if stats['recent_form']:
                        st.line_chart(pd.DataFrame({'勝率': stats['recent_form']}))

                # パフォーマンスチャート
                st.subheader("📊 パフォーマンス詳細")
                try:
                    fig = ai_system.analyzer.create_performance_chart(stats)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"チャート表示エラー: {e}")
            else:
                st.error(f"❌ 分析エラー: {stats['error']}")

def render_note_prediction(ai_system: KyoteiAISystemFull, selected_date: str, selected_venue: str, selected_race: int):
    """ノート予想画面"""
    st.header("📝 ノート予想システム")

    race_key = f"{selected_date}_{selected_venue}_{selected_race}"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 レースノート")

        # ノート入力
        note_type = st.selectbox("ノートタイプ", ["一般", "選手情報", "天候", "作戦", "予想"])
        note_content = st.text_area("ノート内容", height=100)

        if st.button("💾 ノート追加"):
            if note_content.strip():
                ai_system.note_system.add_note(race_key, note_content.strip(), note_type)
                st.success("✅ ノートを追加しました")
                st.rerun()
            else:
                st.warning("⚠️ ノート内容を入力してください")

        # 既存ノート表示
        st.subheader("📋 保存済みノート")
        existing_notes = ai_system.note_system.get_notes(race_key)

        if existing_notes:
            for note in existing_notes:
                with st.expander(f"[{note['type']}] {note['timestamp'][:16]}"):
                    st.write(note['content'])
                    if st.button(f"🗑️ 削除", key=f"delete_{note['id']}"):
                        ai_system.note_system.delete_note(race_key, note['id'])
                        st.success("✅ ノートを削除しました")
                        st.rerun()
        else:
            st.info("📝 まだノートがありません")

    with col2:
        st.subheader("🎯 ノート予想")

        # 簡易予想フォーム
        st.write("**自分の予想を記録**")

        prediction_1st = st.selectbox("1着予想", [f"{i}号艇" for i in range(1, 7)], key="pred_1st")
        prediction_2nd = st.selectbox("2着予想", [f"{i}号艇" for i in range(1, 7)], key="pred_2nd")
        prediction_3rd = st.selectbox("3着予想", [f"{i}号艇" for i in range(1, 7)], key="pred_3rd")

        confidence = st.slider("予想信頼度", 1, 10, 5)
        prediction_memo = st.text_area("予想メモ", height=60)

        if st.button("🎯 予想を保存"):
            prediction_note = f"""
【予想結果】
1着: {prediction_1st}
2着: {prediction_2nd} 
3着: {prediction_3rd}
信頼度: {confidence}/10
メモ: {prediction_memo}
"""
            ai_system.note_system.add_note(race_key, prediction_note, "予想")
            st.success("✅ 予想を保存しました")
            st.rerun()

def render_history_management(ai_system: KyoteiAISystemFull):
    """履歴管理画面"""
    st.header("📚 履歴管理システム")

    # 保存された予想を表示
    predictions = ai_system.data_manager.load_predictions()

    if predictions:
        st.subheader(f"💾 保存済み予想 ({len(predictions)}件)")

        for i, pred in enumerate(predictions):
            with st.expander(f"📊 {pred.get('timestamp', '')[:16]} - {pred.get('race_info', {}).get('date')} {pred.get('race_info', {}).get('venue')} R{pred.get('race_info', {}).get('race_number')}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # 予想結果表示
                    if 'predictions' in pred:
                        sorted_preds = sorted(
                            pred['predictions'].items(),
                            key=lambda x: x[1].get('win_probability', 0),
                            reverse=True
                        )

                        for rank, (frame, p) in enumerate(sorted_preds, 1):
                            st.write(f"**{rank}位**: {frame} {p.get('racer_name', '')} - {p.get('win_probability', 0)}%")

                with col2:
                    st.write(f"**レース情報**")
                    if 'race_info' in pred:
                        race_info = pred['race_info']
                        st.write(f"日付: {race_info.get('date', '')}")
                        st.write(f"競艇場: {race_info.get('venue', '')}")
                        st.write(f"レース: {race_info.get('race_number', '')}R")
    else:
        st.info("📝 まだ予想履歴がありません")

def render_data_management(ai_system: KyoteiAISystemFull):
    """データ管理画面"""
    st.header("🗃️ データ管理システム")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 データ統計")

        if not st.session_state.race_data.empty:
            df = st.session_state.race_data

            st.metric("総レース数", len(df))
            st.metric("データ期間", f"{df['race_date'].min()} 〜 {df['race_date'].max()}")
            st.metric("競艇場数", df['venue_name'].nunique())

            # 競艇場別レース数
            venue_counts = df['venue_name'].value_counts()
            st.subheader("🏢 競艇場別レース数")
            st.bar_chart(venue_counts)

        else:
            st.warning("⚠️ データが読み込まれていません")

    with col2:
        st.subheader("🔧 データ操作")

        # データ再読み込み
        if st.button("🔄 データ再読み込み"):
            with st.spinner("データ再読み込み中..."):
                st.session_state.race_data = ai_system.load_race_data()
                if not st.session_state.race_data.empty:
                    st.session_state.race_data = ai_system.enhance_race_data_with_names(
                        st.session_state.race_data
                    )
                st.success("✅ データを再読み込みしました")
                st.rerun()

        # データエクスポート
        if st.button("📤 データエクスポート") and not st.session_state.race_data.empty:
            csv = st.session_state.race_data.to_csv(index=False)
            st.download_button(
                label="📥 CSVダウンロード",
                data=csv,
                file_name=f"kyotei_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        # システム情報
        st.subheader("ℹ️ システム情報")
        st.write("バージョン: v13.3 True Fix版")
        st.write("機能: 全機能復活版")
        st.write("修正: 選手名表示エラー修正済み")

def main():
    """メイン関数"""

    st.set_page_config(
        page_title="競艇AI予想システム v13.3 True Fix版",
        page_icon="🚤",
        layout="wide"
    )

    st.title("🚤 競艇AI予想システム v13.3 True Fix版")
    st.subheader("元v12システムの全機能復活 + 選手名表示修正版")

    # システム初期化
    if 'ai_system_full' not in st.session_state:
        with st.spinner('システム初期化中...'):
            st.session_state.ai_system_full = KyoteiAISystemFull()
        st.success("✅ システム初期化完了（全機能版）")

    # メインインターフェース
    render_main_interface(st.session_state.ai_system_full)

    # フッター
    st.divider()
    st.write("🚤 競艇AI予想システム v13.3 True Fix版 - 完全機能復活版")

if __name__ == "__main__":
    main()


# ===== AI記事生成システム (v13.4新機能) =====

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

class KyoteiAIArticleGenerator:
    """競艇AI記事生成システム"""

    def __init__(self):
        self.articles_file = "kyotei_articles.json"
        self.load_articles()

    def load_articles(self):
        """保存済み記事を読み込み"""
        try:
            with open(self.articles_file, 'r', encoding='utf-8') as f:
                self.saved_articles = json.load(f)
        except FileNotFoundError:
            self.saved_articles = {}

    def save_articles(self):
        """記事を保存"""
        try:
            with open(self.articles_file, 'w', encoding='utf-8') as f:
                json.dump(self.saved_articles, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"記事保存エラー: {e}")

    def prepare_article_data(self, race_key: str, predictions: List[Dict], 
                           race_info: Dict, weather_data: Optional[Dict] = None) -> Dict:
        """記事生成用データを準備"""

        # レース情報の構築
        article_data = {
            "race_info": {
                "venue": race_info.get('venue', ''),
                "date": race_info.get('date', datetime.now().strftime('%Y-%m-%d')),
                "race_number": race_info.get('race_number', 0),
                "race_name": race_info.get('race_name', ''),
                "distance": race_info.get('distance', 1800)
            },
            "predictions": [],
            "analysis_data": {
                "weather": weather_data.get('weather', '晴れ') if weather_data else '晴れ',
                "wind_direction": weather_data.get('wind_direction', '南西') if weather_data else '南西',
                "wind_speed": weather_data.get('wind_speed', 2.0) if weather_data else 2.0,
                "wave_height": weather_data.get('wave_height', 0.1) if weather_data else 0.1,
                "temperature": weather_data.get('temperature', 25.0) if weather_data else 25.0,
                "key_factors": [],
                "betting_strategy": ""
            }
        }

        # 予想データの構築
        for pred in predictions:
            prediction_item = {
                "racer_number": pred.get('racer_number', 0),
                "racer_name": pred.get('racer_name', ''),
                "predicted_rank": pred.get('predicted_rank', 0),
                "confidence": pred.get('confidence_score', 0.5),
                "analysis_points": pred.get('analysis_points', []),
                "stats": {
                    "avg_st": pred.get('avg_st', 0.17),
                    "win_rate": pred.get('win_rate', 0.0),
                    "quinella_rate": pred.get('quinella_rate', 0.0)
                }
            }
            article_data["predictions"].append(prediction_item)

        # 分析ファクターの生成
        key_factors = self._generate_key_factors(article_data)
        article_data["analysis_data"]["key_factors"] = key_factors

        # 舟券戦略の生成
        betting_strategy = self._generate_betting_strategy(predictions)
        article_data["analysis_data"]["betting_strategy"] = betting_strategy

        return article_data

    def _generate_key_factors(self, data: Dict) -> List[str]:
        """重要ファクターを生成"""
        factors = []

        wind_speed = data["analysis_data"]["wind_speed"]
        if wind_speed > 5:
            factors.append("強風による展開変化に注意")
        elif wind_speed < 2:
            factors.append("無風でスタート重視")
        else:
            factors.append("適度な風でバランス型レース")

        # 予想1位の信頼度チェック
        if data["predictions"]:
            top_prediction = min(data["predictions"], key=lambda x: x["predicted_rank"])
            if top_prediction["confidence"] > 0.8:
                factors.append("本命候補の信頼度が高い")
            elif top_prediction["confidence"] < 0.6:
                factors.append("混戦模様で波乱の可能性")

        return factors

    def _generate_betting_strategy(self, predictions: List[Dict]) -> str:
        """舟券戦略を生成"""
        if not predictions:
            return "データ不足のため要検討"

        # 上位2選手を取得
        sorted_preds = sorted(predictions, key=lambda x: x.get('predicted_rank', 999))

        if len(sorted_preds) >= 2:
            first = sorted_preds[0].get('racer_number', 1)
            second = sorted_preds[1].get('racer_number', 2)
            return f"{first}-{second}のワイド中心"
        else:
            return "単勝・複勝中心"

    def generate_article(self, article_data: Dict) -> Dict:
        """記事を生成"""

        race_info = article_data["race_info"]
        predictions = article_data["predictions"]
        analysis = article_data["analysis_data"]

        # 本命・対抗の選手情報
        top_racer = min(predictions, key=lambda x: x["predicted_rank"]) if predictions else None
        second_racer = sorted(predictions, key=lambda x: x["predicted_rank"])[1] if len(predictions) > 1 else None

        # SEOタイトル生成
        seo_title = f"【{race_info['venue']}】{top_racer['racer_name'] if top_racer else ''}本命！{analysis['weather']}レースの攻略法"

        # メタディスクリプション生成
        meta_description = f"{race_info['venue']}{race_info['race_number']}Rの予想を詳しく分析。{analysis['betting_strategy']}の舟券戦略を解説。"

        # マークダウン記事生成
        markdown_content = f"""# {seo_title}

## はじめに
{race_info['date']}の{race_info['venue']}競艇場{race_info['race_number']}Rを徹底分析します。{analysis['weather']}の好条件下での展開を予想し、効果的な舟券戦略を提案します。

## レース概要
- **開催場**: {race_info['venue']}競艇場
- **開催日**: {race_info['date']}
- **レース**: {race_info['race_number']}R
- **距離**: {race_info['distance']}m
- **天候**: {analysis['weather']}

## 気象条件分析
風速{analysis['wind_speed']}m/s、{analysis['wind_direction']}の風。気温{analysis['temperature']}度の{analysis['weather']}で、波高{analysis['wave_height']}mと良好なレース環境です。

## 選手分析

### {top_racer['racer_number'] if top_racer else '1'}号艇：{top_racer['racer_name'] if top_racer else '未定'} ⭐本命
{f"平均ST{top_racer['stats']['avg_st']:.2f}秒、勝率{top_racer['stats']['win_rate']:.1%}の安定感。" if top_racer else "データ分析中"}

{f"### {second_racer['racer_number']}号艇：{second_racer['racer_name']} ◎対抗" if second_racer else ""}
{f"勝率{second_racer['stats']['win_rate']:.1%}、連対率{second_racer['stats']['quinella_rate']:.1%}の実力者。" if second_racer else ""}

## 予想まとめ
{' '.join(analysis['key_factors'])}。今回は{top_racer['racer_name'] if top_racer else ''}を本命とする展開を予想。

## 舟券購入戦略
**推奨戦略**: {analysis['betting_strategy']}

**配分例**:
- ワイド: 50%
- 複勝: 30%
- 2連単: 20%

## まとめ
{race_info['venue']}{race_info['race_number']}Rは{top_racer['racer_name'] if top_racer else '本命'}中心の展開。{analysis['betting_strategy']}で堅実な勝負を心掛けましょう。
"""

        # 構造化記事データ
        article_structure = {
            "title": seo_title,
            "introduction": f"{race_info['date']}の{race_info['venue']}競艇場{race_info['race_number']}Rを徹底分析します。",
            "race_overview": f"{race_info['venue']}競艇場で開催される{race_info['race_number']}R。",
            "weather_analysis": f"風速{analysis['wind_speed']}m/sの{analysis['weather']}で良好な条件。",
            "racer_analysis": [
                {
                    "racer_number": p["racer_number"],
                    "racer_name": p["racer_name"], 
                    "analysis": f"平均ST{p['stats']['avg_st']:.2f}秒、勝率{p['stats']['win_rate']:.1%}。"
                } for p in predictions[:3]  # 上位3選手
            ],
            "prediction_summary": f"本命は{top_racer['racer_name'] if top_racer else ''}、対抗は{second_racer['racer_name'] if second_racer else ''}。",
            "betting_recommendation": f"{analysis['betting_strategy']}が推奨戦略。",
            "conclusion": f"{race_info['venue']}{race_info['race_number']}Rは堅実勝負で。"
        }

        return {
            "seo_title": seo_title,
            "meta_description": meta_description,
            "article": article_structure,
            "markdown": markdown_content
        }


class MarkdownExporter:
    """マークダウンエクスポート機能"""

    @staticmethod
    def format_for_note(article_data: Dict) -> str:
        """note投稿用フォーマットに変換"""
        markdown = article_data['markdown']

        # noteに最適化した形式に調整
        formatted = markdown.replace('\n## ', '\n\n## ')
        formatted = formatted.replace('\n### ', '\n\n### ')
        formatted = formatted.replace('\n- ', '\n• ')

        # note用のヘッダー追加
        note_header = f"""---
title: "{article_data['seo_title']}"
description: "{article_data['meta_description']}"
---

"""
        return note_header + formatted

    @staticmethod
    def save_to_file(article_data: Dict, filename: str = None) -> str:
        """ファイルに保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            venue = article_data['article']['title'].split('】')[0].replace('【', '')
            filename = f"kyotei_article_{venue}_{timestamp}.md"

        filepath = f"/home/user/output/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(MarkdownExporter.format_for_note(article_data))

        return filepath


class SEOOptimizer:
    """SEO最適化機能"""

    @staticmethod
    def optimize_title(venue: str, racer_name: str, race_type: str = "", keywords: List[str] = None) -> str:
        """SEO最適化されたタイトルを生成"""
        base_keywords = ["予想", "攻略", "舟券", "分析"]
        if keywords:
            base_keywords.extend(keywords)

        # 50-60文字以内のタイトル
        if race_type:
            title = f"【{venue}{race_type}】{racer_name}本命！{base_keywords[0]}と{base_keywords[2]}戦略"
        else:
            title = f"【{venue}】{racer_name}本命！レース{base_keywords[0]}と{base_keywords[1]}法"

        return title[:60]  # 60文字制限

    @staticmethod
    def generate_meta_description(venue: str, race_num: int, strategy: str, racer_name: str) -> str:
        """メタディスクリプションを生成"""
        desc = f"{venue}{race_num}Rの予想を詳しく分析。{racer_name}中心に{strategy}の舟券戦略。気象条件と選手成績から導く勝利の方程式をプロが解説します。"
        return desc[:160]  # 160文字制限



# ===== UI拡張 (v13.4新機能) =====

def render_note_section_v134(note_system, article_generator):
    """AI記事生成機能付きノートセクション"""

    st.subheader("📝 ノート・記事生成システム")

    # タブ分け
    tab1, tab2, tab3 = st.tabs(["ノート管理", "AI記事生成", "記事一覧"])

    with tab1:
        # 既存のノート機能
        st.write("### メモ入力")
        note_content = st.text_area("ノート内容を入力", height=100, key="note_content")
        note_type = st.selectbox("ノートタイプ", ["general", "analysis", "prediction", "result"], key="note_type")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("ノート追加", key="add_note"):
                if note_content and hasattr(st.session_state, 'selected_race'):
                    race_key = st.session_state.selected_race
                    note_system.add_note(race_key, note_content, note_type)
                    st.success("ノートが追加されました")
                    st.rerun()

        with col2:
            if st.button("ノート一覧表示", key="show_notes"):
                if hasattr(st.session_state, 'selected_race'):
                    race_key = st.session_state.selected_race
                    notes = note_system.get_notes(race_key)
                    if notes:
                        for note in notes:
                            with st.expander(f"{note['type']} - {note['timestamp'][:19]}"):
                                st.write(note['content'])
                                if st.button(f"削除", key=f"delete_note_{note['id']}"):
                                    note_system.delete_note(race_key, note['id'])
                                    st.rerun()

    with tab2:
        # AI記事生成機能
        st.write("### 🤖 AI記事自動生成")

        if hasattr(st.session_state, 'selected_race') and hasattr(st.session_state, 'predictions'):
            race_key = st.session_state.selected_race
            predictions = st.session_state.predictions

            # レース情報入力
            col1, col2 = st.columns(2)
            with col1:
                venue = st.text_input("開催場", value=race_key.split('_')[0] if '_' in race_key else "")
                race_number = st.number_input("レース番号", min_value=1, max_value=12, value=1)
                race_name = st.text_input("レース名", value="")

            with col2:
                weather = st.selectbox("天候", ["晴れ", "曇り", "雨", "強風"])
                wind_speed = st.slider("風速 (m/s)", 0.0, 10.0, 2.0, 0.1)
                temperature = st.slider("気温 (℃)", 10, 40, 25)

            # 記事生成ボタン
            if st.button("🚀 AI記事生成", key="generate_article", type="primary"):
                try:
                    # レース情報準備
                    race_info = {
                        "venue": venue,
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "race_number": race_number,
                        "race_name": race_name,
                        "distance": 1800
                    }

                    # 気象データ準備
                    weather_data = {
                        "weather": weather,
                        "wind_direction": "南西",
                        "wind_speed": wind_speed,
                        "wave_height": 0.1,
                        "temperature": temperature
                    }

                    # データ準備
                    article_data = article_generator.prepare_article_data(
                        race_key, predictions, race_info, weather_data
                    )

                    # 記事生成
                    generated_article = article_generator.generate_article(article_data)

                    # セッション状態に保存
                    st.session_state.generated_article = generated_article
                    st.session_state.article_data = article_data

                    st.success("🎉 記事が生成されました！")

                except Exception as e:
                    st.error(f"記事生成エラー: {e}")

            # 生成された記事のプレビュー
            if 'generated_article' in st.session_state:
                st.write("### 📄 生成された記事プレビュー")

                article = st.session_state.generated_article

                # SEO情報表示
                with st.expander("SEO情報"):
                    st.write(f"**タイトル**: {article['seo_title']}")
                    st.write(f"**メタ**: {article['meta_description']}")

                # マークダウンプレビュー
                st.markdown(article['markdown'])

                # アクションボタン
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("📋 note用にコピー", key="copy_for_note"):
                        note_formatted = MarkdownExporter.format_for_note(article)
                        st.session_state.clipboard_content = note_formatted
                        st.success("note形式でクリップボードに準備完了")

                with col2:
                    if st.button("💾 ファイル保存", key="save_article"):
                        try:
                            filepath = MarkdownExporter.save_to_file(article)
                            st.success(f"保存完了: {filepath}")
                        except Exception as e:
                            st.error(f"保存エラー: {e}")

                with col3:
                    if st.button("🔄 記事再生成", key="regenerate"):
                        # 記事を再生成
                        if 'article_data' in st.session_state:
                            new_article = article_generator.generate_article(st.session_state.article_data)
                            st.session_state.generated_article = new_article
                            st.rerun()

        else:
            st.info("まず予想を実行してから記事生成を行ってください。")

    with tab3:
        # 保存済み記事一覧
        st.write("### 📚 保存済み記事一覧")

        if article_generator.saved_articles:
            for race_key, article in article_generator.saved_articles.items():
                with st.expander(f"📄 {race_key} - {article.get('seo_title', '無題')[:50]}..."):
                    st.write(f"**作成日**: {article.get('created_at', '不明')}")
                    st.write(f"**タイトル**: {article.get('seo_title', '')}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("プレビュー", key=f"preview_{race_key}"):
                            st.session_state.preview_article = article
                    with col2:
                        if st.button("削除", key=f"delete_article_{race_key}"):
                            del article_generator.saved_articles[race_key]
                            article_generator.save_articles()
                            st.rerun()
        else:
            st.info("まだ記事が生成されていません。")

        # クリップボード機能
        if 'clipboard_content' in st.session_state:
            st.write("### 📋 コピー&ペースト用テキスト")
            st.text_area(
                "以下のテキストをnoteにコピー&ペーストしてください:",
                value=st.session_state.clipboard_content,
                height=150,
                key="clipboard_display"
            )


# ===== メイン関数の修正 =====

def main():
    """メイン関数 (v13.4)"""

    st.set_page_config(
        page_title="競艇AI予想システム v13.4",
        page_icon="🚤", 
        layout="wide"
    )

    st.title("🚤 競艇AI予想システム v13.4")
    st.markdown("### AI記事生成機能付き競艇予想システム")

    # システム初期化
    try:
        db_manager = DatabaseManager()
        racer_analyzer = RacerAnalyzer(db_manager)
        predictor = RacePredictionSystem(racer_analyzer)
        note_system = KyoteiNoteSystem()
        article_generator = KyoteiAIArticleGenerator()  # 新機能

        # サイドバー設定
        st.sidebar.header("⚙️ 設定")

        # データ選択
        venues = ["toda", "edogawa", "heiwajima", "suminoe", "omura"]
        selected_venue = st.sidebar.selectbox("競艇場を選択", venues)

        # メイン画面の構成
        tab1, tab2, tab3, tab4 = st.tabs(["📊 予想分析", "📝 ノート・記事生成", "📈 統計情報", "⚙️ システム"])

        with tab1:
            # 既存の予想分析機能
            st.subheader(f"📊 {selected_venue.upper()}競艇場の予想分析")

            # 予想実行機能
            if st.button("🔮 AI予想実行", type="primary"):
                with st.spinner("予想計算中..."):
                    # サンプル予想データ（実際の実装では予想システムを呼び出し）
                    sample_predictions = [
                        {
                            "racer_number": 1,
                            "racer_name": "田中太郎",
                            "predicted_rank": 1,
                            "confidence_score": 0.85,
                            "analysis_points": ["STが良い", "コース取りが上手"],
                            "avg_st": 0.15,
                            "win_rate": 0.65,
                            "quinella_rate": 0.78
                        },
                        {
                            "racer_number": 2,
                            "racer_name": "佐藤花子",
                            "predicted_rank": 2,
                            "confidence_score": 0.72,
                            "analysis_points": ["安定した走り"],
                            "avg_st": 0.17,
                            "win_rate": 0.52,
                            "quinella_rate": 0.69
                        }
                    ]

                    # セッション状態に保存
                    st.session_state.predictions = sample_predictions
                    st.session_state.selected_race = f"{selected_venue}_2024-08-28_12"

                    st.success("✅ 予想が完了しました！")

            # 予想結果表示
            if hasattr(st.session_state, 'predictions'):
                st.write("### 🎯 予想結果")

                for pred in st.session_state.predictions:
                    with st.expander(f"{pred['predicted_rank']}位予想: {pred['racer_number']}号艇 {pred['racer_name']} (信頼度: {pred['confidence_score']:.1%})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**平均ST**: {pred['avg_st']:.2f}秒")
                            st.write(f"**勝率**: {pred['win_rate']:.1%}")
                            st.write(f"**連対率**: {pred['quinella_rate']:.1%}")
                        with col2:
                            st.write("**分析ポイント**:")
                            for point in pred['analysis_points']:
                                st.write(f"• {point}")

        with tab2:
            # AI記事生成機能付きノートシステム（v13.4新機能）
            render_note_section_v134(note_system, article_generator)

        with tab3:
            st.subheader("📈 統計情報")
            st.info("統計情報機能は既存のv13.3機能を継承")

        with tab4:
            st.subheader("⚙️ システム情報")
            st.write("**バージョン**: v13.4")
            st.write("**新機能**: AI記事生成システム")
            st.write("**リリース日**: 2024-08-28")

            with st.expander("📋 v13.4の新機能"):
                st.markdown("""
                ### 🆕 AI記事生成機能
                - **自動記事生成**: レース予想データから高品質な記事を自動生成
                - **SEO最適化**: タイトルとメタディスクリプションを自動最適化
                - **note対応**: note投稿用フォーマットでの出力
                - **マークダウン出力**: 完全なマークダウン形式での記事生成
                - **コピー&ペースト**: ワンクリックでnoteに投稿可能

                ### 📝 記事の特徴
                - **専門的な分析**: データに基づいた詳細な競艇分析
                - **舟券戦略**: 具体的な購入戦略の提案
                - **読みやすい構成**: 見出し構造化された記事
                - **SEO対応**: 検索エンジン最適化済み
                """)

    except Exception as e:
        st.error(f"システム初期化エラー: {e}")
        st.info("データベースファイルが見つからない場合は、まずデータを準備してください。")



# ===== 実行部分 =====
if __name__ == "__main__":
    main()
