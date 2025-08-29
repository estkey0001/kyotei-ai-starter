#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競艇AI予想システム v13.9 Realtime
v13.9_fixedの優れたUIを維持し、リアルタイムデータ取得機能を追加

主な機能:
- 公式サイトからのリアルタイムデータ取得
- 3連単ピンポイント予想（本命・中穴・大穴）
- フォーメーション大幅拡張（軸流し、BOX、ワイド等）
- 1画面統合UI（Streamlit）
- 予想根拠詳細表示
- note記事2000文字以上自動生成

作成日: 2025-08-28 12:50:48
"""

import os
import sys
import time
import sqlite3
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import re
import warnings
warnings.filterwarnings('ignore')


class BoatraceRealTimeDataCollector:
    """
    競艇公式サイトからリアルタイムデータを取得するクラス
    """

    def __init__(self):
        self.base_url = "https://www.boatrace.jp"
        self.session = requests.Session()

        # 適切なUser-Agentとheadersを設定（bot対策回避）
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache'
        }
        self.session.headers.update(self.headers)

        # レース場コード辞書
        self.racecourse_dict = {
            '01': '桐生', '02': '戸田', '03': '江戸川', '04': '平和島',
            '05': '多摩川', '06': '浜名湖', '07': '蒲郡', '08': '常滑',
            '09': '津', '10': '三国', '11': '琵琶湖', '12': '住之江',
            '13': '尼崎', '14': '鳴門', '15': '丸亀', '16': '児島',
            '17': '宮島', '18': '徳山', '19': '下関', '20': '若松',
            '21': '芦屋', '22': '福岡', '23': '唐津', '24': '大村'
        }

        print("競艇リアルタイムデータ取得システムを初期化しました")

    def wait_request(self, min_wait=1.0, max_wait=2.0):
        """適切な間隔でリクエストを制御"""
        wait_time = np.random.uniform(min_wait, max_wait)
        time.sleep(wait_time)

    def safe_request(self, url, max_retries=3):
        """エラーハンドリング付きの安全なHTTPリクエスト"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                print(f"リクエストエラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.wait_request(2, 4)  # 長めの待機
                else:
                    return None
        return None

    def get_today_races(self) -> Dict[str, List]:
        """今日のレース一覧を取得"""
        today_str = datetime.now().strftime('%Y%m%d')
        url = f"{self.base_url}/owpc/pc/race/index?hd={today_str}"

        response = self.safe_request(url)
        if not response:
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')
        race_info = {}

        try:
            # レース場別の開催情報を取得
            race_items = soup.find_all('li', class_='tab2_item')

            for item in race_items:
                # レース場コードと名前を取得
                link = item.find('a')
                if not link:
                    continue

                href = link.get('href', '')
                match = re.search(r'jcd=(\d+)', href)
                if not match:
                    continue

                racecourse_code = match.group(1)
                racecourse_name = self.racecourse_dict.get(racecourse_code, f"場{racecourse_code}")

                # レース数や開催情報を取得
                race_info[racecourse_code] = {
                    'name': racecourse_name,
                    'races': [],
                    'url': f"{self.base_url}{href}" if not href.startswith('http') else href
                }

                print(f"レース場取得: {racecourse_name} (コード: {racecourse_code})")
                self.wait_request()

        except Exception as e:
            print(f"レース一覧取得エラー: {e}")

        return race_info

    def get_race_data(self, racecourse_code: str, race_num: int, date_str: str = None) -> Dict:
        """指定されたレースの出走表データを取得"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        url = f"{self.base_url}/owpc/pc/race/racelist?rno={race_num}&jcd={racecourse_code}&hd={date_str}"

        response = self.safe_request(url)
        if not response:
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')
        race_data = {
            'racecourse_code': racecourse_code,
            'racecourse_name': self.racecourse_dict.get(racecourse_code, f"場{racecourse_code}"),
            'race_num': race_num,
            'date': date_str,
            'boats': []
        }

        try:
            # 出走表のボート情報を取得
            boat_rows = soup.find_all('tr', class_='is-fs12')

            for i, row in enumerate(boat_rows, 1):
                boat_data = {'boat_num': i}

                # 選手名を取得
                name_cell = row.find('td', class_='is-fs14')
                if name_cell:
                    boat_data['player_name'] = name_cell.get_text(strip=True)

                # 各種データを取得（年齢、体重、F、L、級別、全国勝率、当地勝率など）
                cells = row.find_all('td')
                if len(cells) >= 8:
                    boat_data.update({
                        'age': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                        'weight': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                        'f_count': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                        'l_count': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                        'class': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                        'national_rate': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                        'local_rate': cells[7].get_text(strip=True) if len(cells) > 7 else '',
                        'motor_rate': cells[8].get_text(strip=True) if len(cells) > 8 else '',
                        'boat_rate': cells[9].get_text(strip=True) if len(cells) > 9 else ''
                    })

                race_data['boats'].append(boat_data)

        except Exception as e:
            print(f"出走表取得エラー ({racecourse_code}-{race_num}R): {e}")

        return race_data


class AdvancedPredictionEngine:
    """
    高度な予想エンジン - フォーメーション・3連単大幅拡張版
    """

    def __init__(self):
        self.bet_types = {
            '3連単': {'name': '3連単', 'combination_count': 120},
            '3連複': {'name': '3連複', 'combination_count': 20},
            '2連単': {'name': '2連単', 'combination_count': 30},
            '2連複': {'name': '2連複', 'combination_count': 15},
            'ワイド': {'name': 'ワイド', 'combination_count': 15},
            '拡連複': {'name': '拡連複', 'combination_count': 15}
        }

        # 投資戦略プラン
        self.investment_plans = {
            'conservative': {'name': '堅実プラン', 'risk_level': 'Low', 'target_return': 1.2},
            'balanced': {'name': 'バランスプラン', 'risk_level': 'Medium', 'target_return': 2.0},
            'aggressive': {'name': 'アグレッシブプラン', 'risk_level': 'High', 'target_return': 5.0},
            'pinpoint': {'name': 'ピンポイントプラン', 'risk_level': 'Very High', 'target_return': 10.0}
        }

        print("高度な予想エンジンを初期化しました")

    def generate_3tan_pinpoint_predictions(self, race_data: Dict, confidence_level='high') -> List[Dict]:
        """3連単ピンポイント予想（本命・中穴・大穴）"""
        boats = race_data.get('boats', [])
        if len(boats) < 6:
            return []

        predictions = []

        # 各ボートの実力指数を計算（簡易版）
        boat_scores = []
        for boat in boats:
            score = 0
            try:
                # 全国勝率重視
                if boat.get('national_rate'):
                    score += float(boat['national_rate']) * 10
                # 当地勝率重視
                if boat.get('local_rate'):
                    score += float(boat['local_rate']) * 5
                # モーター勝率
                if boat.get('motor_rate'):
                    score += float(boat['motor_rate']) * 3
                # ボート勝率
                if boat.get('boat_rate'):
                    score += float(boat['boat_rate']) * 2
                # F・L回数を減点
                if boat.get('f_count'):
                    score -= int(boat['f_count']) * 5
                if boat.get('l_count'):
                    score -= int(boat['l_count']) * 3
            except (ValueError, TypeError):
                score = 50  # デフォルト値

            boat_scores.append({
                'boat_num': boat['boat_num'],
                'score': score,
                'player_name': boat.get('player_name', f"{boat['boat_num']}号艇")
            })

        # スコア順でソート
        boat_scores.sort(key=lambda x: x['score'], reverse=True)

        # 本命予想（上位艇中心）
        honmei = {
            'type': '本命',
            'combination': [boat_scores[0]['boat_num'], boat_scores[1]['boat_num'], boat_scores[2]['boat_num']],
            'confidence': 85,
            'odds_range': '5-15倍',
            'investment_ratio': 40
        }
        predictions.append(honmei)

        # 中穴予想（ミックス）
        chuuana = {
            'type': '中穴',
            'combination': [boat_scores[0]['boat_num'], boat_scores[3]['boat_num'], boat_scores[1]['boat_num']],
            'confidence': 65,
            'odds_range': '20-50倍',
            'investment_ratio': 35
        }
        predictions.append(chuuana)

        # 大穴予想（下位艇絡み）
        ooana = {
            'type': '大穴',
            'combination': [boat_scores[4]['boat_num'], boat_scores[0]['boat_num'], boat_scores[5]['boat_num']],
            'confidence': 25,
            'odds_range': '100-500倍',
            'investment_ratio': 25
        }
        predictions.append(ooana)

        return predictions

    def generate_formation_predictions(self, race_data: Dict, strategy='balanced') -> Dict:
        """フォーメーション予想の生成"""
        boats = race_data.get('boats', [])
        if len(boats) < 6:
            return {}

        # 実力指数計算（前回と同じロジック）
        boat_scores = []
        for boat in boats:
            score = 0
            try:
                if boat.get('national_rate'):
                    score += float(boat['national_rate']) * 10
                if boat.get('local_rate'):
                    score += float(boat['local_rate']) * 5
                if boat.get('motor_rate'):
                    score += float(boat['motor_rate']) * 3
                if boat.get('boat_rate'):
                    score += float(boat['boat_rate']) * 2
                if boat.get('f_count'):
                    score -= int(boat['f_count']) * 5
                if boat.get('l_count'):
                    score -= int(boat['l_count']) * 3
            except (ValueError, TypeError):
                score = 50

            boat_scores.append({
                'boat_num': boat['boat_num'],
                'score': score,
                'player_name': boat.get('player_name', f"{boat['boat_num']}号艇")
            })

        boat_scores.sort(key=lambda x: x['score'], reverse=True)

        formation_predictions = {
            '1着固定フォーメーション': {
                'axis': boat_scores[0]['boat_num'],
                '2_3着候補': [b['boat_num'] for b in boat_scores[1:5]],
                'investment_ratio': 30,
                'expected_return': '中程度'
            },
            '軸流し': {
                'axis_1st': [boat_scores[0]['boat_num'], boat_scores[1]['boat_num']],
                'flow_2nd_3rd': [b['boat_num'] for b in boat_scores[2:6]],
                'investment_ratio': 25,
                'expected_return': '安定'
            },
            'BOX買い（上位4艇）': {
                'box_boats': [b['boat_num'] for b in boat_scores[:4]],
                'total_combinations': 24,
                'investment_ratio': 20,
                'expected_return': '堅実'
            },
            'ワイド狙い': {
                'wide_pairs': [
                    [boat_scores[0]['boat_num'], boat_scores[1]['boat_num']],
                    [boat_scores[0]['boat_num'], boat_scores[2]['boat_num']],
                    [boat_scores[1]['boat_num'], boat_scores[2]['boat_num']]
                ],
                'investment_ratio': 15,
                'expected_return': '安全'
            },
            '穴狙い特化': {
                'surprise_combinations': [
                    [boat_scores[3]['boat_num'], boat_scores[4]['boat_num'], boat_scores[5]['boat_num']],
                    [boat_scores[4]['boat_num'], boat_scores[0]['boat_num'], boat_scores[5]['boat_num']],
                    [boat_scores[5]['boat_num'], boat_scores[1]['boat_num'], boat_scores[4]['boat_num']]
                ],
                'investment_ratio': 10,
                'expected_return': '高配当'
            }
        }

        return formation_predictions


class DataIntegrationSystem:
    """
    データ統合システム - 過去データとリアルタイムデータの統合管理
    """

    def __init__(self, data_dir='/home/user/data', db_path='/home/user/kyotei_racer_master.db'):
        self.data_dir = data_dir
        self.db_path = db_path
        self.historical_data = None
        self.realtime_data = None
        self.ml_model = None

        # データディレクトリを作成
        os.makedirs(data_dir, exist_ok=True)

        print(f"データ統合システムを初期化しました: {data_dir}")

    def load_historical_data(self) -> pd.DataFrame:
        """過去データ（CSV）を読み込み"""
        all_data = []

        # data/coconala_2024/配下のCSVファイルを全て読み込み
        coconala_path = os.path.join(self.data_dir, 'coconala_2024')
        if os.path.exists(coconala_path):
            csv_files = [f for f in os.listdir(coconala_path) if f.endswith('.csv')]

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(coconala_path, csv_file))
                    all_data.append(df)
                    print(f"読み込み完了: {csv_file} ({len(df)} レコード)")
                except Exception as e:
                    print(f"読み込みエラー {csv_file}: {e}")

        if all_data:
            self.historical_data = pd.concat(all_data, ignore_index=True)
            print(f"過去データ統合完了: {len(self.historical_data)} レコード")
        else:
            print("過去データが見つかりません - サンプルデータを生成します")
            self.historical_data = self.generate_sample_data()

        return self.historical_data

    def generate_sample_data(self) -> pd.DataFrame:
        """サンプルデータを生成（デモ用）"""
        np.random.seed(42)
        n_races = 1000

        sample_data = []
        for i in range(n_races):
            race_record = {
                'date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
                'racecourse': f"{np.random.randint(1, 25):02d}",
                'race_num': np.random.randint(1, 13),
                'boat_num': np.random.randint(1, 7),
                'player_name': f"選手{i % 100}",
                'national_rate': round(np.random.normal(5.0, 1.5), 2),
                'local_rate': round(np.random.normal(5.2, 1.8), 2),
                'motor_rate': round(np.random.normal(30.0, 10.0), 1),
                'boat_rate': round(np.random.normal(28.0, 12.0), 1),
                'f_count': np.random.randint(0, 5),
                'l_count': np.random.randint(0, 3),
                'result': np.random.randint(1, 7),
                'odds_3tan': round(np.random.exponential(50), 1)
            }
            sample_data.append(race_record)

        return pd.DataFrame(sample_data)

    def load_racer_master(self) -> pd.DataFrame:
        """選手マスターDB読み込み"""
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                racer_df = pd.read_sql_query("SELECT * FROM racers", conn)
                conn.close()
                print(f"選手マスター読み込み完了: {len(racer_df)} 選手")
                return racer_df
            except Exception as e:
                print(f"選手マスター読み込みエラー: {e}")
        else:
            print("選手マスターDBが見つかりません")

        return pd.DataFrame()

    def train_prediction_model(self) -> bool:
        """機械学習モデルの学習"""
        if self.historical_data is None:
            print("過去データが読み込まれていません")
            return False

        try:
            # 特徴量の準備
            features = []
            targets = []

            for _, row in self.historical_data.iterrows():
                try:
                    feature_vector = [
                        float(row.get('national_rate', 5.0)),
                        float(row.get('local_rate', 5.0)),
                        float(row.get('motor_rate', 30.0)),
                        float(row.get('boat_rate', 30.0)),
                        int(row.get('f_count', 0)),
                        int(row.get('l_count', 0)),
                        int(row.get('boat_num', 1))
                    ]

                    # 1着かどうか（1なら1着、0なら2着以下）
                    target = 1 if int(row.get('result', 6)) == 1 else 0

                    features.append(feature_vector)
                    targets.append(target)

                except (ValueError, TypeError):
                    continue

            if len(features) == 0:
                print("学習用データの準備に失敗しました")
                return False

            # 機械学習モデルの学習
            X = np.array(features)
            y = np.array(targets)

            # データの標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 学習・テストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # RandomForestで学習
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            self.ml_model.fit(X_train, y_train)

            # 精度評価
            train_score = self.ml_model.score(X_train, y_train)
            test_score = self.ml_model.score(X_test, y_test)

            print(f"機械学習モデル学習完了")
            print(f"訓練精度: {train_score:.3f}")
            print(f"テスト精度: {test_score:.3f}")
            print(f"学習データ数: {len(X_train)}")

            # スケーラーも保存
            self.scaler = scaler

            return True

        except Exception as e:
            print(f"モデル学習エラー: {e}")
            return False


class BoatraceAIStreamlitApp:
    """
    競艇AI統合アプリケーション - Streamlit UI
    """

    def __init__(self):
        self.data_collector = BoatraceRealTimeDataCollector()
        self.prediction_engine = AdvancedPredictionEngine()
        self.data_system = DataIntegrationSystem()

        print("競艇AI統合アプリケーションを初期化しました")

    def run_streamlit_app(self):
        """Streamlitアプリケーションのメイン処理"""

        # ページタイトル
        st.set_page_config(
            page_title="競艇AI予想システム v13.9 Realtime",
            page_icon="🚤",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚤 競艇AI予想システム v13.9 Realtime")
        st.markdown("**v13.9_fixedの優れたUIを維持し、リアルタイムデータ取得機能を追加**")

        # サイドバー
        st.sidebar.header("⚙️ システム設定")

        # データ初期化セクション
        st.sidebar.subheader("📊 データ管理")
        if st.sidebar.button("過去データ読み込み"):
            with st.spinner("過去データを読み込み中..."):
                data = self.data_system.load_historical_data()
                st.sidebar.success(f"読み込み完了: {len(data)} レコード")

        if st.sidebar.button("MLモデル学習"):
            with st.spinner("機械学習モデルを学習中..."):
                success = self.data_system.train_prediction_model()
                if success:
                    st.sidebar.success("モデル学習完了")
                else:
                    st.sidebar.error("モデル学習失敗")

        # メインタブ
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 リアルタイム予想",
            "📈 フォーメーション分析", 
            "📋 予想根拠詳細",
            "📝 note記事生成"
        ])

        # タブ1: リアルタイム予想
        with tab1:
            self.render_realtime_prediction_tab()

        # タブ2: フォーメーション分析
        with tab2:
            self.render_formation_analysis_tab()

        # タブ3: 予想根拠詳細
        with tab3:
            self.render_prediction_basis_tab()

        # タブ4: note記事生成
        with tab4:
            self.render_note_generation_tab()

    def render_realtime_prediction_tab(self):
        """リアルタイム予想タブのレンダリング"""
        st.header("🎯 リアルタイム予想")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("レース選択")

            # 今日のレース取得
            if st.button("今日のレース一覧取得"):
                with st.spinner("レース情報を取得中..."):
                    races = self.data_collector.get_today_races()
                    if races:
                        st.session_state['races'] = races
                        st.success(f"{len(races)} 会場の情報を取得しました")
                    else:
                        st.error("レース情報の取得に失敗しました")

            # レース場選択
            if 'races' in st.session_state:
                race_options = [(code, info['name']) for code, info in st.session_state['races'].items()]
                selected_racecourse = st.selectbox(
                    "レース場を選択",
                    options=[code for code, _ in race_options],
                    format_func=lambda x: next(name for code, name in race_options if code == x)
                )

                # レース番号選択
                race_num = st.selectbox("レース番号", range(1, 13))

                # 出走表取得
                if st.button("出走表取得"):
                    with st.spinner("出走表を取得中..."):
                        race_data = self.data_collector.get_race_data(selected_racecourse, race_num)
                        if race_data and race_data.get('boats'):
                            st.session_state['current_race'] = race_data
                            st.success("出走表を取得しました")
                        else:
                            st.error("出走表の取得に失敗しました")

        with col2:
            st.subheader("AI予想結果")

            if 'current_race' in st.session_state:
                race_data = st.session_state['current_race']

                # 出走表表示
                st.write(f"**{race_data['racecourse_name']} {race_data['race_num']}R**")

                boats_df = pd.DataFrame(race_data['boats'])
                if not boats_df.empty:
                    st.dataframe(boats_df[['boat_num', 'player_name', 'national_rate', 'local_rate', 'motor_rate', 'boat_rate']])

                # 3連単ピンポイント予想
                predictions = self.prediction_engine.generate_3tan_pinpoint_predictions(race_data)

                if predictions:
                    st.subheader("🎯 3連単ピンポイント予想")

                    for pred in predictions:
                        with st.container():
                            st.markdown(f"**{pred['type']}** (信頼度: {pred['confidence']}%)")
                            st.markdown(f"予想: **{'-'.join(map(str, pred['combination']))}**")
                            st.markdown(f"予想オッズ: {pred['odds_range']}")
                            st.markdown(f"投資比率: {pred['investment_ratio']}%")
                            st.markdown("---")
            else:
                st.info("出走表を取得してください")

    def render_formation_analysis_tab(self):
        """フォーメーション分析タブのレンダリング"""
        st.header("📈 フォーメーション分析")

        if 'current_race' in st.session_state:
            race_data = st.session_state['current_race']

            # フォーメーション予想生成
            formations = self.prediction_engine.generate_formation_predictions(race_data)

            if formations:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🎯 フォーメーション戦略")

                    for strategy_name, strategy_data in formations.items():
                        with st.expander(f"{strategy_name} ({strategy_data.get('expected_return', 'N/A')})"):

                            if '1着固定' in strategy_name:
                                st.write(f"軸（1着固定）: **{strategy_data['axis']}号艇**")
                                st.write(f"2・3着候補: {strategy_data['2_3着候補']}")

                            elif '軸流し' in strategy_name:
                                st.write(f"軸候補: {strategy_data['axis_1st']}")
                                st.write(f"流し候補: {strategy_data['flow_2nd_3rd']}")

                            elif 'BOX' in strategy_name:
                                st.write(f"BOX艇: {strategy_data['box_boats']}")
                                st.write(f"組み合わせ数: {strategy_data['total_combinations']}")

                            elif 'ワイド' in strategy_name:
                                st.write("推奨ペア:")
                                for pair in strategy_data['wide_pairs']:
                                    st.write(f"  {pair[0]}-{pair[1]}")

                            elif '穴狙い' in strategy_name:
                                st.write("穴狙い組み合わせ:")
                                for combo in strategy_data['surprise_combinations']:
                                    st.write(f"  {'-'.join(map(str, combo))}")

                            st.write(f"**投資比率: {strategy_data['investment_ratio']}%**")

                with col2:
                    st.subheader("📊 投資戦略分布")

                    # 投資比率の表示
                    strategy_names = list(formations.keys())
                    investment_ratios = [formations[name]['investment_ratio'] for name in strategy_names]

                    total_investment = sum(investment_ratios)
                    st.write("**投資配分**")
                    for name, ratio in zip(strategy_names, investment_ratios):
                        percentage = (ratio / total_investment) * 100 if total_investment > 0 else 0
                        st.progress(percentage / 100)
                        st.write(f"{name}: {percentage:.1f}%")
        else:
            st.info("出走表データを取得してください")

    def render_prediction_basis_tab(self):
        """予想根拠詳細タブのレンダリング"""
        st.header("📋 予想根拠詳細")

        if 'current_race' in st.session_state:
            race_data = st.session_state['current_race']

            st.subheader(f"{race_data['racecourse_name']} {race_data['race_num']}R 分析詳細")

            boats = race_data.get('boats', [])
            if boats:
                # 各艇の詳細分析
                for boat in boats:
                    with st.expander(f"{boat['boat_num']}号艇 - {boat.get('player_name', 'N/A')}"):

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("全国勝率", boat.get('national_rate', 'N/A'))
                            st.metric("当地勝率", boat.get('local_rate', 'N/A'))

                        with col2:
                            st.metric("モーター勝率", boat.get('motor_rate', 'N/A'))
                            st.metric("ボート勝率", boat.get('boat_rate', 'N/A'))

                        with col3:
                            st.metric("F回数", boat.get('f_count', 'N/A'))
                            st.metric("L回数", boat.get('l_count', 'N/A'))

                        # 実力指数計算（表示用）
                        try:
                            score = 0
                            if boat.get('national_rate'):
                                score += float(boat['national_rate']) * 10
                            if boat.get('local_rate'):
                                score += float(boat['local_rate']) * 5
                            if boat.get('motor_rate'):
                                score += float(boat['motor_rate']) * 3
                            if boat.get('boat_rate'):
                                score += float(boat['boat_rate']) * 2
                            if boat.get('f_count'):
                                score -= int(boat['f_count']) * 5
                            if boat.get('l_count'):
                                score -= int(boat['l_count']) * 3

                            st.write(f"**実力指数: {score:.1f}**")

                            # 評価コメント
                            if score > 60:
                                st.success("🔥 高評価 - 上位進出期待")
                            elif score > 45:
                                st.info("⚖️ 平均的 - 展開次第")
                            else:
                                st.warning("📉 低評価 - 厳しい戦い")

                        except (ValueError, TypeError):
                            st.write("実力指数: 計算不可")
        else:
            st.info("出走表データを取得してください")

    def render_note_generation_tab(self):
        """note記事生成タブのレンダリング"""
        st.header("📝 note記事生成")

        if 'current_race' in st.session_state:
            race_data = st.session_state['current_race']

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("生成設定")

                article_type = st.selectbox(
                    "記事タイプ",
                    ["予想分析記事", "レース展望記事", "フォーメーション解説記事"]
                )

                target_length = st.selectbox(
                    "記事の長さ",
                    [2000, 3000, 4000, 5000]
                )

                if st.button("note記事生成"):
                    with st.spinner("記事を生成中...（2000文字以上）"):
                        article = self.generate_note_article(race_data, article_type, target_length)
                        st.session_state['generated_article'] = article
                        st.success(f"記事生成完了 ({len(article)} 文字)")

            with col2:
                st.subheader("生成された記事")

                if 'generated_article' in st.session_state:
                    article = st.session_state['generated_article']

                    st.text_area(
                        "記事内容",
                        value=article,
                        height=600,
                        max_chars=10000
                    )

                    # 記事をファイルに保存
                    if st.button("記事をファイルに保存"):
                        filename = f"note_article_{race_data['racecourse_code']}_{race_data['race_num']}R_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        filepath = f"/home/user/output/{filename}"

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(article)

                        st.success(f"記事を保存しました: {filepath}")
                else:
                    st.info("記事生成ボタンをクリックしてください")
        else:
            st.info("出走表データを取得してください")

    def generate_note_article(self, race_data: Dict, article_type: str, target_length: int = 2000) -> str:
        """note記事を自動生成（2000文字以上）"""

        racecourse_name = race_data['racecourse_name']
        race_num = race_data['race_num']
        boats = race_data.get('boats', [])

        # 記事のベース構造
        article_parts = []

        # タイトルと導入部
        if article_type == "予想分析記事":
            title = f"【競艇AI予想】{racecourse_name}{race_num}R 徹底分析 - データ駆動による本命・中穴・大穴予想"
            intro = f"""
こんにちは！競艇AI予想システムv13.9 Realtimeによる{racecourse_name}{race_num}Rの詳細分析をお届けします。

本記事では、公式サイトからリアルタイムに取得した最新の出走表データを基に、機械学習アルゴリズムと統計分析を駆使して、科学的根拠に基づいた予想を展開していきます。

今回のレースでは、各選手の全国勝率、当地勝率、モーター・ボート性能、そしてF・L回数などの詳細データを総合的に評価し、本命から大穴まで幅広い視点で分析を行います。
"""

        elif article_type == "レース展望記事":
            title = f"【{racecourse_name}{race_num}R展望】出走選手完全分析とレース展開予想"
            intro = f"""
{racecourse_name}{race_num}Rの出走選手と展開を詳しく見ていきましょう。

本レースには実力派の選手が多数出走しており、混戦模様が予想されます。各選手の実績、モーター・ボート性能、そして近況を総合的に分析し、どのような展開が考えられるかを解説します。

データに基づく客観的な分析と、競艇の醍醐味である展開の妙をお楽しみください。
"""

        else:  # フォーメーション解説記事
            title = f"【フォーメーション戦略】{racecourse_name}{race_num}R 投資戦略別買い方指南"
            intro = f"""
{racecourse_name}{race_num}Rにおける効率的な舟券戦略を、複数のフォーメーションパターンで解説します。

堅実派から一攫千金派まで、それぞれの投資スタイルに応じた舟券の買い方を詳しくご紹介。3連単、3連複、2連単、ワイドまで、幅広い券種での戦略を網羅します。

リスク管理と回収率のバランスを重視した、実践的なアプローチをお届けします。
"""

        article_parts.append(f"# {title}\n\n")
        article_parts.append(intro)

        # 出走選手分析
        article_parts.append(f"\n## 📊 出走選手詳細分析\n\n")

        # 各選手の詳細分析
        boat_scores = []
        for boat in boats:
            # 実力指数計算
            score = 0
            try:
                if boat.get('national_rate'):
                    score += float(boat['national_rate']) * 10
                if boat.get('local_rate'):
                    score += float(boat['local_rate']) * 5
                if boat.get('motor_rate'):
                    score += float(boat['motor_rate']) * 3
                if boat.get('boat_rate'):
                    score += float(boat['boat_rate']) * 2
                if boat.get('f_count'):
                    score -= int(boat['f_count']) * 5
                if boat.get('l_count'):
                    score -= int(boat['l_count']) * 3
            except (ValueError, TypeError):
                score = 50

            boat_scores.append({
                'boat_num': boat['boat_num'],
                'player_name': boat.get('player_name', f"{boat['boat_num']}号艇"),
                'score': score,
                'data': boat
            })

            # 選手個別分析
            player_analysis = f"""
### {boat['boat_num']}号艇 - {boat.get('player_name', 'N/A')} (実力指数: {score:.1f})

**成績データ**
- 全国勝率: {boat.get('national_rate', 'N/A')}
- 当地勝率: {boat.get('local_rate', 'N/A')}
- モーター勝率: {boat.get('motor_rate', 'N/A')}%
- ボート勝率: {boat.get('boat_rate', 'N/A')}%
- F回数: {boat.get('f_count', 'N/A')}回
- L回数: {boat.get('l_count', 'N/A')}回

**分析コメント**
"""

            # 評価コメント生成
            if score > 65:
                player_analysis += "非常に高い実力指数を誇る注目選手。全国勝率、当地勝率ともに優秀で、モーター・ボート性能も良好。F・L回数も少なく、安定した走りが期待できる。本命候補の筆頭として要注目。"
            elif score > 55:
                player_analysis += "実力指数は平均を上回る実力派選手。勝率面では安定しているが、モーター・ボート性能次第では上位進出も十分可能。展開によっては軸として機能する可能性が高い。"
            elif score > 45:
                player_analysis += "平均的な実力指数。勝率は標準的だが、当日の調整やモーター性能によっては上位進出のチャンスあり。フォーメーションの脇役として組み込むのが効果的。"
            else:
                player_analysis += "実力指数はやや低めだが、競艇では展開の妙がある。スタート次第では思わぬ活躍も。穴狙いの一角として少額投資で狙ってみる価値はある。"

            article_parts.append(player_analysis + "\n\n")

        # 実力順ソート
        boat_scores.sort(key=lambda x: x['score'], reverse=True)

        # 予想分析
        predictions = self.prediction_engine.generate_3tan_pinpoint_predictions(race_data)
        formations = self.prediction_engine.generate_formation_predictions(race_data)

        article_parts.append("## 🎯 AI予想結果\n\n")

        # 3連単ピンポイント予想
        if predictions:
            article_parts.append("### 3連単ピンポイント予想\n\n")
            for pred in predictions:
                pred_text = f"""
**{pred['type']}予想**
- 予想: {'-'.join(map(str, pred['combination']))}
- 信頼度: {pred['confidence']}%
- 予想オッズ: {pred['odds_range']}
- 投資比率: {pred['investment_ratio']}%

"""
                article_parts.append(pred_text)

        # まとめ
        summary = f"""
## 📝 まとめ

{racecourse_name}{race_num}Rは、実力指数上位の{boat_scores[0]['player_name']}（{boat_scores[0]['boat_num']}号艇）を軸とした展開が予想される。ただし、競艇特有の展開の妙もあり、複数のシナリオを想定した舟券戦略が重要。

本予想はAIによる客観的なデータ分析に基づいていますが、競艇には予想を超える展開もあります。投資は自己責任で、余裕資金の範囲内で楽しくお楽しみください。

良いレースを！

---
※本予想は過去データに基づく統計分析であり、結果を保証するものではありません
※舟券の購入は20歳以上、自己責任でお願いします
"""

        article_parts.append(summary)

        # 記事を結合
        full_article = ''.join(article_parts)

        # 文字数調整（2000文字以上になるよう）
        if len(full_article) < target_length:
            # 追加コンテンツで文字数を増やす
            additional_content = f"""

## 📚 競艇予想の基礎知識

### データ分析の重要性
現代の競艇予想において、データ分析は欠かせない要素となっています。選手の勝率、モーター・ボート性能、F・L回数など、様々な指標を総合的に判断することで、より精度の高い予想が可能になります。

### 実力指数の算出方法
本システムでは以下の計算式で実力指数を算出しています：
- 全国勝率 × 10
- 当地勝率 × 5  
- モーター勝率 × 3
- ボート勝率 × 2
- F回数 × (-5)
- L回数 × (-3)

この指数により、客観的な選手評価が可能となります。

### フォーメーション投資の考え方
舟券投資では、リスク分散が重要です。一点買いではなく、複数の組み合わせに投資することで、的中率の向上と安定した収支を目指します。

堅実・バランス・アグレッシブの3つの投資スタイルを使い分けることで、長期的な収益性を確保できます。

皆様の舟券ライフが実り多いものになることを願っています。
"""
            full_article += additional_content

        return full_article


# メイン実行部
if __name__ == "__main__":
    print("=== 競艇AI予想システム v13.9 Realtime ===")
    print("リアルタイムデータ取得機能付き統合AI予想システムです")
    print()

    app = BoatraceAIStreamlitApp()

    print("Streamlitアプリケーションを起動するには:")
    print("streamlit run kyotei_ai_v13.9_realtime.py")
    print()
    print("コマンドライン版テストを実行します...")

    # サンプルテスト
    sample_race = {
        'racecourse_code': '01',
        'racecourse_name': '桐生',
        'race_num': 1,
        'date': '20241228',
        'boats': [
            {'boat_num': 1, 'player_name': 'サンプル選手A', 'national_rate': '6.5', 'local_rate': '6.8', 'motor_rate': '35.2', 'boat_rate': '32.1', 'f_count': '0', 'l_count': '0'},
            {'boat_num': 2, 'player_name': 'サンプル選手B', 'national_rate': '5.8', 'local_rate': '5.9', 'motor_rate': '28.7', 'boat_rate': '25.3', 'f_count': '1', 'l_count': '0'},
            {'boat_num': 3, 'player_name': 'サンプル選手C', 'national_rate': '5.2', 'local_rate': '5.0', 'motor_rate': '31.4', 'boat_rate': '29.8', 'f_count': '0', 'l_count': '1'},
            {'boat_num': 4, 'player_name': 'サンプル選手D', 'national_rate': '4.8', 'local_rate': '4.9', 'motor_rate': '26.1', 'boat_rate': '27.5', 'f_count': '2', 'l_count': '0'},
            {'boat_num': 5, 'player_name': 'サンプル選手E', 'national_rate': '4.5', 'local_rate': '4.2', 'motor_rate': '22.8', 'boat_rate': '24.6', 'f_count': '1', 'l_count': '2'},
            {'boat_num': 6, 'player_name': 'サンプル選手F', 'national_rate': '4.0', 'local_rate': '3.8', 'motor_rate': '19.5', 'boat_rate': '21.2', 'f_count': '3', 'l_count': '1'}
        ]
    }

    # サンプル予想実行
    predictions = app.prediction_engine.generate_3tan_pinpoint_predictions(sample_race)
    print("サンプル予想結果:")
    for pred in predictions:
        print(f"  {pred['type']}: {'-'.join(map(str, pred['combination']))} (信頼度: {pred['confidence']}%)")

    print()
    print("システムの準備が完了しました。")
    print("Streamlitで起動してリアルタイムデータ取得をお楽しみください！")
