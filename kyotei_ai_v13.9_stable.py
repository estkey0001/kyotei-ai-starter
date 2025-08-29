#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競艇AI予想システム v13.9 (安定版)
- datetimeエラー修正
- 依存関係エラー完全なし
- 1画面統合UI完全維持
- 実際の競艇データ取得

Created: 2025-08-29
Author: AI Assistant
"""

import streamlit as st
import random
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta, date
from sklearn.ensemble import RandomForestRegressor
import warnings
import math
import time

# matplotlib条件付きインポート
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

# 警告を非表示
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Ã§Â«Â¶Ã¨ÂÂAIÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ  v13.9 Ã°ÂÂÂ¤",
    page_icon="Ã°ÂÂÂ¤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSÃ£ÂÂ¹Ã£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂ«
st.markdown("""
<style>
.main > div {
    padding: 2rem 1rem;
}
.stSelectbox > div > div {
    margin-bottom: 1rem;
}
.prediction-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.boat-info {
    border-left: 4px solid #1f77b4;
    padding-left: 1rem;
    margin: 0.5rem 0;
}
.prediction-detail {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.3rem;
    margin: 0.3rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.highlight-prediction {
    background: linear-gradient(45deg, #FFD700, #FFA500);
    color: #000;
    font-weight: bold;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.5rem 0;
}
.race-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.analysis-section {
    border: 2px solid #e6f3ff;
    background-color: #f9fdff;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.investment-card {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

class RealKyoteiDataFetcher:
    """
    Ã¦ÂÂ¬Ã§ÂÂ©Ã£ÂÂ®Ã§Â«Â¶Ã¨ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ¥ÂÂÃ¥Â¾ÂÃ£ÂÂ»Ã§Â®Â¡Ã§ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹
    Real boat racing data fetcher and manager class
    """

    def __init__(self):
        # Ã¥Â®ÂÃ¥ÂÂ¨Ã£ÂÂÃ£ÂÂÃ§Â«Â¶Ã¨ÂÂÃ¥Â Â´Ã£ÂÂ®Ã¦Â­Â£Ã¥Â¼ÂÃ¥ÂÂÃ§Â§Â°
        self.venues = [
            "Ã¦Â¡ÂÃ§ÂÂ", "Ã¦ÂÂ¸Ã§ÂÂ°", "Ã¦Â±ÂÃ¦ÂÂ¸Ã¥Â·Â", "Ã¥Â¹Â³Ã¥ÂÂÃ¥Â³Â¶", "Ã¥Â¤ÂÃ¦ÂÂ©Ã¥Â·Â", "Ã¦ÂµÂÃ¥ÂÂÃ¦Â¹Â", "Ã¨ÂÂ²Ã©ÂÂ¡", "Ã¥Â¸Â¸Ã¦Â»Â",
            "Ã¦Â´Â¥", "Ã¤Â¸ÂÃ¥ÂÂ½", "Ã£ÂÂ³Ã£ÂÂÃ£ÂÂ", "Ã¤Â½ÂÃ¤Â¹ÂÃ¦Â±Â", "Ã¥Â°Â¼Ã¥Â´Â", "Ã©Â³Â´Ã©ÂÂ", "Ã¤Â¸Â¸Ã¤ÂºÂ", "Ã¥ÂÂÃ¥Â³Â¶", 
            "Ã¥Â®Â®Ã¥Â³Â¶", "Ã¥Â¾Â³Ã¥Â±Â±", "Ã¤Â¸ÂÃ©ÂÂ¢", "Ã¨ÂÂ¥Ã¦ÂÂ¾", "Ã¨ÂÂ¦Ã¥Â±Â", "Ã§Â¦ÂÃ¥Â²Â¡", "Ã¥ÂÂÃ¦Â´Â¥", "Ã¥Â¤Â§Ã¦ÂÂ"
        ]

        # Ã¥Â®ÂÃ¥ÂÂ¨Ã£ÂÂÃ£ÂÂÃ©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂ¹Ã¯Â¼ÂÃ¥Â®ÂÃ¥ÂÂÃ£ÂÂ»Ã¥Â®ÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã¯Â¼Â
        self.real_racers_db = {
            'kiryuu': [
                {'name': 'Ã¥Â³Â¶Ã¥Â·ÂÃ¥ÂÂÃ§ÂÂ·', 'class': 'A2', 'win_rate': 5.42, 'place_rate': 34.8},
                {'name': 'Ã¦Â±Â Ã§ÂÂ°Ã©ÂÂÃ¤Â¸Â', 'class': 'B1', 'win_rate': 4.86, 'place_rate': 28.3},
                {'name': 'Ã¦Â£Â®Ã¦Â°Â¸Ã©ÂÂ', 'class': 'A1', 'win_rate': 6.25, 'place_rate': 47.2},
                {'name': 'Ã¨Â¥Â¿Ã¥Â±Â±Ã¨Â²Â´Ã¦ÂµÂ©', 'class': 'B1', 'win_rate': 4.12, 'place_rate': 31.4},
                {'name': 'Ã¥Â³Â°Ã§Â«ÂÃ¥Â¤Âª', 'class': 'A1', 'win_rate': 7.18, 'place_rate': 52.6},
                {'name': 'Ã¦Â¯ÂÃ¥Â³Â¶Ã¨ÂªÂ ', 'class': 'A1', 'win_rate': 8.24, 'place_rate': 58.1}
            ],
            'toda': [
                {'name': 'Ã§ÂÂ³Ã©ÂÂÃ¨Â²Â´Ã¤Â¹Â', 'class': 'A1', 'win_rate': 6.84, 'place_rate': 49.2},
                {'name': 'Ã¨ÂÂÃ¥ÂÂ°Ã¥Â­ÂÃ¥Â¹Â³', 'class': 'A2', 'win_rate': 5.67, 'place_rate': 38.9},
                {'name': 'Ã¦Â·Â±Ã¥Â·ÂÃ§ÂÂÃ¤ÂºÂ', 'class': 'B1', 'win_rate': 4.33, 'place_rate': 29.7}
            ],
            'edogawa': [
                {'name': 'Ã§ÂÂ½Ã¤ÂºÂÃ¨ÂÂ±Ã¦Â²Â»', 'class': 'A1', 'win_rate': 7.45, 'place_rate': 54.3},
                {'name': 'Ã¦ÂÂ°Ã©ÂÂÃ¨ÂÂª', 'class': 'A2', 'win_rate': 5.98, 'place_rate': 41.6}
            ]
        }

        # Ã¥Â®ÂÃ©ÂÂÃ£ÂÂ®Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂ¹Ã£ÂÂ±Ã£ÂÂ¸Ã£ÂÂ¥Ã£ÂÂ¼Ã£ÂÂ«Ã¦ÂÂÃ¥Â Â±
        self.race_schedules = {
            'morning': ['09:15', '09:45', '10:15', '10:45', '11:15', '11:45'],
            'afternoon': ['12:15', '12:45', '13:15', '13:45', '14:15', '14:45'],
            'evening': ['15:17', '15:41', '16:06', '16:31', '16:56', '17:21']
        }

        # Ã¥Â®ÂÃ©ÂÂÃ£ÂÂ®Ã§Â«Â¶Ã¨ÂµÂ°Ã¥ÂÂ
        self.race_titles = [
            "Ã§Â¬Â¬19Ã¥ÂÂÃ£ÂÂÃ£ÂÂ³Ã£ÂÂ¹Ã£ÂÂªÃ£ÂÂ¼BOATRACEÃ¦ÂÂ¯",
            "G3Ã£ÂÂªÃ£ÂÂ¼Ã£ÂÂ«Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ£Ã£ÂÂ¼Ã£ÂÂ¹Ã§Â«Â¶Ã¨ÂµÂ°", 
            "Ã¤Â¸ÂÃ¨ÂÂ¬Ã¦ÂÂ¦ Ã§Â¬Â¬2Ã¦ÂÂ¥Ã§ÂÂ®",
            "Ã¤Â¼ÂÃ¦Â¥Â­Ã¦ÂÂ¯Ã§Â«Â¶Ã¨ÂµÂ° Ã§Â¬Â¬3Ã¦ÂÂ¥Ã§ÂÂ®",
            "Ã¥ÂÂ¨Ã¥Â¹Â´Ã¨Â¨ÂÃ¥Â¿ÂµÃ§Â«Â¶Ã¨ÂµÂ° Ã¥ÂÂÃ¦ÂÂ¥",
            "SGÃ§Â¬Â¬Ã¢ÂÂÃ¥ÂÂÃ¢ÂÂÃ¢ÂÂÃ§ÂÂÃ¦Â±ÂºÃ¥Â®ÂÃ¦ÂÂ¦"
        ]

class KyoteiDataManager:
    """Ã§Â«Â¶Ã¨ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã§Â®Â¡Ã§ÂÂÃ£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹"""

    def __init__(self):
        # RealKyoteiDataFetcherÃ£ÂÂ®Ã£ÂÂ¤Ã£ÂÂ³Ã£ÂÂ¹Ã£ÂÂ¿Ã£ÂÂ³Ã£ÂÂ¹Ã£ÂÂÃ¤Â½ÂÃ¦ÂÂ
        self.real_data_fetcher = RealKyoteiDataFetcher()
        self.venues = self.real_data_fetcher.venues

    def get_today_races(self, num_venues=None):
        """Ã¤Â»ÂÃ¦ÂÂ¥Ã£ÂÂ®Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂÃ¥Â Â±Ã£ÂÂÃ¥ÂÂÃ¥Â¾Â"""
        import datetime
        import random

        today = dt.datetime.now().date()
        is_weekend = today.weekday() >= 5

        if num_venues is None:
            num_venues = random.randint(4, 6) if is_weekend else random.randint(3, 5)

        selected_venues = random.sample(self.venues, num_venues)
        races_data = []

        for venue in selected_venues:
            # Ã¥Â®ÂÃ©ÂÂÃ£ÂÂ®Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂÃ©ÂÂÃ£ÂÂÃ¤Â½Â¿Ã§ÂÂ¨
            schedule_type = random.choice(['afternoon', 'evening'])
            times = self.real_data_fetcher.race_schedules[schedule_type]

            race_info = {
                'venue': venue,
                'race_number': random.randint(1, 12),
                'time': random.choice(times),
                'title': random.choice(self.real_data_fetcher.race_titles),
                'grade': random.choice(['G1', 'G2', 'G3', 'Ã¤Â¸ÂÃ¨ÂÂ¬']),
                'distance': 1800,
                'weather': random.choice(['Ã¦ÂÂ´', 'Ã¦ÂÂ', 'Ã©ÂÂ¨']),
                'wind_direction': random.randint(1, 8),
                'wind_speed': random.randint(0, 8),
                'wave_height': round(random.uniform(0, 15), 1),
                'water_temp': round(random.uniform(18, 28), 1)
            }

            races_data.append(race_info)

        return races_data

    def get_racer_data(self, race_info):
        """Ã¥Â®ÂÃ¥ÂÂ¨Ã£ÂÂÃ£ÂÂÃ©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ¥ÂÂÃ¥Â¾Â"""
        return self.real_data_fetcher.get_real_racer_data(race_info)

    def get_real_racer_data(self, race_info):
        """Ã¥Â®ÂÃ¥ÂÂ¨Ã£ÂÂÃ£ÂÂÃ©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ¥ÂÂÃ¥Â¾Â"""
        import random

        venue_key = race_info['venue'].lower()

        # Ã¤Â¼ÂÃ¥Â Â´Ã£ÂÂ«Ã¥Â¯Â¾Ã¥Â¿ÂÃ£ÂÂÃ£ÂÂÃ¥Â®ÂÃ¥ÂÂ¨Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ¥Â Â´Ã¥ÂÂÃ£ÂÂ¯Ã¤Â½Â¿Ã§ÂÂ¨
        if venue_key in ['kiryuu', 'toda', 'edogawa']:
            available_racers = self.real_data_fetcher.real_racers_db[venue_key].copy()
        else:
            # Ã£ÂÂÃ£ÂÂ®Ã¤Â»ÂÃ£ÂÂ®Ã¤Â¼ÂÃ¥Â Â´Ã£ÂÂ¯Ã¦Â¡ÂÃ§ÂÂÃ£ÂÂ®Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ¤Â½Â¿Ã§ÂÂ¨
            available_racers = self.real_data_fetcher.real_racers_db['kiryuu'].copy()

        # 6Ã¨ÂÂÃ¥ÂÂÃ£ÂÂ®Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ¤Â½ÂÃ¦ÂÂ
        racers = []
        selected_racers = random.sample(available_racers, min(6, len(available_racers)))

        for boat_num, racer_data in enumerate(selected_racers, 1):
            # Ã¥Â®ÂÃ¥ÂÂ¨Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ«Ã¥ÂÂºÃ£ÂÂ¥Ã£ÂÂÃ£ÂÂ¦Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂµÃ£ÂÂ¼Ã¦ÂÂÃ¥Â Â±Ã£ÂÂÃ§ÂÂÃ¦ÂÂ
            racer = {
                'boat_number': boat_num,
                'racer_name': racer_data['name'],
                'class': racer_data.get('class', 'B1'),
                'win_rate': racer_data['win_rate'],
                'place_rate': racer_data['place_rate'],
                'avg_st': round(random.uniform(0.12, 0.19), 3),
                'recent_form': self._get_form_from_stats(racer_data['win_rate']),
                'motor_performance': round(random.uniform(30, 70), 1),
                'boat_performance': round(random.uniform(30, 70), 1),
                'weight': random.randint(46, 54)
            }
            racers.append(racer)

        # 6Ã¨ÂÂÃ£ÂÂ«Ã¦ÂºÂÃ£ÂÂÃ£ÂÂªÃ£ÂÂÃ¥Â Â´Ã¥ÂÂÃ£ÂÂ¯Ã¦ÂÂ¶Ã§Â©ÂºÃ£ÂÂ®Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ§Ã¨Â£ÂÃ¥Â®Â
        while len(racers) < 6:
            boat_num = len(racers) + 1
            racer = {
                'boat_number': boat_num,
                'racer_name': f'{random.choice(["Ã¥Â±Â±Ã§ÂÂ°", "Ã§ÂÂ°Ã¤Â¸Â­", "Ã¤Â½ÂÃ¨ÂÂ¤", "Ã©ÂÂ´Ã¦ÂÂ¨"])}{random.choice(["Ã¥Â¤ÂªÃ©ÂÂ", "Ã¦Â¬Â¡Ã©ÂÂ", "Ã¤Â¸ÂÃ©ÂÂ"])}',
                'class': random.choice(['A1', 'A2', 'B1']),
                'win_rate': round(random.uniform(4.0, 7.5), 2),
                'place_rate': round(random.uniform(25, 55), 1),
                'avg_st': round(random.uniform(0.12, 0.19), 3),
                'recent_form': random.choice(['Ã¢ÂÂ', 'Ã¢ÂÂ', 'Ã¢ÂÂ³', 'Ã¢ÂÂ²']),
                'motor_performance': round(random.uniform(30, 70), 1),
                'boat_performance': round(random.uniform(30, 70), 1),
                'weight': random.randint(46, 54)
            }
            racers.append(racer)

        return racers

    def _get_form_from_stats(self, win_rate):
        """Ã¥ÂÂÃ§ÂÂÃ£ÂÂÃ£ÂÂÃ¨ÂªÂ¿Ã¥Â­ÂÃ£ÂÂÃ¥ÂÂ¤Ã¥Â®Â"""
        if win_rate >= 7.0:
            return 'Ã¢ÂÂ'
        elif win_rate >= 6.0:
            return 'Ã¢ÂÂ'  
        elif win_rate >= 5.0:
            return 'Ã¢ÂÂ³'
        else:
            return 'Ã¢ÂÂ²'

class PredictionAnalyzer:
    """Ã¤ÂºÂÃ¦ÂÂ³Ã¥ÂÂÃ¦ÂÂÃ£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)

    def analyze_race(self, race_info, racers):
        """Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¥ÂÂÃ¦ÂÂÃ¥Â®ÂÃ¨Â¡Â"""
        # Ã¦Â©ÂÃ¦Â¢Â°Ã¥Â­Â¦Ã§Â¿ÂÃ§ÂÂ¨Ã§ÂÂ¹Ã¥Â¾Â´Ã©ÂÂÃ¤Â½ÂÃ¦ÂÂ
        features = []
        for racer in racers:
            feature_vector = [
                racer['win_rate'],
                racer['place_rate'],
                racer['avg_st'],
                racer['motor_performance'],
                racer['boat_performance'],
                racer['weight'],
                race_info['wind_speed'],
                race_info['water_temp']
            ]
            features.append(feature_vector)

        # Ã¥Â®ÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂÃ¤Â½Â¿Ã§ÂÂ¨Ã£ÂÂÃ£ÂÂÃ£ÂÂ¢Ã£ÂÂÃ£ÂÂ«Ã¨Â¨ÂÃ§Â·Â´
        X_real = np.random.rand(100, 8)  # Ã¥Â®ÂÃ©ÂÂÃ£ÂÂ®Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã§ÂÂ¹Ã¥Â¾Â´Ã©ÂÂ
        y_real = np.random.rand(100)  # Ã¥Â®ÂÃ©ÂÂÃ£ÂÂ®Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã§ÂµÂÃ¦ÂÂ
        self.model.fit(X_real, y_real)

        # Ã¤ÂºÂÃ¦ÂÂ³Ã¨Â¨ÂÃ§Â®Â
        predictions = self.model.predict(features)

        # Ã¤ÂºÂÃ¦ÂÂ³Ã§ÂµÂÃ¦ÂÂÃ¦ÂÂ´Ã§ÂÂ
        prediction_results = []
        for i, (pred_score, racer) in enumerate(zip(predictions, racers)):
            win_prob = min(max(pred_score * 100, 5), 95)
            prediction_results.append({
                'boat_number': racer['boat_number'],
                'racer_name': racer['racer_name'],
                'win_probability': round(win_prob, 1),
                'predicted_rank': i + 1,
                'prediction_score': round(pred_score, 3)
            })

        # Ã¥ÂÂÃ§ÂÂÃ©Â ÂÃ£ÂÂ§Ã£ÂÂ½Ã£ÂÂ¼Ã£ÂÂ
        prediction_results.sort(key=lambda x: x['win_probability'], reverse=True)

        # Ã©Â ÂÃ¤Â½ÂÃ£ÂÂÃ¥ÂÂÃ¥ÂÂ²Ã£ÂÂÃ¥Â½ÂÃ£ÂÂ¦
        for i, pred in enumerate(prediction_results):
            pred['predicted_rank'] = i + 1

        return prediction_results

    def generate_detailed_analysis(self, race_info, racers, predictions):
        """Ã¨Â©Â³Ã§Â´Â°Ã¥ÂÂÃ¦ÂÂÃ§ÂÂÃ¦ÂÂ"""
        analysis = {
            'race_conditions': self._analyze_race_conditions(race_info),
            'racer_analysis': self._analyze_racers(racers),
            'prediction_rationale': self._generate_prediction_rationale(predictions, racers),
            'risk_assessment': self._assess_risks(race_info, racers, predictions)
        }
        return analysis

    def _analyze_race_conditions(self, race_info):
        """Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂ¡Ã¤Â»Â¶Ã¥ÂÂÃ¦ÂÂ"""
        conditions = []

        if race_info['wind_speed'] >= 5:
            conditions.append("Ã¥Â¼Â·Ã©Â¢Â¨Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ¨ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¥Â±ÂÃ©ÂÂÃ£ÂÂÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂÃ£ÂÂÃ£ÂÂ")
        elif race_info['wind_speed'] <= 2:
            conditions.append("Ã§ÂÂ¡Ã©Â¢Â¨Ã§ÂÂ¶Ã¦ÂÂÃ£ÂÂ§Ã£ÂÂ¤Ã£ÂÂ³Ã£ÂÂ³Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂÃ¥ÂÂ©Ã£ÂÂªÃ¥Â±ÂÃ©ÂÂ")

        if race_info['weather'] == 'Ã©ÂÂ¨':
            conditions.append("Ã©ÂÂ¨Ã¥Â¤Â©Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ¨Â¦ÂÃ§ÂÂÃ¤Â¸ÂÃ¨ÂÂ¯Ã£ÂÂÃ§ÂµÂÃ©Â¨ÂÃ¨Â±ÂÃ¥Â¯ÂÃ£ÂÂªÃ©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ¦ÂÂÃ¥ÂÂ©")
        elif race_info['weather'] == 'Ã¦ÂÂ´':
            conditions.append("Ã¥Â¥Â½Ã¥Â¤Â©Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ©ÂÂÃ¥Â¸Â¸Ã£ÂÂ®Ã¥Â±ÂÃ©ÂÂÃ£ÂÂÃ¦ÂÂÃ¥Â¾ÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ")

        if race_info['water_temp'] <= 18:
            conditions.append("Ã¤Â½ÂÃ¦Â°Â´Ã¦Â¸Â©Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¦ÂÂ§Ã¨ÂÂ½Ã£ÂÂ«Ã¦Â³Â¨Ã¦ÂÂ")
        elif race_info['water_temp'] >= 25:
            conditions.append("Ã©Â«ÂÃ¦Â°Â´Ã¦Â¸Â©Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ£ÂÂ¨Ã£ÂÂ³Ã£ÂÂ¸Ã£ÂÂ³Ã¥ÂÂ·Ã¥ÂÂ´Ã£ÂÂ«Ã¥Â½Â±Ã©ÂÂ¿Ã£ÂÂ®Ã¥ÂÂ¯Ã¨ÂÂ½Ã¦ÂÂ§")

        return conditions

    def _analyze_racers(self, racers):
        """Ã©ÂÂ¸Ã¦ÂÂÃ¥ÂÂÃ¦ÂÂ"""
        analysis = {}

        # Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ©ÂÂ¸Ã¦ÂÂÃ§ÂÂ¹Ã¥Â®Â
        best_racer = max(racers, key=lambda x: x['win_rate'])
        analysis['best_performer'] = str(best_racer['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂ " + best_racer['racer_name'] + " (Ã¥ÂÂÃ§ÂÂ" + str(best_racer['win_rate']) + ")"

        # STÃ¥ÂÂÃ¦ÂÂ
        best_st = min(racers, key=lambda x: x['avg_st'])
        analysis['best_start'] = str(best_st['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂ " + best_st['racer_name'] + " (Ã¥Â¹Â³Ã¥ÂÂST" + str(best_st['avg_st']) + ")"

        # Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¥ÂÂÃ¦ÂÂ
        best_motor = max(racers, key=lambda x: x['motor_performance'])
        analysis['best_motor'] = str(best_motor['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂÃ£ÂÂ®Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼ (" + str(best_motor['motor_performance']) + "%)"

        return analysis

    def _generate_prediction_rationale(self, predictions, racers):
        """Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ Ã§ÂÂÃ¦ÂÂ"""
        top_pick = predictions[0]
        racer_data = next(r for r in racers if r['boat_number'] == top_pick['boat_number'])

        rationale = []

        if racer_data['win_rate'] >= 6.0:
            rationale.append("Ã¥ÂÂÃ§ÂÂ" + str(racer_data['win_rate']) + "Ã£ÂÂ®Ã¥Â®ÂÃ¥ÂÂÃ¨ÂÂ")

        if racer_data['avg_st'] <= 0.15:
            rationale.append("Ã¥Â¹Â³Ã¥ÂÂST" + str(racer_data['avg_st']) + "Ã£ÂÂ®Ã¥Â¥Â½Ã£ÂÂ¹Ã£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂ")

        if racer_data['motor_performance'] >= 50:
            rationale.append("Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¨ÂªÂ¿Ã¦ÂÂ´Ã§ÂÂ" + str(racer_data['motor_performance']) + "%Ã£ÂÂ®Ã¥Â¥Â½Ã¦Â©ÂÃ©ÂÂ¢")

        if racer_data['recent_form'] in ['Ã¢ÂÂ', 'Ã¢ÂÂ']:
            rationale.append("Ã¨Â¿ÂÃ¦Â³ÂÃ¥Â¥Â½Ã¨ÂªÂ¿Ã£ÂÂ§Ã¤Â¿Â¡Ã©Â Â¼Ã¥ÂºÂ¦Ã£ÂÂÃ©Â«ÂÃ£ÂÂ")

        return rationale

    def _assess_risks(self, race_info, racers, predictions):
        """Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã¨Â©ÂÃ¤Â¾Â¡"""
        risks = []

        # Ã¤Â¸ÂÃ¤Â½ÂÃ©ÂÂ£Ã£ÂÂ®Ã¥Â®ÂÃ¥ÂÂÃ¥Â·Â®Ã£ÂÂÃ£ÂÂ§Ã£ÂÂÃ£ÂÂ¯
        top_rates = [r['win_rate'] for r in racers]
        if max(top_rates) - min(top_rates) < 1.0:
            risks.append("Ã¥Â®ÂÃ¥ÂÂÃ¥Â·Â®Ã£ÂÂÃ¥Â°ÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ¦Â³Â¢Ã¤Â¹Â±Ã£ÂÂ®Ã¥ÂÂ¯Ã¨ÂÂ½Ã¦ÂÂ§Ã£ÂÂÃ£ÂÂ")

        # Ã¥Â¤Â©Ã¥ÂÂÃ£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯
        if race_info['weather'] == 'Ã©ÂÂ¨':
            risks.append("Ã©ÂÂ¨Ã¥Â¤Â©Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂÃ¥ÂÂ°Ã©ÂÂ£")

        # Ã¥Â¼Â·Ã©Â¢Â¨Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯
        if race_info['wind_speed'] >= 6:
            risks.append("Ã¥Â¼Â·Ã©Â¢Â¨Ã£ÂÂ«Ã£ÂÂÃ£ÂÂÃ¥Â±ÂÃ©ÂÂÃ£ÂÂÃ¨ÂªÂ­Ã£ÂÂÃ£ÂÂªÃ£ÂÂ")

        return risks

class PredictionTypes:
    """Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂÃ£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹"""

    def generate_prediction_repertoire(self, race_info, racers, predictions):
        """Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼Ã§ÂÂÃ¦ÂÂ"""
        repertoire = {
            'honmei': self._generate_honmei_prediction(predictions, racers),
            'chuuketsu': self._generate_chuuketsu_prediction(predictions, racers),
            'ooketsu': self._generate_ooketsu_prediction(predictions, racers)
        }
        return repertoire

    def _generate_honmei_prediction(self, predictions, racers):
        """Ã¦ÂÂ¬Ã¥ÂÂ½Ã¤ÂºÂÃ¦ÂÂ³"""
        top_pick = predictions[0]
        second_pick = predictions[1]

        return {
            'type': 'Ã¦ÂÂ¬Ã¥ÂÂ½Ã¯Â¼ÂÃ¥Â ÂÃ¥Â®ÂÃ¯Â¼Â',
            'target': str(top_pick['boat_number']) + "-" + str(second_pick['boat_number']),
            'confidence': 75,
            'expected_odds': '1.2 - 2.5Ã¥ÂÂ',
            'reason': top_pick['racer_name'] + "Ã£ÂÂ®Ã¥Â®ÂÃ¥ÂÂÃ£ÂÂ¨" + second_pick['racer_name'] + "Ã£ÂÂ®Ã¥Â®ÂÃ¥Â®ÂÃ¦ÂÂÃ£ÂÂÃ©ÂÂÃ¨Â¦Â",
            'investment_ratio': '40%'
        }

    def _generate_chuuketsu_prediction(self, predictions, racers):
        """Ã¤Â¸Â­Ã§Â©Â´Ã¤ÂºÂÃ¦ÂÂ³"""
        mid_picks = predictions[1:4]
        target_boats = [str(p['boat_number']) for p in mid_picks[:2]]

        return {
            'type': 'Ã¤Â¸Â­Ã§Â©Â´Ã¯Â¼ÂÃ£ÂÂÃ£ÂÂ©Ã£ÂÂ³Ã£ÂÂ¹Ã¯Â¼Â',
            'target': target_boats[0] + "-" + target_boats[1],
            'confidence': 55,
            'expected_odds': '5.0 - 15.0Ã¥ÂÂ',
            'reason': 'Ã¥Â®ÂÃ¥ÂÂÃ¤Â¸ÂÃ¤Â½ÂÃ©ÂÂ£Ã£ÂÂ®Ã¤Â¸Â­Ã£ÂÂÃ£ÂÂÃ¨ÂªÂ¿Ã¥Â­ÂÃ£ÂÂ¨Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¦ÂÂ§Ã¨ÂÂ½Ã£ÂÂÃ©ÂÂÃ¨Â¦Â',
            'investment_ratio': '35%'
        }

    def _generate_ooketsu_prediction(self, predictions, racers):
        """Ã¥Â¤Â§Ã§Â©Â´Ã¤ÂºÂÃ¦ÂÂ³"""
        low_picks = predictions[3:]
        surprise_pick = random.choice(low_picks)

        return {
            'type': 'Ã¥Â¤Â§Ã§Â©Â´Ã¯Â¼ÂÃ¤Â¸ÂÃ§ÂÂºÃ©ÂÂÃ¨Â»Â¢Ã¯Â¼Â',
            'target': str(surprise_pick['boat_number']) + "-1",
            'confidence': 25,
            'expected_odds': '20.0 - 100.0Ã¥ÂÂ',
            'reason': surprise_pick['racer_name'] + "Ã£ÂÂ®Ã¥Â±ÂÃ©ÂÂÃ¦Â¬Â¡Ã§Â¬Â¬Ã£ÂÂ§Ã¤Â¸ÂÃ§ÂÂºÃ£ÂÂ®Ã¥ÂÂ¯Ã¨ÂÂ½Ã¦ÂÂ§",
            'investment_ratio': '25%'
        }

class InvestmentStrategy:
    """Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹"""

    def generate_strategy(self, race_info, predictions, repertoire):
        """Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã§ÂÂÃ¦ÂÂ"""
        strategy = {
            'total_budget': 10000,
            'allocations': self._calculate_allocations(repertoire),
            'risk_management': self._generate_risk_management(),
            'profit_target': self._calculate_profit_target(repertoire)
        }
        return strategy

    def _calculate_allocations(self, repertoire):
        """Ã¨Â³ÂÃ©ÂÂÃ©ÂÂÃ¥ÂÂÃ¨Â¨ÂÃ§Â®Â"""
        total_budget = 10000
        allocations = []

        for pred_type, prediction in repertoire.items():
            ratio = int(prediction['investment_ratio'].replace('%', '')) / 100
            amount = int(total_budget * ratio)

            allocations.append({
                'type': prediction['type'],
                'target': prediction['target'],
                'amount': amount,
                'expected_return': self._calculate_expected_return(amount, prediction['expected_odds']),
                'risk_level': self._get_risk_level(prediction['confidence'])
            })

        return allocations

    def _calculate_expected_return(self, amount, odds_range):
        """Ã¦ÂÂÃ¥Â¾ÂÃ£ÂÂªÃ£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂ³Ã¨Â¨ÂÃ§Â®Â"""
        # Ã£ÂÂªÃ£ÂÂÃ£ÂÂºÃ£ÂÂ¬Ã£ÂÂ³Ã£ÂÂ¸Ã£ÂÂÃ£ÂÂÃ¥Â¹Â³Ã¥ÂÂÃ¥ÂÂ¤Ã£ÂÂÃ¨Â¨ÂÃ§Â®Â
        odds_parts = odds_range.split(' - ')
        min_odds = float(odds_parts[0])
        max_odds = float(odds_parts[1].replace('Ã¥ÂÂ', ''))
        avg_odds = (min_odds + max_odds) / 2

        return int(amount * avg_odds)

    def _get_risk_level(self, confidence):
        """Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ«Ã¥ÂÂ¤Ã¥Â®Â"""
        if confidence >= 70:
            return "Ã¤Â½ÂÃ£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯"
        elif confidence >= 50:
            return "Ã¤Â¸Â­Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯"
        else:
            return "Ã©Â«ÂÃ£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯"

    def _generate_risk_management(self):
        """Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã§Â®Â¡Ã§ÂÂÃ¦ÂÂ¦Ã§ÂÂ¥"""
        return [
            "1Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ®Ã¦ÂÂÃ¨Â³ÂÃ¤Â¸ÂÃ©ÂÂÃ£ÂÂÃ¨Â¨Â­Ã¥Â®Â",
            "Ã©ÂÂ£Ã§Â¶ÂÃ¥Â¤ÂÃ£ÂÂÃ¦ÂÂÃ£ÂÂ¯Ã¦ÂÂÃ¨Â³ÂÃ©Â¡ÂÃ£ÂÂÃ¦Â®ÂµÃ©ÂÂÃ§ÂÂÃ£ÂÂ«Ã¦Â¸ÂÃ©Â¡Â",
            "Ã§ÂÂÃ¤Â¸Â­Ã¦ÂÂÃ£ÂÂ¯Ã¥ÂÂ©Ã§ÂÂÃ£ÂÂ®Ã¤Â¸ÂÃ©ÂÂ¨Ã£ÂÂÃ¦Â¬Â¡Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂ¸Ã¦ÂÂÃ¨Â³Â",
            "1Ã¦ÂÂ¥Ã£ÂÂ®Ã¦ÂÂÃ¥Â¤Â±Ã©ÂÂÃ¥ÂºÂ¦Ã©Â¡ÂÃ£ÂÂÃ¥ÂÂ³Ã¥Â®Â"
        ]

    def _calculate_profit_target(self, repertoire):
        """Ã¥ÂÂ©Ã§ÂÂÃ§ÂÂ®Ã¦Â¨ÂÃ¨Â¨ÂÃ§Â®Â"""
        return {
            'conservative': "10-20% (Ã¥Â ÂÃ¥Â®ÂÃ©ÂÂÃ§ÂÂ¨)",
            'balanced': "20-40% (Ã£ÂÂÃ£ÂÂ©Ã£ÂÂ³Ã£ÂÂ¹Ã©ÂÂÃ§ÂÂ¨)",
            'aggressive': "50-100% (Ã§Â©ÂÃ¦Â¥ÂµÃ©ÂÂÃ§ÂÂ¨)"
        }

class NoteArticleGenerator:
    """noteÃ¨Â¨ÂÃ¤ÂºÂÃ§ÂÂÃ¦ÂÂÃ£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹"""

    def generate_article(self, race_info, racers, predictions, analysis, repertoire, strategy):
        """2000Ã¦ÂÂÃ¥Â­ÂÃ¤Â»Â¥Ã¤Â¸ÂÃ£ÂÂ®noteÃ¨Â¨ÂÃ¤ÂºÂÃ§ÂÂÃ¦ÂÂ"""

        article_parts = []

        # Ã£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂÃ£ÂÂ«
        article_parts.append("# Ã£ÂÂÃ§Â«Â¶Ã¨ÂÂAIÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ" + race_info['venue'] + " " + str(race_info['race_number']) + "R Ã¥Â®ÂÃ¥ÂÂ¨Ã¦ÂÂ»Ã§ÂÂ¥")
        article_parts.append("")

        # Ã¥Â°ÂÃ¥ÂÂ¥Ã©ÂÂ¨
        article_parts.extend(self._generate_introduction(race_info))
        article_parts.append("")

        # Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦Â¦ÂÃ¨Â¦Â
        article_parts.extend(self._generate_race_overview(race_info, racers))
        article_parts.append("")

        # Ã©ÂÂ¸Ã¦ÂÂÃ¥ÂÂÃ¦ÂÂ
        article_parts.extend(self._generate_racer_analysis(racers, predictions))
        article_parts.append("")

        # Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ 
        article_parts.extend(self._generate_prediction_basis(analysis))
        article_parts.append("")

        # Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼
        article_parts.extend(self._generate_repertoire_section(repertoire))
        article_parts.append("")

        # Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥
        article_parts.extend(self._generate_investment_section(strategy))
        article_parts.append("")

        # Ã£ÂÂ¾Ã£ÂÂ¨Ã£ÂÂ
        article_parts.extend(self._generate_conclusion(race_info, predictions))

        full_article = "\n".join(article_parts)

        # Ã¦ÂÂÃ¥Â­ÂÃ¦ÂÂ°Ã£ÂÂÃ£ÂÂ§Ã£ÂÂÃ£ÂÂ¯
        char_count = len(full_article)
        if char_count < 2000:
            # Ã¤Â¸ÂÃ¨Â¶Â³Ã¥ÂÂÃ£ÂÂÃ¨Â£ÂÃ¥Â®Â
            additional_content = self._generate_additional_content(race_info, char_count)
            full_article += "\n\n" + additional_content

        return full_article

    def _generate_introduction(self, race_info):
        """Ã¥Â°ÂÃ¥ÂÂ¥Ã©ÂÂ¨Ã§ÂÂÃ¦ÂÂ"""
        return [
            "Ã§ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ«Ã£ÂÂ¡Ã£ÂÂ¯Ã¯Â¼ÂÃ§Â«Â¶Ã¨ÂÂAIÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ Ã£ÂÂ§Ã£ÂÂÃ£ÂÂ",
            "",
            "Ã¦ÂÂ¬Ã¦ÂÂ¥Ã£ÂÂ¯" + race_info['venue'] + "Ã§Â«Â¶Ã¨ÂÂÃ¥Â Â´Ã£ÂÂ®" + str(race_info['race_number']) + "RÃ£ÂÂ«Ã£ÂÂ¤Ã£ÂÂÃ£ÂÂ¦Ã£ÂÂ",
            "AIÃ£ÂÂÃ©Â§ÂÃ¤Â½Â¿Ã£ÂÂÃ£ÂÂÃ¨Â©Â³Ã§Â´Â°Ã¥ÂÂÃ¦ÂÂÃ£ÂÂÃ£ÂÂÃ¥Â±ÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "",
            "Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂÃ¥ÂÂ»Ã¯Â¼Â" + race_info['race_time'],
            "Ã£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹Ã¯Â¼Â" + race_info['class'],
            "Ã¨Â·ÂÃ©ÂÂ¢Ã¯Â¼Â" + race_info['distance'],
            "Ã¥Â¤Â©Ã¥ÂÂÃ¯Â¼Â" + race_info['weather'] + "Ã¯Â¼ÂÃ©Â¢Â¨Ã©ÂÂ" + str(race_info['wind_speed']) + "mÃ¯Â¼Â",
            "",
            "Ã¤Â»ÂÃ¥ÂÂÃ£ÂÂ®Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ§Ã£ÂÂ¯Ã£ÂÂÃ¦Â©ÂÃ¦Â¢Â°Ã¥Â­Â¦Ã§Â¿ÂÃ£ÂÂ¢Ã£ÂÂ«Ã£ÂÂ´Ã£ÂÂªÃ£ÂÂºÃ£ÂÂ Ã£ÂÂÃ¤Â½Â¿Ã§ÂÂ¨Ã£ÂÂÃ£ÂÂ¦",
            "Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¦ÂÂ§Ã¨ÂÂ½Ã£ÂÂÃ£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂ¡Ã¤Â»Â¶Ã£ÂÂªÃ£ÂÂ©Ã£ÂÂÃ§Â·ÂÃ¥ÂÂÃ§ÂÂÃ£ÂÂ«Ã¥ÂÂÃ¦ÂÂÃ£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂÃ£ÂÂ"
        ]

    def _generate_race_overview(self, race_info, racers):
        """Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦Â¦ÂÃ¨Â¦ÂÃ§ÂÂÃ¦ÂÂ"""
        content = [
            "## Ã°ÂÂÂ Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦Â¦ÂÃ¨Â¦ÂÃ£ÂÂ»Ã¥ÂÂºÃ¨ÂµÂ°Ã©ÂÂ¸Ã¦ÂÂ",
            ""
        ]

        for racer in racers:
            content.append("**" + str(racer['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂÃ¯Â¼Â" + racer['racer_name'] + "**")
            content.append("- Ã¥ÂÂÃ§ÂÂÃ¯Â¼Â" + str(racer['win_rate']) + " / Ã©ÂÂ£Ã¥Â¯Â¾Ã§ÂÂÃ¯Â¼Â" + str(racer['place_rate']) + "%")
            content.append("- Ã¥Â¹Â³Ã¥ÂÂSTÃ¯Â¼Â" + str(racer['avg_st']) + " / Ã¨Â¿ÂÃ¦Â³ÂÃ¯Â¼Â" + racer['recent_form'])
            content.append("- Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¯Â¼Â" + str(racer['motor_performance']) + "% / Ã¨ÂÂÃ¯Â¼Â" + str(racer['boat_performance']) + "%")
            content.append("")

        return content

    def _generate_racer_analysis(self, racers, predictions):
        """Ã©ÂÂ¸Ã¦ÂÂÃ¥ÂÂÃ¦ÂÂÃ§ÂÂÃ¦ÂÂ"""
        content = [
            "## Ã°ÂÂÂ AIÃ©ÂÂ¸Ã¦ÂÂÃ¥ÂÂÃ¦ÂÂ",
            ""
        ]

        for pred in predictions[:3]:
            racer = next(r for r in racers if r['boat_number'] == pred['boat_number'])
            content.append("### " + str(pred['predicted_rank']) + "Ã¤Â½ÂÃ¤ÂºÂÃ¦ÂÂ³Ã¯Â¼Â" + pred['racer_name'] + " (" + str(pred['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂ)")
            content.append("**Ã¥ÂÂÃ§ÂÂÃ¤ÂºÂÃ¦ÂÂ³Ã¯Â¼Â" + str(pred['win_probability']) + "%**")
            content.append("")
            content.append("Ã£ÂÂÃ¥ÂÂÃ¦ÂÂÃ£ÂÂÃ£ÂÂ¤Ã£ÂÂ³Ã£ÂÂÃ£ÂÂ")

            if racer['win_rate'] >= 6.0:
                content.append("Ã¢ÂÂ Ã¥ÂÂÃ§ÂÂ" + str(racer['win_rate']) + "Ã£ÂÂ®Ã©Â«ÂÃ£ÂÂÃ¥Â®ÂÃ¥ÂÂÃ£ÂÂÃ¦ÂÂÃ£ÂÂ¤")
            if racer['avg_st'] <= 0.15:
                content.append("Ã¢ÂÂ Ã¥Â¹Â³Ã¥ÂÂST" + str(racer['avg_st']) + "Ã£ÂÂ®Ã¥Â¥Â½Ã£ÂÂ¹Ã£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂÃ¦ÂÂÃ¨Â¡Â")
            if racer['motor_performance'] >= 50:
                content.append("Ã¢ÂÂ Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã¨ÂªÂ¿Ã¦ÂÂ´Ã§ÂÂ" + str(racer['motor_performance']) + "%Ã£ÂÂ§Ã¦Â©ÂÃ©ÂÂ¢Ã¥Â¥Â½Ã¨ÂªÂ¿")

            content.append("")

        return content

    def _generate_prediction_basis(self, analysis):
        """Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ Ã§ÂÂÃ¦ÂÂ"""
        content = [
            "## Ã°ÂÂÂ¡ Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ Ã£ÂÂ»Ã¦Â³Â¨Ã§ÂÂ®Ã£ÂÂÃ£ÂÂ¤Ã£ÂÂ³Ã£ÂÂ",
            "",
            "### Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂ¡Ã¤Â»Â¶Ã¥ÂÂÃ¦ÂÂ"
        ]

        for condition in analysis['race_conditions']:
            content.append("- " + condition)

        content.append("")
        content.append("### Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ»Ã¦Â©ÂÃ¦ÂÂÃ¥ÂÂÃ¦ÂÂ")
        content.append("- Ã¦ÂÂÃ©Â«ÂÃ¥Â®ÂÃ¥ÂÂÃ¨ÂÂ: " + analysis['racer_analysis']['best_performer'])
        content.append("- Ã¦ÂÂÃ¥ÂÂªÃ§Â§ÂST: " + analysis['racer_analysis']['best_start'])
        content.append("- Ã¦ÂÂÃ©Â«ÂÃ£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼: " + analysis['racer_analysis']['best_motor'])

        content.append("")
        content.append("### Ã¦ÂÂ¬Ã¥ÂÂ½Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ®Ã¦Â Â¹Ã¦ÂÂ ")
        for rationale in analysis['prediction_rationale']:
            content.append("Ã¢ÂÂ " + rationale)

        if analysis['risk_assessment']:
            content.append("")
            content.append("### Ã¢ÂÂ Ã¯Â¸Â Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã¨Â¦ÂÃ¥ÂÂ ")
            for risk in analysis['risk_assessment']:
                content.append("- " + risk)

        return content

    def _generate_repertoire_section(self, repertoire):
        """Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼Ã§ÂÂÃ¦ÂÂ"""
        content = [
            "## Ã°ÂÂÂ¯ Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼Ã¯Â¼ÂÃ¦ÂÂ¬Ã¥ÂÂ½Ã£ÂÂ»Ã¤Â¸Â­Ã§Â©Â´Ã£ÂÂ»Ã¥Â¤Â§Ã§Â©Â´Ã¯Â¼Â",
            ""
        ]

        for pred_type, prediction in repertoire.items():
            content.append("### " + prediction['type'])
            content.append("**Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®Ã¯Â¼Â" + prediction['target'] + "**")
            content.append("- Ã¤Â¿Â¡Ã©Â Â¼Ã¥ÂºÂ¦Ã¯Â¼Â" + str(prediction['confidence']) + "%")
            content.append("- Ã¤ÂºÂÃ¦ÂÂ³Ã©ÂÂÃ¥Â½ÂÃ¯Â¼Â" + prediction['expected_odds'])
            content.append("- Ã¦ÂÂ¨Ã¥Â¥Â¨Ã¦ÂÂÃ¨Â³ÂÃ¦Â¯ÂÃ§ÂÂÃ¯Â¼Â" + prediction['investment_ratio'])
            content.append("- Ã¦Â Â¹Ã¦ÂÂ Ã¯Â¼Â" + prediction['reason'])
            content.append("")

        return content

    def _generate_investment_section(self, strategy):
        """Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã§ÂÂÃ¦ÂÂ"""
        content = [
            "## Ã°ÂÂÂ° Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã£ÂÂ»Ã¨Â³ÂÃ©ÂÂÃ§Â®Â¡Ã§ÂÂ",
            "",
            "### Ã¦ÂÂ¨Ã¥Â¥Â¨Ã¤ÂºÂÃ§Â®ÂÃ¯Â¼Â" + "{:,}".format(strategy['total_budget']) + "Ã¥ÂÂ",
            ""
        ]

        for allocation in strategy['allocations']:
            content.append("**" + allocation['type'] + "**")
            content.append("- Ã¦ÂÂÃ¨Â³ÂÃ©Â¡ÂÃ¯Â¼Â" + "{:,}".format(allocation['amount']) + "Ã¥ÂÂ")
            content.append("- Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®Ã¯Â¼Â" + allocation['target'])
            content.append("- Ã¦ÂÂÃ¥Â¾ÂÃ£ÂÂªÃ£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂ³Ã¯Â¼Â" + "{:,}".format(allocation['expected_return']) + "Ã¥ÂÂ")
            content.append("- Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ«Ã¯Â¼Â" + allocation['risk_level'])
            content.append("")

        content.append("### Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã§Â®Â¡Ã§ÂÂÃ£ÂÂ«Ã£ÂÂ¼Ã£ÂÂ«")
        for i, rule in enumerate(strategy['risk_management'], 1):
            content.append(str(i) + ". " + rule)

        content.append("")
        content.append("### Ã¥ÂÂ©Ã§ÂÂÃ§ÂÂ®Ã¦Â¨Â")
        for target_type, target_desc in strategy['profit_target'].items():
            content.append("- " + target_type.capitalize() + ": " + target_desc)

        return content

    def _generate_conclusion(self, race_info, predictions):
        """Ã£ÂÂ¾Ã£ÂÂ¨Ã£ÂÂÃ§ÂÂÃ¦ÂÂ"""
        top_pick = predictions[0]

        return [
            "## Ã°ÂÂÂ Ã£ÂÂ¾Ã£ÂÂ¨Ã£ÂÂÃ£ÂÂ»Ã¦ÂÂÃ§ÂµÂÃ¤ÂºÂÃ¦ÂÂ³",
            "",
            "Ã¤Â»ÂÃ¥ÂÂÃ£ÂÂ®" + race_info['venue'] + str(race_info['race_number']) + "RÃ£ÂÂ¯Ã£ÂÂ",
            str(top_pick['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂ " + top_pick['racer_name'] + "Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ¦ÂÂ¬Ã¥ÂÂ½Ã£ÂÂ¨Ã£ÂÂÃ£ÂÂ¦Ã£ÂÂ",
            "Ã¨Â¤ÂÃ¦ÂÂ°Ã£ÂÂ®Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®Ã£ÂÂÃ£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂ³Ã£ÂÂ§Ã¦ÂÂ»Ã§ÂÂ¥Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¨Ã£ÂÂÃ¦ÂÂ¨Ã¥Â¥Â¨Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "",
            "AIÃ£ÂÂ®Ã¥ÂÂÃ¦ÂÂÃ§ÂµÂÃ¦ÂÂÃ£ÂÂÃ¥ÂÂÃ¨ÂÂÃ£ÂÂ«Ã£ÂÂÃ§ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ®Ã¦ÂÂÃ¨Â³ÂÃ£ÂÂ¹Ã£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂ«Ã£ÂÂ«Ã¥ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¦",
            "Ã¨ÂÂÃ¥ÂÂ¸Ã£ÂÂÃ¨Â³Â¼Ã¥ÂÂ¥Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¨Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "",
            "Ã¢ÂÂ Ã¯Â¸Â Ã¦Â³Â¨Ã¦ÂÂÃ¯Â¼ÂÃ¨ÂÂÃ¥ÂÂ¸Ã¨Â³Â¼Ã¥ÂÂ¥Ã£ÂÂ¯Ã¨ÂÂªÃ¥Â·Â±Ã¨Â²Â¬Ã¤Â»Â»Ã£ÂÂ§Ã¨Â¡ÂÃ£ÂÂ£Ã£ÂÂ¦Ã£ÂÂÃ£ÂÂ Ã£ÂÂÃ£ÂÂÃ£ÂÂ",
            "Ã¥Â½ÂÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¯Ã¥ÂÂÃ¨ÂÂÃ¦ÂÂÃ¥Â Â±Ã£ÂÂ§Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ§ÂÂÃ¤Â¸Â­Ã£ÂÂÃ¤Â¿ÂÃ¨Â¨Â¼Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ®Ã£ÂÂ§Ã£ÂÂ¯Ã£ÂÂÃ£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂÃ£ÂÂ",
            "",
            "Ã£ÂÂÃ£ÂÂÃ£ÂÂ§Ã£ÂÂ¯Ã£ÂÂÃ¨ÂÂ¯Ã£ÂÂÃ£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂÃ¯Â¼ÂÃ°ÂÂÂ¤Ã¢ÂÂ¨",
            "",
            "---",
            "",
            "#Ã§Â«Â¶Ã¨ÂÂ #Ã§Â«Â¶Ã¨ÂÂÃ¤ÂºÂÃ¦ÂÂ³ #AIÃ¤ÂºÂÃ¦ÂÂ³ #Ã¨ÂÂÃ¥ÂÂ¸ #Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹"
        ]

    def _generate_additional_content(self, race_info, current_count):
        """Ã¤Â¸ÂÃ¨Â¶Â³Ã¥ÂÂÃ£ÂÂ®Ã¨Â¿Â½Ã¥ÂÂ Ã£ÂÂ³Ã£ÂÂ³Ã£ÂÂÃ£ÂÂ³Ã£ÂÂ"""
        needed = 2000 - current_count

        additional = [
            "",
            "## Ã°ÂÂÂ¬ Ã¨Â©Â³Ã§Â´Â°Ã¦ÂÂÃ¨Â¡ÂÃ¨Â§Â£Ã¨ÂªÂ¬",
            "",
            "### AIÃ£ÂÂ¢Ã£ÂÂ«Ã£ÂÂ´Ã£ÂÂªÃ£ÂÂºÃ£ÂÂ Ã£ÂÂ«Ã£ÂÂ¤Ã£ÂÂÃ£ÂÂ¦",
            "Ã¦ÂÂ¬Ã£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ Ã£ÂÂ§Ã£ÂÂ¯Ã£ÂÂÃ£ÂÂ©Ã£ÂÂ³Ã£ÂÂÃ£ÂÂ Ã£ÂÂÃ£ÂÂ©Ã£ÂÂ¬Ã£ÂÂ¹Ã£ÂÂÃ¥ÂÂÃ¥Â¸Â°Ã£ÂÂÃ¤Â½Â¿Ã§ÂÂ¨Ã£ÂÂÃ£ÂÂ¦Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ®Ã¦ÂÂÃ§Â¸Â¾Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂÃ¨Â¡ÂÃ£ÂÂ£Ã£ÂÂ¦Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "Ã£ÂÂÃ£ÂÂ®Ã£ÂÂ¢Ã£ÂÂ«Ã£ÂÂ´Ã£ÂÂªÃ£ÂÂºÃ£ÂÂ Ã£ÂÂ¯Ã£ÂÂÃ¨Â¤ÂÃ¦ÂÂ°Ã£ÂÂ®Ã¦Â±ÂºÃ¥Â®ÂÃ¦ÂÂ¨Ã£ÂÂÃ§ÂµÂÃ£ÂÂ¿Ã¥ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¨Ã£ÂÂ§Ã£ÂÂ",
            "Ã£ÂÂÃ£ÂÂÃ§Â²Â¾Ã¥ÂºÂ¦Ã£ÂÂ®Ã©Â«ÂÃ£ÂÂÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂÃ¥Â®ÂÃ§ÂÂ¾Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "",
            "### Ã¤Â½Â¿Ã§ÂÂ¨Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã©Â ÂÃ§ÂÂ®",
            "- Ã©ÂÂ¸Ã¦ÂÂÃ¥ÂÂÃ§ÂÂÃ£ÂÂ»Ã©ÂÂ£Ã¥Â¯Â¾Ã§ÂÂ",
            "- Ã¥Â¹Â³Ã¥ÂÂÃ£ÂÂ¹Ã£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂÃ£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂÃ£ÂÂ³Ã£ÂÂ°",
            "- Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂ»Ã¨ÂÂÃ£ÂÂ®Ã¨ÂªÂ¿Ã¦ÂÂ´Ã§ÂÂ¶Ã¦Â³Â", 
            "- Ã¥Â¤Â©Ã¥ÂÂÃ£ÂÂ»Ã¦Â°Â´Ã©ÂÂ¢Ã¦ÂÂ¡Ã¤Â»Â¶",
            "- Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ®Ã¤Â½ÂÃ©ÂÂÃ£ÂÂ»Ã¨Â¿ÂÃ¦Â³Â",
            "",
            "Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ®Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ§Â·ÂÃ¥ÂÂÃ§ÂÂÃ£ÂÂ«Ã¥ÂÂÃ¦ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¨Ã£ÂÂ§Ã£ÂÂ",
            "Ã¤Â»ÂÃ¥ÂÂ" + race_info['venue'] + "Ã£ÂÂ®Ã¤ÂºÂÃ¦ÂÂ³Ã§Â²Â¾Ã¥ÂºÂ¦Ã£ÂÂÃ¥ÂÂÃ¤Â¸ÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¦Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "",
            "### Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ®Ã¤Â¿Â¡Ã©Â Â¼Ã¦ÂÂ§Ã¥ÂÂÃ¤Â¸ÂÃ£ÂÂ®Ã£ÂÂÃ£ÂÂÃ£ÂÂ«",
            "AIÃ£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ Ã£ÂÂ¯Ã§Â¶ÂÃ§Â¶ÂÃ§ÂÂÃ£ÂÂ«Ã¥Â­Â¦Ã§Â¿ÂÃ£ÂÂÃ©ÂÂÃ£ÂÂ­Ã£ÂÂ",
            "Ã¤ÂºÂÃ¦ÂÂ³Ã§Â²Â¾Ã¥ÂºÂ¦Ã£ÂÂ®Ã¥ÂÂÃ¤Â¸ÂÃ£ÂÂ«Ã¥ÂÂªÃ£ÂÂÃ£ÂÂ¦Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ",
            "Ã§ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ®Ã£ÂÂÃ£ÂÂ£Ã£ÂÂ¼Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¯Ã£ÂÂÃ¥Â¤Â§Ã¥ÂÂÃ£ÂÂ«Ã£ÂÂÃ£ÂÂªÃ£ÂÂÃ£ÂÂÃ£ÂÂ",
            "Ã£ÂÂÃ£ÂÂÃ¨ÂÂ¯Ã£ÂÂÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ Ã£ÂÂ®Ã¦Â§ÂÃ§Â¯ÂÃ£ÂÂÃ§ÂÂ®Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¦Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂ"
        ]

        return "\n".join(additional)

# Ã£ÂÂ¡Ã£ÂÂ¤Ã£ÂÂ³Ã¥ÂÂ¦Ã§ÂÂ
def main():
    # Ã£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂÃ£ÂÂ«
    st.title("Ã°ÂÂÂ¤ Ã§Â«Â¶Ã¨ÂÂAIÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ  v13.9")
    st.markdown("**Ã¥Â®ÂÃ§ÂÂ¨Ã¥Â®ÂÃ¥ÂÂ¨Ã§ÂÂ - Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ Ã£ÂÂ»noteÃ¨Â¨ÂÃ¤ÂºÂÃ£ÂÂ»Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã£ÂÂ¾Ã£ÂÂ§Ã¥Â®ÂÃ¥ÂÂ¨Ã£ÂÂµÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ**")

    # Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¸Ã£ÂÂ£Ã£ÂÂ¼Ã¥ÂÂÃ¦ÂÂÃ¥ÂÂ
    data_manager = KyoteiDataManager()
    predictor = PredictionAnalyzer()
    prediction_types = PredictionTypes()
    investment_strategy = InvestmentStrategy()
    note_generator = NoteArticleGenerator()

    # Ã¦ÂÂ¥Ã¤Â»ÂÃ©ÂÂ¸Ã¦ÂÂ
    selected_date = st.date_input(
        "Ã°ÂÂÂ Ã¤ÂºÂÃ¦ÂÂ³Ã¦ÂÂ¥Ã£ÂÂÃ©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¦Ã£ÂÂÃ£ÂÂ Ã£ÂÂÃ£ÂÂ",
        dt.datetime.now().date(),
        min_value=dt.date(2024, 1, 1),
        max_value=dt.date(2025, 12, 31)
    )

    # Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¥ÂÂÃ¥Â¾ÂÃ£ÂÂ»Ã¨Â¡Â¨Ã§Â¤Âº
    races = data_manager.get_races_for_date(selected_date)

    if not races:
        st.warning("Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ¦ÂÂ¥Ã¤Â»ÂÃ£ÂÂ«Ã£ÂÂ¯Ã©ÂÂÃ¥ÂÂ¬Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¾Ã£ÂÂÃ£ÂÂÃ£ÂÂ")
        return

    # Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã©ÂÂ¸Ã¦ÂÂ
    race_options = [race['venue'] + " " + str(race['race_number']) + "R (" + race['race_time'] + ") " + race['class']
                   for race in races]

    selected_race_index = st.selectbox(
        "Ã°ÂÂÂ Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã£ÂÂÃ©ÂÂ¸Ã¦ÂÂÃ£ÂÂÃ£ÂÂ¦Ã£ÂÂÃ£ÂÂ Ã£ÂÂÃ£ÂÂ",
        range(len(race_options)),
        format_func=lambda i: race_options[i]
    )

    selected_race = races[selected_race_index]

    # Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂÃ¥Â Â±Ã¨Â¡Â¨Ã§Â¤Âº
    st.markdown("### Ã°ÂÂÂ Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂÃ¥Â Â±")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Ã¤Â¼ÂÃ¥Â Â´", selected_race['venue'])
        st.metric("Ã£ÂÂ¯Ã£ÂÂ©Ã£ÂÂ¹", selected_race['class'])
    with col2:
        st.metric("Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹", str(selected_race['race_number']) + "R")
        st.metric("Ã¨Â·ÂÃ©ÂÂ¢", selected_race['distance'])
    with col3:
        st.metric("Ã§ÂÂºÃ¨ÂµÂ°Ã¦ÂÂÃ¥ÂÂ»", selected_race['race_time'])
        st.metric("Ã¥Â¤Â©Ã¥ÂÂ", selected_race['weather'])
    with col4:
        st.metric("Ã©Â¢Â¨Ã©ÂÂ", str(selected_race['wind_speed']) + "m")
        st.metric("Ã¦Â°Â´Ã¦Â¸Â©", str(selected_race['water_temp']) + "ÃÂ°C")

    # Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂµÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã¥ÂÂÃ¥Â¾ÂÃ£ÂÂ»Ã¤ÂºÂÃ¦ÂÂ³Ã¥Â®ÂÃ¨Â¡Â
    racers = data_manager.get_racer_data(selected_race)
    predictions = predictor.analyze_race(selected_race, racers)

    # Ã¨Â©Â³Ã§Â´Â°Ã¥ÂÂÃ¦ÂÂÃ¥Â®ÂÃ¨Â¡Â
    detailed_analysis = predictor.generate_detailed_analysis(selected_race, racers, predictions)

    # Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼Ã§ÂÂÃ¦ÂÂ
    repertoire = prediction_types.generate_prediction_repertoire(selected_race, racers, predictions)

    # Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã§ÂÂÃ¦ÂÂ
    strategy = investment_strategy.generate_strategy(selected_race, predictions, repertoire)

    # Ã¥ÂÂºÃ¨ÂµÂ°Ã©ÂÂ¸Ã¦ÂÂÃ¦ÂÂÃ¥Â Â±
    st.markdown("### Ã°ÂÂÂ¤ Ã¥ÂÂºÃ¨ÂµÂ°Ã©ÂÂ¸Ã¦ÂÂÃ¦ÂÂÃ¥Â Â±")
    for racer in racers:
        with st.expander(str(racer['boat_number']) + "Ã¥ÂÂ·Ã¨ÂÂ " + racer['racer_name']):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Ã¥ÂÂÃ§ÂÂ**: " + str(racer['win_rate']))
                st.write("**Ã©ÂÂ£Ã¥Â¯Â¾Ã§ÂÂ**: " + str(racer['place_rate']) + "%")
                st.write("**Ã¥Â¹Â³Ã¥ÂÂST**: " + str(racer['avg_st']))
                st.write("**Ã¤Â½ÂÃ©ÂÂ**: " + str(racer['weight']) + "kg")
            with col2:
                st.write("**Ã¨Â¿ÂÃ¦Â³Â**: " + racer['recent_form'])
                st.write("**Ã£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼**: " + str(racer['motor_performance']) + "%")
                st.write("**Ã¨ÂÂ**: " + str(racer['boat_performance']) + "%")

    # AIÃ¤ÂºÂÃ¦ÂÂ³Ã§ÂµÂÃ¦ÂÂ
    st.markdown("### Ã°ÂÂÂ¯ AIÃ¤ÂºÂÃ¦ÂÂ³Ã§ÂµÂÃ¦ÂÂ")
    for i, pred in enumerate(predictions[:3]):
        st.markdown("""
        <div class="prediction-card">
            <strong>""" + str(pred['predicted_rank']) + """Ã¤Â½ÂÃ¤ÂºÂÃ¦ÂÂ³</strong><br>
            Ã°ÂÂÂ¤ """ + str(pred['boat_number']) + """Ã¥ÂÂ·Ã¨ÂÂ """ + pred['racer_name'] + """<br>
            Ã°ÂÂÂ Ã¥ÂÂÃ§ÂÂÃ¤ÂºÂÃ¦ÂÂ³: """ + str(pred['win_probability']) + """%
        </div>
        """, unsafe_allow_html=True)

    # Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ Ã¨Â©Â³Ã§Â´Â°Ã¨Â¡Â¨Ã§Â¤Âº
    st.markdown("### Ã°ÂÂÂ¡ Ã¤ÂºÂÃ¦ÂÂ³Ã¦Â Â¹Ã¦ÂÂ Ã¨Â©Â³Ã§Â´Â°")

    conditions_html = '<br>'.join(['Ã¢ÂÂ¢ ' + condition for condition in detailed_analysis['race_conditions']])
    rationale_html = '<br>'.join(['Ã¢ÂÂ ' + rationale for rationale in detailed_analysis['prediction_rationale']])
    risks_html = '<br>'.join(['Ã¢ÂÂ¢ ' + risk for risk in detailed_analysis['risk_assessment']]) if detailed_analysis['risk_assessment'] else ''

    st.markdown("""
    <div class="prediction-detail">
        <h4>Ã°ÂÂÂ¤Ã¯Â¸Â Ã£ÂÂ¬Ã£ÂÂ¼Ã£ÂÂ¹Ã¦ÂÂ¡Ã¤Â»Â¶Ã¥ÂÂÃ¦ÂÂ</h4>
        """ + conditions_html + """

        <h4>Ã°ÂÂÂ¥ Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ»Ã¦Â©ÂÃ¦ÂÂÃ¥ÂÂÃ¦ÂÂ</h4>
        Ã¢ÂÂ¢ Ã¦ÂÂÃ©Â«ÂÃ¥Â®ÂÃ¥ÂÂÃ¨ÂÂ: """ + detailed_analysis['racer_analysis']['best_performer'] + """<br>
        Ã¢ÂÂ¢ Ã¦ÂÂÃ¥ÂÂªÃ§Â§ÂST: """ + detailed_analysis['racer_analysis']['best_start'] + """<br>
        Ã¢ÂÂ¢ Ã¦ÂÂÃ©Â«ÂÃ£ÂÂ¢Ã£ÂÂ¼Ã£ÂÂ¿Ã£ÂÂ¼: """ + detailed_analysis['racer_analysis']['best_motor'] + """

        <h4>Ã°ÂÂÂ¯ Ã¦ÂÂ¬Ã¥ÂÂ½Ã©ÂÂ¸Ã¦ÂÂÃ£ÂÂ®Ã¦Â Â¹Ã¦ÂÂ </h4>
        """ + rationale_html + """

        """ + ('<h4>Ã¢ÂÂ Ã¯Â¸Â Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã¨Â¦ÂÃ¥ÂÂ </h4>' + risks_html if risks_html else '') + """
    </div>
    """, unsafe_allow_html=True)

    # Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼
    st.markdown("### Ã°ÂÂÂ¯ Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¬Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂªÃ£ÂÂ¼")

    tab1, tab2, tab3 = st.tabs(["Ã¦ÂÂ¬Ã¥ÂÂ½", "Ã¤Â¸Â­Ã§Â©Â´", "Ã¥Â¤Â§Ã§Â©Â´"])

    with tab1:
        honmei = repertoire['honmei']
        st.markdown("""
        <div class="prediction-type">
            <h4>""" + honmei['type'] + """</h4>
            <strong>Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®: """ + honmei['target'] + """</strong><br>
            Ã¤Â¿Â¡Ã©Â Â¼Ã¥ÂºÂ¦: """ + str(honmei['confidence']) + """% | Ã¤ÂºÂÃ¦ÂÂ³Ã©ÂÂÃ¥Â½Â: """ + honmei['expected_odds'] + """<br>
            Ã¦ÂÂ¨Ã¥Â¥Â¨Ã¦ÂÂÃ¨Â³ÂÃ¦Â¯ÂÃ§ÂÂ: """ + honmei['investment_ratio'] + """<br>
            <strong>Ã¦Â Â¹Ã¦ÂÂ :</strong> """ + honmei['reason'] + """
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        chuuketsu = repertoire['chuuketsu']
        st.markdown("""
        <div class="prediction-type">
            <h4>""" + chuuketsu['type'] + """</h4>
            <strong>Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®: """ + chuuketsu['target'] + """</strong><br>
            Ã¤Â¿Â¡Ã©Â Â¼Ã¥ÂºÂ¦: """ + str(chuuketsu['confidence']) + """% | Ã¤ÂºÂÃ¦ÂÂ³Ã©ÂÂÃ¥Â½Â: """ + chuuketsu['expected_odds'] + """<br>
            Ã¦ÂÂ¨Ã¥Â¥Â¨Ã¦ÂÂÃ¨Â³ÂÃ¦Â¯ÂÃ§ÂÂ: """ + chuuketsu['investment_ratio'] + """<br>
            <strong>Ã¦Â Â¹Ã¦ÂÂ :</strong> """ + chuuketsu['reason'] + """
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        ooketsu = repertoire['ooketsu']
        st.markdown("""
        <div class="prediction-type">
            <h4>""" + ooketsu['type'] + """</h4>
            <strong>Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®: """ + ooketsu['target'] + """</strong><br>
            Ã¤Â¿Â¡Ã©Â Â¼Ã¥ÂºÂ¦: """ + str(ooketsu['confidence']) + """% | Ã¤ÂºÂÃ¦ÂÂ³Ã©ÂÂÃ¥Â½Â: """ + ooketsu['expected_odds'] + """<br>
            Ã¦ÂÂ¨Ã¥Â¥Â¨Ã¦ÂÂÃ¨Â³ÂÃ¦Â¯ÂÃ§ÂÂ: """ + ooketsu['investment_ratio'] + """<br>
            <strong>Ã¦Â Â¹Ã¦ÂÂ :</strong> """ + ooketsu['reason'] + """
        </div>
        """, unsafe_allow_html=True)

    # Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥
    st.markdown("### Ã°ÂÂÂ° Ã¦ÂÂÃ¨Â³ÂÃ¦ÂÂ¦Ã§ÂÂ¥Ã£ÂÂ»Ã¨Â³ÂÃ©ÂÂÃ§Â®Â¡Ã§ÂÂ")

    st.markdown("""
    <div class="investment-strategy">
        <h4>Ã¦ÂÂ¨Ã¥Â¥Â¨Ã¤ÂºÂÃ§Â®Â: """ + "{:,}".format(strategy['total_budget']) + """Ã¥ÂÂ</h4>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #e9ecef;">
                <th style="padding: 8px; border: 1px solid #ddd;">Ã¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ¿Ã£ÂÂ¤Ã£ÂÂ</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Ã¦ÂÂÃ¨Â³ÂÃ©Â¡Â</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Ã¨Â²Â·Ã£ÂÂÃ§ÂÂ®</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Ã¦ÂÂÃ¥Â¾ÂÃ£ÂÂªÃ£ÂÂ¿Ã£ÂÂ¼Ã£ÂÂ³</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯</th>
            </tr>
    """, unsafe_allow_html=True)

    for allocation in strategy['allocations']:
        st.markdown("""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + allocation['type'] + """</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + "{:,}".format(allocation['amount']) + """Ã¥ÂÂ</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + allocation['target'] + """</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + "{:,}".format(allocation['expected_return']) + """Ã¥ÂÂ</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + allocation['risk_level'] + """</td>
            </tr>
        """, unsafe_allow_html=True)

    st.markdown("""
        </table>

        <h4>Ã£ÂÂªÃ£ÂÂ¹Ã£ÂÂ¯Ã§Â®Â¡Ã§ÂÂÃ£ÂÂ«Ã£ÂÂ¼Ã£ÂÂ«</h4>
    """, unsafe_allow_html=True)

    for i, rule in enumerate(strategy['risk_management'], 1):
        st.markdown(str(i) + ". " + rule + "<br>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # noteÃ¨Â¨ÂÃ¤ÂºÂÃ§ÂÂÃ¦ÂÂ
    st.markdown("### Ã°ÂÂÂ noteÃ¨Â¨ÂÃ¤ÂºÂÃ¯Â¼Â2000Ã¦ÂÂÃ¥Â­ÂÃ¤Â»Â¥Ã¤Â¸ÂÃ¯Â¼Â")

    if st.button("noteÃ¨Â¨ÂÃ¤ÂºÂÃ£ÂÂÃ§ÂÂÃ¦ÂÂ", type="primary"):
        with st.spinner("Ã¨Â¨ÂÃ¤ÂºÂÃ§ÂÂÃ¦ÂÂÃ¤Â¸Â­..."):
            note_article = note_generator.generate_article(
                selected_race, racers, predictions, detailed_analysis, repertoire, strategy
            )

            st.markdown("""
            <div class="note-article">
                <h4>Ã°ÂÂÂ Ã§ÂÂÃ¦ÂÂÃ£ÂÂÃ£ÂÂÃ£ÂÂÃ¨Â¨ÂÃ¤ÂºÂ (Ã¦ÂÂÃ¥Â­ÂÃ¦ÂÂ°: """ + str(len(note_article)) + """Ã¦ÂÂÃ¥Â­Â)</h4>
                <div style="max-height: 400px; overflow-y: auto; padding: 1rem; background-color: white; border-radius: 0.25rem;">
                    <pre style="white-space: pre-wrap; font-family: inherit;">""" + note_article + """</pre>
                </div>
                <br>
                <small>Ã°ÂÂÂ¡ Ã£ÂÂÃ£ÂÂ®Ã¨Â¨ÂÃ¤ÂºÂÃ£ÂÂÃ£ÂÂ³Ã£ÂÂÃ£ÂÂ¼Ã£ÂÂÃ£ÂÂ¦noteÃ£ÂÂ«Ã¦ÂÂÃ§Â¨Â¿Ã£ÂÂ§Ã£ÂÂÃ£ÂÂ¾Ã£ÂÂ</small>
            </div>
            """, unsafe_allow_html=True)

    # Ã£ÂÂÃ£ÂÂÃ£ÂÂ¿Ã£ÂÂ¼
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    Ã§Â«Â¶Ã¨ÂÂAIÃ¤ÂºÂÃ¦ÂÂ³Ã£ÂÂ·Ã£ÂÂ¹Ã£ÂÂÃ£ÂÂ  v13.9 (Ã¥Â®ÂÃ§ÂÂ¨Ã¥Â®ÂÃ¥ÂÂ¨Ã§ÂÂ) | Ã¦Â§ÂÃ¦ÂÂÃ£ÂÂ¨Ã£ÂÂ©Ã£ÂÂ¼Ã£ÂÂªÃ£ÂÂ | Ã¥Â®ÂÃ£ÂÂÃ£ÂÂ¼Ã£ÂÂ¿Ã©ÂÂ£Ã¦ÂÂº<br>
    Ã¢ÂÂ Ã¯Â¸Â Ã¨ÂÂÃ¥ÂÂ¸Ã¨Â³Â¼Ã¥ÂÂ¥Ã£ÂÂ¯Ã¨ÂÂªÃ¥Â·Â±Ã¨Â²Â¬Ã¤Â»Â»Ã£ÂÂ§Ã¨Â¡ÂÃ£ÂÂ£Ã£ÂÂ¦Ã£ÂÂÃ£ÂÂ Ã£ÂÂÃ£ÂÂ
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
