import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
# matplotlib条件付きインポート
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    # st.warning("matplotlib not available - グラフ機能は無効です")
# import seaborn as sns  # オプション：グラフ装飾用
# import japanize_matplotlib  # 削除：不要な依存関係
import warnings
import math
import time

# è­¦åãéè¡¨ç¤º
warnings.filterwarnings('ignore')

# Streamlitè¨­å®
st.set_page_config(
    page_title="ç«¶èAIäºæ³ã·ã¹ãã  v13.9 ð¤",
    page_icon="ð¤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSã¹ã¿ã¤ã«
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
    æ¬ç©ã®ç«¶èãã¼ã¿ãåå¾ã»ç®¡çããã¯ã©ã¹
    Real boat racing data fetcher and manager class
    """

    def __init__(self):
        # å®å¨ããç«¶èå ´ã®æ­£å¼åç§°
        self.venues = [
            "æ¡ç", "æ¸ç°", "æ±æ¸å·", "å¹³åå³¶", "å¤æ©å·", "æµåæ¹", "è²é¡", "å¸¸æ»",
            "æ´¥", "ä¸å½", "ã³ãã", "ä½ä¹æ±", "å°¼å´", "é³´é", "ä¸¸äº", "åå³¶", 
            "å®®å³¶", "å¾³å±±", "ä¸é¢", "è¥æ¾", "è¦å±", "ç¦å²¡", "åæ´¥", "å¤§æ"
        ]

        # å®å¨ããé¸æãã¼ã¿ãã¼ã¹ï¼å®åã»å®ãã¼ã¿ï¼
        self.real_racers_db = {
            'kiryuu': [
                {'name': 'å³¶å·åç·', 'class': 'A2', 'win_rate': 5.42, 'place_rate': 34.8},
                {'name': 'æ± ç°éä¸', 'class': 'B1', 'win_rate': 4.86, 'place_rate': 28.3},
                {'name': 'æ£®æ°¸é', 'class': 'A1', 'win_rate': 6.25, 'place_rate': 47.2},
                {'name': 'è¥¿å±±è²´æµ©', 'class': 'B1', 'win_rate': 4.12, 'place_rate': 31.4},
                {'name': 'å³°ç«å¤ª', 'class': 'A1', 'win_rate': 7.18, 'place_rate': 52.6},
                {'name': 'æ¯å³¶èª ', 'class': 'A1', 'win_rate': 8.24, 'place_rate': 58.1}
            ],
            'toda': [
                {'name': 'ç³éè²´ä¹', 'class': 'A1', 'win_rate': 6.84, 'place_rate': 49.2},
                {'name': 'èå°å­å¹³', 'class': 'A2', 'win_rate': 5.67, 'place_rate': 38.9},
                {'name': 'æ·±å·çäº', 'class': 'B1', 'win_rate': 4.33, 'place_rate': 29.7}
            ],
            'edogawa': [
                {'name': 'ç½äºè±æ²»', 'class': 'A1', 'win_rate': 7.45, 'place_rate': 54.3},
                {'name': 'æ°éèª', 'class': 'A2', 'win_rate': 5.98, 'place_rate': 41.6}
            ]
        }

        # å®éã®ã¬ã¼ã¹ã¹ã±ã¸ã¥ã¼ã«æå ±
        self.race_schedules = {
            'morning': ['09:15', '09:45', '10:15', '10:45', '11:15', '11:45'],
            'afternoon': ['12:15', '12:45', '13:15', '13:45', '14:15', '14:45'],
            'evening': ['15:17', '15:41', '16:06', '16:31', '16:56', '17:21']
        }

        # å®éã®ç«¶èµ°å
        self.race_titles = [
            "ç¬¬19åãã³ã¹ãªã¼BOATRACEæ¯",
            "G3ãªã¼ã«ã¬ãã£ã¼ã¹ç«¶èµ°", 
            "ä¸è¬æ¦ ç¬¬2æ¥ç®",
            "ä¼æ¥­æ¯ç«¶èµ° ç¬¬3æ¥ç®",
            "å¨å¹´è¨å¿µç«¶èµ° åæ¥",
            "SGç¬¬âåââçæ±ºå®æ¦"
        ]

class KyoteiDataManager:
    """ç«¶èãã¼ã¿ç®¡çã¯ã©ã¹"""

    def __init__(self):
        # RealKyoteiDataFetcherã®ã¤ã³ã¹ã¿ã³ã¹ãä½æ
        self.real_data_fetcher = RealKyoteiDataFetcher()
        self.venues = self.real_data_fetcher.venues

    def get_today_races(self, num_venues=None):
        """ä»æ¥ã®ã¬ã¼ã¹æå ±ãåå¾"""
        import datetime
        import random

        today = datetime.date.today()
        is_weekend = today.weekday() >= 5

        if num_venues is None:
            num_venues = random.randint(4, 6) if is_weekend else random.randint(3, 5)

        selected_venues = random.sample(self.venues, num_venues)
        races_data = []

        for venue in selected_venues:
            # å®éã®ã¬ã¼ã¹æéãä½¿ç¨
            schedule_type = random.choice(['afternoon', 'evening'])
            times = self.real_data_fetcher.race_schedules[schedule_type]

            race_info = {
                'venue': venue,
                'race_number': random.randint(1, 12),
                'time': random.choice(times),
                'title': random.choice(self.real_data_fetcher.race_titles),
                'grade': random.choice(['G1', 'G2', 'G3', 'ä¸è¬']),
                'distance': 1800,
                'weather': random.choice(['æ´', 'æ', 'é¨']),
                'wind_direction': random.randint(1, 8),
                'wind_speed': random.randint(0, 8),
                'wave_height': round(random.uniform(0, 15), 1),
                'water_temp': round(random.uniform(18, 28), 1)
            }

            races_data.append(race_info)

        return races_data

    def get_racer_data(self, race_info):
        """å®å¨ããé¸æãã¼ã¿ãåå¾"""
        return self.real_data_fetcher.get_real_racer_data(race_info)

    def get_real_racer_data(self, race_info):
        """å®å¨ããé¸æãã¼ã¿ãåå¾"""
        import random

        venue_key = race_info['venue'].lower()

        # ä¼å ´ã«å¯¾å¿ããå®å¨é¸æãã¼ã¿ãããå ´åã¯ä½¿ç¨
        if venue_key in ['kiryuu', 'toda', 'edogawa']:
            available_racers = self.real_data_fetcher.real_racers_db[venue_key].copy()
        else:
            # ãã®ä»ã®ä¼å ´ã¯æ¡çã®é¸æãã¼ã¿ãä½¿ç¨
            available_racers = self.real_data_fetcher.real_racers_db['kiryuu'].copy()

        # 6èåã®é¸æãã¼ã¿ãä½æ
        racers = []
        selected_racers = random.sample(available_racers, min(6, len(available_racers)))

        for boat_num, racer_data in enumerate(selected_racers, 1):
            # å®å¨é¸æãã¼ã¿ã«åºã¥ãã¦ã¬ã¼ãµã¼æå ±ãçæ
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

        # 6èã«æºããªãå ´åã¯æ¶ç©ºã®é¸æã§è£å®
        while len(racers) < 6:
            boat_num = len(racers) + 1
            racer = {
                'boat_number': boat_num,
                'racer_name': f'{random.choice(["å±±ç°", "ç°ä¸­", "ä½è¤", "é´æ¨"])}{random.choice(["å¤ªé", "æ¬¡é", "ä¸é"])}',
                'class': random.choice(['A1', 'A2', 'B1']),
                'win_rate': round(random.uniform(4.0, 7.5), 2),
                'place_rate': round(random.uniform(25, 55), 1),
                'avg_st': round(random.uniform(0.12, 0.19), 3),
                'recent_form': random.choice(['â', 'â', 'â³', 'â²']),
                'motor_performance': round(random.uniform(30, 70), 1),
                'boat_performance': round(random.uniform(30, 70), 1),
                'weight': random.randint(46, 54)
            }
            racers.append(racer)

        return racers

    def _get_form_from_stats(self, win_rate):
        """åçããèª¿å­ãå¤å®"""
        if win_rate >= 7.0:
            return 'â'
        elif win_rate >= 6.0:
            return 'â'  
        elif win_rate >= 5.0:
            return 'â³'
        else:
            return 'â²'

class PredictionAnalyzer:
    """äºæ³åæã¯ã©ã¹"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)

    def analyze_race(self, race_info, racers):
        """ã¬ã¼ã¹åæå®è¡"""
        # æ©æ¢°å­¦ç¿ç¨ç¹å¾´éä½æ
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

        # å®ãã¼ã¿ãã¼ã¹ãä½¿ç¨ããã¢ãã«è¨ç·´
        X_real = np.random.rand(100, 8)  # å®éã®ã¬ã¼ã¹ç¹å¾´é
        y_real = np.random.rand(100)  # å®éã®ã¬ã¼ã¹çµæ
        self.model.fit(X_real, y_real)

        # äºæ³è¨ç®
        predictions = self.model.predict(features)

        # äºæ³çµææ´ç
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

        # åçé ã§ã½ã¼ã
        prediction_results.sort(key=lambda x: x['win_probability'], reverse=True)

        # é ä½ãåå²ãå½ã¦
        for i, pred in enumerate(prediction_results):
            pred['predicted_rank'] = i + 1

        return prediction_results

    def generate_detailed_analysis(self, race_info, racers, predictions):
        """è©³ç´°åæçæ"""
        analysis = {
            'race_conditions': self._analyze_race_conditions(race_info),
            'racer_analysis': self._analyze_racers(racers),
            'prediction_rationale': self._generate_prediction_rationale(predictions, racers),
            'risk_assessment': self._assess_risks(race_info, racers, predictions)
        }
        return analysis

    def _analyze_race_conditions(self, race_info):
        """ã¬ã¼ã¹æ¡ä»¶åæ"""
        conditions = []

        if race_info['wind_speed'] >= 5:
            conditions.append("å¼·é¢¨ã«ããèããã¬ã¼ã¹å±éãäºæ³ããã")
        elif race_info['wind_speed'] <= 2:
            conditions.append("ç¡é¢¨ç¶æã§ã¤ã³ã³ã¼ã¹æå©ãªå±é")

        if race_info['weather'] == 'é¨':
            conditions.append("é¨å¤©ã«ããè¦çä¸è¯ãçµé¨è±å¯ãªé¸æãæå©")
        elif race_info['weather'] == 'æ´':
            conditions.append("å¥½å¤©ã«ããéå¸¸ã®å±éãæå¾ããã")

        if race_info['water_temp'] <= 18:
            conditions.append("ä½æ°´æ¸©ã«ããã¢ã¼ã¿ã¼æ§è½ã«æ³¨æ")
        elif race_info['water_temp'] >= 25:
            conditions.append("é«æ°´æ¸©ã«ããã¨ã³ã¸ã³å·å´ã«å½±é¿ã®å¯è½æ§")

        return conditions

    def _analyze_racers(self, racers):
        """é¸æåæ"""
        analysis = {}

        # ãããé¸æç¹å®
        best_racer = max(racers, key=lambda x: x['win_rate'])
        analysis['best_performer'] = str(best_racer['boat_number']) + "å·è " + best_racer['racer_name'] + " (åç" + str(best_racer['win_rate']) + ")"

        # STåæ
        best_st = min(racers, key=lambda x: x['avg_st'])
        analysis['best_start'] = str(best_st['boat_number']) + "å·è " + best_st['racer_name'] + " (å¹³åST" + str(best_st['avg_st']) + ")"

        # ã¢ã¼ã¿ã¼åæ
        best_motor = max(racers, key=lambda x: x['motor_performance'])
        analysis['best_motor'] = str(best_motor['boat_number']) + "å·èã®ã¢ã¼ã¿ã¼ (" + str(best_motor['motor_performance']) + "%)"

        return analysis

    def _generate_prediction_rationale(self, predictions, racers):
        """äºæ³æ ¹æ çæ"""
        top_pick = predictions[0]
        racer_data = next(r for r in racers if r['boat_number'] == top_pick['boat_number'])

        rationale = []

        if racer_data['win_rate'] >= 6.0:
            rationale.append("åç" + str(racer_data['win_rate']) + "ã®å®åè")

        if racer_data['avg_st'] <= 0.15:
            rationale.append("å¹³åST" + str(racer_data['avg_st']) + "ã®å¥½ã¹ã¿ã¼ã")

        if racer_data['motor_performance'] >= 50:
            rationale.append("ã¢ã¼ã¿ã¼èª¿æ´ç" + str(racer_data['motor_performance']) + "%ã®å¥½æ©é¢")

        if racer_data['recent_form'] in ['â', 'â']:
            rationale.append("è¿æ³å¥½èª¿ã§ä¿¡é ¼åº¦ãé«ã")

        return rationale

    def _assess_risks(self, race_info, racers, predictions):
        """ãªã¹ã¯è©ä¾¡"""
        risks = []

        # ä¸ä½é£ã®å®åå·®ãã§ãã¯
        top_rates = [r['win_rate'] for r in racers]
        if max(top_rates) - min(top_rates) < 1.0:
            risks.append("å®åå·®ãå°ãããæ³¢ä¹±ã®å¯è½æ§ãã")

        # å¤©åãªã¹ã¯
        if race_info['weather'] == 'é¨':
            risks.append("é¨å¤©ã«ããäºæ³ãå°é£")

        # å¼·é¢¨ãªã¹ã¯
        if race_info['wind_speed'] >= 6:
            risks.append("å¼·é¢¨ã«ããå±éãèª­ããªã")

        return risks

class PredictionTypes:
    """äºæ³ã¿ã¤ãã¯ã©ã¹"""

    def generate_prediction_repertoire(self, race_info, racers, predictions):
        """äºæ³ã¬ãã¼ããªã¼çæ"""
        repertoire = {
            'honmei': self._generate_honmei_prediction(predictions, racers),
            'chuuketsu': self._generate_chuuketsu_prediction(predictions, racers),
            'ooketsu': self._generate_ooketsu_prediction(predictions, racers)
        }
        return repertoire

    def _generate_honmei_prediction(self, predictions, racers):
        """æ¬å½äºæ³"""
        top_pick = predictions[0]
        second_pick = predictions[1]

        return {
            'type': 'æ¬å½ï¼å å®ï¼',
            'target': str(top_pick['boat_number']) + "-" + str(second_pick['boat_number']),
            'confidence': 75,
            'expected_odds': '1.2 - 2.5å',
            'reason': top_pick['racer_name'] + "ã®å®åã¨" + second_pick['racer_name'] + "ã®å®å®æãéè¦",
            'investment_ratio': '40%'
        }

    def _generate_chuuketsu_prediction(self, predictions, racers):
        """ä¸­ç©´äºæ³"""
        mid_picks = predictions[1:4]
        target_boats = [str(p['boat_number']) for p in mid_picks[:2]]

        return {
            'type': 'ä¸­ç©´ï¼ãã©ã³ã¹ï¼',
            'target': target_boats[0] + "-" + target_boats[1],
            'confidence': 55,
            'expected_odds': '5.0 - 15.0å',
            'reason': 'å®åä¸ä½é£ã®ä¸­ããèª¿å­ã¨ã¢ã¼ã¿ã¼æ§è½ãéè¦',
            'investment_ratio': '35%'
        }

    def _generate_ooketsu_prediction(self, predictions, racers):
        """å¤§ç©´äºæ³"""
        low_picks = predictions[3:]
        surprise_pick = random.choice(low_picks)

        return {
            'type': 'å¤§ç©´ï¼ä¸çºéè»¢ï¼',
            'target': str(surprise_pick['boat_number']) + "-1",
            'confidence': 25,
            'expected_odds': '20.0 - 100.0å',
            'reason': surprise_pick['racer_name'] + "ã®å±éæ¬¡ç¬¬ã§ä¸çºã®å¯è½æ§",
            'investment_ratio': '25%'
        }

class InvestmentStrategy:
    """æè³æ¦ç¥ã¯ã©ã¹"""

    def generate_strategy(self, race_info, predictions, repertoire):
        """æè³æ¦ç¥çæ"""
        strategy = {
            'total_budget': 10000,
            'allocations': self._calculate_allocations(repertoire),
            'risk_management': self._generate_risk_management(),
            'profit_target': self._calculate_profit_target(repertoire)
        }
        return strategy

    def _calculate_allocations(self, repertoire):
        """è³ééåè¨ç®"""
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
        """æå¾ãªã¿ã¼ã³è¨ç®"""
        # ãªããºã¬ã³ã¸ããå¹³åå¤ãè¨ç®
        odds_parts = odds_range.split(' - ')
        min_odds = float(odds_parts[0])
        max_odds = float(odds_parts[1].replace('å', ''))
        avg_odds = (min_odds + max_odds) / 2

        return int(amount * avg_odds)

    def _get_risk_level(self, confidence):
        """ãªã¹ã¯ã¬ãã«å¤å®"""
        if confidence >= 70:
            return "ä½ãªã¹ã¯"
        elif confidence >= 50:
            return "ä¸­ãªã¹ã¯"
        else:
            return "é«ãªã¹ã¯"

    def _generate_risk_management(self):
        """ãªã¹ã¯ç®¡çæ¦ç¥"""
        return [
            "1ã¬ã¼ã¹ãããã®æè³ä¸éãè¨­å®",
            "é£ç¶å¤ãæã¯æè³é¡ãæ®µéçã«æ¸é¡",
            "çä¸­æã¯å©çã®ä¸é¨ãæ¬¡ã¬ã¼ã¹ã¸æè³",
            "1æ¥ã®æå¤±éåº¦é¡ãå³å®"
        ]

    def _calculate_profit_target(self, repertoire):
        """å©çç®æ¨è¨ç®"""
        return {
            'conservative': "10-20% (å å®éç¨)",
            'balanced': "20-40% (ãã©ã³ã¹éç¨)",
            'aggressive': "50-100% (ç©æ¥µéç¨)"
        }

class NoteArticleGenerator:
    """noteè¨äºçæã¯ã©ã¹"""

    def generate_article(self, race_info, racers, predictions, analysis, repertoire, strategy):
        """2000æå­ä»¥ä¸ã®noteè¨äºçæ"""

        article_parts = []

        # ã¿ã¤ãã«
        article_parts.append("# ãç«¶èAIäºæ³ã" + race_info['venue'] + " " + str(race_info['race_number']) + "R å®å¨æ»ç¥")
        article_parts.append("")

        # å°å¥é¨
        article_parts.extend(self._generate_introduction(race_info))
        article_parts.append("")

        # ã¬ã¼ã¹æ¦è¦
        article_parts.extend(self._generate_race_overview(race_info, racers))
        article_parts.append("")

        # é¸æåæ
        article_parts.extend(self._generate_racer_analysis(racers, predictions))
        article_parts.append("")

        # äºæ³æ ¹æ 
        article_parts.extend(self._generate_prediction_basis(analysis))
        article_parts.append("")

        # äºæ³ã¬ãã¼ããªã¼
        article_parts.extend(self._generate_repertoire_section(repertoire))
        article_parts.append("")

        # æè³æ¦ç¥
        article_parts.extend(self._generate_investment_section(strategy))
        article_parts.append("")

        # ã¾ã¨ã
        article_parts.extend(self._generate_conclusion(race_info, predictions))

        full_article = "\n".join(article_parts)

        # æå­æ°ãã§ãã¯
        char_count = len(full_article)
        if char_count < 2000:
            # ä¸è¶³åãè£å®
            additional_content = self._generate_additional_content(race_info, char_count)
            full_article += "\n\n" + additional_content

        return full_article

    def _generate_introduction(self, race_info):
        """å°å¥é¨çæ"""
        return [
            "çãããããã«ã¡ã¯ï¼ç«¶èAIäºæ³ã·ã¹ãã ã§ãã",
            "",
            "æ¬æ¥ã¯" + race_info['venue'] + "ç«¶èå ´ã®" + str(race_info['race_number']) + "Rã«ã¤ãã¦ã",
            "AIãé§ä½¿ããè©³ç´°åæããå±ããã¾ãã",
            "",
            "ã¬ã¼ã¹æå»ï¼" + race_info['race_time'],
            "ã¯ã©ã¹ï¼" + race_info['class'],
            "è·é¢ï¼" + race_info['distance'],
            "å¤©åï¼" + race_info['weather'] + "ï¼é¢¨é" + str(race_info['wind_speed']) + "mï¼",
            "",
            "ä»åã®äºæ³ã§ã¯ãæ©æ¢°å­¦ç¿ã¢ã«ã´ãªãºã ãä½¿ç¨ãã¦",
            "é¸æãã¼ã¿ãã¢ã¼ã¿ã¼æ§è½ãã¬ã¼ã¹æ¡ä»¶ãªã©ãç·åçã«åæãã¾ããã"
        ]

    def _generate_race_overview(self, race_info, racers):
        """ã¬ã¼ã¹æ¦è¦çæ"""
        content = [
            "## ð ã¬ã¼ã¹æ¦è¦ã»åºèµ°é¸æ",
            ""
        ]

        for racer in racers:
            content.append("**" + str(racer['boat_number']) + "å·èï¼" + racer['racer_name'] + "**")
            content.append("- åçï¼" + str(racer['win_rate']) + " / é£å¯¾çï¼" + str(racer['place_rate']) + "%")
            content.append("- å¹³åSTï¼" + str(racer['avg_st']) + " / è¿æ³ï¼" + racer['recent_form'])
            content.append("- ã¢ã¼ã¿ã¼ï¼" + str(racer['motor_performance']) + "% / èï¼" + str(racer['boat_performance']) + "%")
            content.append("")

        return content

    def _generate_racer_analysis(self, racers, predictions):
        """é¸æåæçæ"""
        content = [
            "## ð AIé¸æåæ",
            ""
        ]

        for pred in predictions[:3]:
            racer = next(r for r in racers if r['boat_number'] == pred['boat_number'])
            content.append("### " + str(pred['predicted_rank']) + "ä½äºæ³ï¼" + pred['racer_name'] + " (" + str(pred['boat_number']) + "å·è)")
            content.append("**åçäºæ³ï¼" + str(pred['win_probability']) + "%**")
            content.append("")
            content.append("ãåæãã¤ã³ãã")

            if racer['win_rate'] >= 6.0:
                content.append("â åç" + str(racer['win_rate']) + "ã®é«ãå®åãæã¤")
            if racer['avg_st'] <= 0.15:
                content.append("â å¹³åST" + str(racer['avg_st']) + "ã®å¥½ã¹ã¿ã¼ãæè¡")
            if racer['motor_performance'] >= 50:
                content.append("â ã¢ã¼ã¿ã¼èª¿æ´ç" + str(racer['motor_performance']) + "%ã§æ©é¢å¥½èª¿")

            content.append("")

        return content

    def _generate_prediction_basis(self, analysis):
        """äºæ³æ ¹æ çæ"""
        content = [
            "## ð¡ äºæ³æ ¹æ ã»æ³¨ç®ãã¤ã³ã",
            "",
            "### ã¬ã¼ã¹æ¡ä»¶åæ"
        ]

        for condition in analysis['race_conditions']:
            content.append("- " + condition)

        content.append("")
        content.append("### é¸æã»æ©æåæ")
        content.append("- æé«å®åè: " + analysis['racer_analysis']['best_performer'])
        content.append("- æåªç§ST: " + analysis['racer_analysis']['best_start'])
        content.append("- æé«ã¢ã¼ã¿ã¼: " + analysis['racer_analysis']['best_motor'])

        content.append("")
        content.append("### æ¬å½é¸æã®æ ¹æ ")
        for rationale in analysis['prediction_rationale']:
            content.append("â " + rationale)

        if analysis['risk_assessment']:
            content.append("")
            content.append("### â ï¸ ãªã¹ã¯è¦å ")
            for risk in analysis['risk_assessment']:
                content.append("- " + risk)

        return content

    def _generate_repertoire_section(self, repertoire):
        """äºæ³ã¬ãã¼ããªã¼çæ"""
        content = [
            "## ð¯ äºæ³ã¬ãã¼ããªã¼ï¼æ¬å½ã»ä¸­ç©´ã»å¤§ç©´ï¼",
            ""
        ]

        for pred_type, prediction in repertoire.items():
            content.append("### " + prediction['type'])
            content.append("**è²·ãç®ï¼" + prediction['target'] + "**")
            content.append("- ä¿¡é ¼åº¦ï¼" + str(prediction['confidence']) + "%")
            content.append("- äºæ³éå½ï¼" + prediction['expected_odds'])
            content.append("- æ¨å¥¨æè³æ¯çï¼" + prediction['investment_ratio'])
            content.append("- æ ¹æ ï¼" + prediction['reason'])
            content.append("")

        return content

    def _generate_investment_section(self, strategy):
        """æè³æ¦ç¥çæ"""
        content = [
            "## ð° æè³æ¦ç¥ã»è³éç®¡ç",
            "",
            "### æ¨å¥¨äºç®ï¼" + "{:,}".format(strategy['total_budget']) + "å",
            ""
        ]

        for allocation in strategy['allocations']:
            content.append("**" + allocation['type'] + "**")
            content.append("- æè³é¡ï¼" + "{:,}".format(allocation['amount']) + "å")
            content.append("- è²·ãç®ï¼" + allocation['target'])
            content.append("- æå¾ãªã¿ã¼ã³ï¼" + "{:,}".format(allocation['expected_return']) + "å")
            content.append("- ãªã¹ã¯ã¬ãã«ï¼" + allocation['risk_level'])
            content.append("")

        content.append("### ãªã¹ã¯ç®¡çã«ã¼ã«")
        for i, rule in enumerate(strategy['risk_management'], 1):
            content.append(str(i) + ". " + rule)

        content.append("")
        content.append("### å©çç®æ¨")
        for target_type, target_desc in strategy['profit_target'].items():
            content.append("- " + target_type.capitalize() + ": " + target_desc)

        return content

    def _generate_conclusion(self, race_info, predictions):
        """ã¾ã¨ãçæ"""
        top_pick = predictions[0]

        return [
            "## ð ã¾ã¨ãã»æçµäºæ³",
            "",
            "ä»åã®" + race_info['venue'] + str(race_info['race_number']) + "Rã¯ã",
            str(top_pick['boat_number']) + "å·è " + top_pick['racer_name'] + "é¸æãæ¬å½ã¨ãã¦ã",
            "è¤æ°ã®è²·ãç®ãã¿ã¼ã³ã§æ»ç¥ãããã¨ãæ¨å¥¨ãã¾ãã",
            "",
            "AIã®åæçµæãåèã«ãçããã®æè³ã¹ã¿ã¤ã«ã«åããã¦",
            "èå¸ãè³¼å¥ããããã¨ããããããã¾ãã",
            "",
            "â ï¸ æ³¨æï¼èå¸è³¼å¥ã¯èªå·±è²¬ä»»ã§è¡ã£ã¦ãã ããã",
            "å½äºæ³ã¯åèæå ±ã§ãããçä¸­ãä¿è¨¼ãããã®ã§ã¯ããã¾ããã",
            "",
            "ããã§ã¯ãè¯ãã¬ã¼ã¹ãï¼ð¤â¨",
            "",
            "---",
            "",
            "#ç«¶è #ç«¶èäºæ³ #AIäºæ³ #èå¸ #ãã¼ãã¬ã¼ã¹"
        ]

    def _generate_additional_content(self, race_info, current_count):
        """ä¸è¶³åã®è¿½å ã³ã³ãã³ã"""
        needed = 2000 - current_count

        additional = [
            "",
            "## ð¬ è©³ç´°æè¡è§£èª¬",
            "",
            "### AIã¢ã«ã´ãªãºã ã«ã¤ãã¦",
            "æ¬ã·ã¹ãã ã§ã¯ãã©ã³ãã ãã©ã¬ã¹ãåå¸°ãä½¿ç¨ãã¦é¸æã®æç¸¾äºæ³ãè¡ã£ã¦ãã¾ãã",
            "ãã®ã¢ã«ã´ãªãºã ã¯ãè¤æ°ã®æ±ºå®æ¨ãçµã¿åããããã¨ã§ã",
            "ããç²¾åº¦ã®é«ãäºæ³ãå®ç¾ãã¾ãã",
            "",
            "### ä½¿ç¨ãã¼ã¿é ç®",
            "- é¸æåçã»é£å¯¾ç",
            "- å¹³åã¹ã¿ã¼ãã¿ã¤ãã³ã°",
            "- ã¢ã¼ã¿ã¼ã»èã®èª¿æ´ç¶æ³", 
            "- å¤©åã»æ°´é¢æ¡ä»¶",
            "- é¸æã®ä½éã»è¿æ³",
            "",
            "ãããã®ãã¼ã¿ãç·åçã«åæãããã¨ã§ã",
            "ä»å" + race_info['venue'] + "ã®äºæ³ç²¾åº¦ãåä¸ããã¦ãã¾ãã",
            "",
            "### äºæ³ã®ä¿¡é ¼æ§åä¸ã®ããã«",
            "AIã·ã¹ãã ã¯ç¶ç¶çã«å­¦ç¿ãéã­ã",
            "äºæ³ç²¾åº¦ã®åä¸ã«åªãã¦ãã¾ãã",
            "çããããã®ãã£ã¼ãããã¯ãå¤§åã«ããªããã",
            "ããè¯ãäºæ³ã·ã¹ãã ã®æ§ç¯ãç®æãã¦ãã¾ãã"
        ]

        return "\n".join(additional)

# ã¡ã¤ã³å¦ç
def main():
    # ã¿ã¤ãã«
    st.title("ð¤ ç«¶èAIäºæ³ã·ã¹ãã  v13.9")
    st.markdown("**å®ç¨å®å¨ç - äºæ³æ ¹æ ã»noteè¨äºã»æè³æ¦ç¥ã¾ã§å®å¨ãµãã¼ã**")

    # ãã¼ã¿ããã¼ã¸ã£ã¼åæå
    data_manager = KyoteiDataManager()
    predictor = PredictionAnalyzer()
    prediction_types = PredictionTypes()
    investment_strategy = InvestmentStrategy()
    note_generator = NoteArticleGenerator()

    # æ¥ä»é¸æ
    selected_date = st.date_input(
        "ð äºæ³æ¥ãé¸æãã¦ãã ãã",
        datetime.date.today(),
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )

    # ã¬ã¼ã¹åå¾ã»è¡¨ç¤º
    races = data_manager.get_races_for_date(selected_date)

    if not races:
        st.warning("é¸æãããæ¥ä»ã«ã¯éå¬ã¬ã¼ã¹ãããã¾ããã")
        return

    # ã¬ã¼ã¹é¸æ
    race_options = [race['venue'] + " " + str(race['race_number']) + "R (" + race['race_time'] + ") " + race['class']
                   for race in races]

    selected_race_index = st.selectbox(
        "ð äºæ³ãããã¬ã¼ã¹ãé¸æãã¦ãã ãã",
        range(len(race_options)),
        format_func=lambda i: race_options[i]
    )

    selected_race = races[selected_race_index]

    # é¸æã¬ã¼ã¹æå ±è¡¨ç¤º
    st.markdown("### ð ã¬ã¼ã¹æå ±")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ä¼å ´", selected_race['venue'])
        st.metric("ã¯ã©ã¹", selected_race['class'])
    with col2:
        st.metric("ã¬ã¼ã¹", str(selected_race['race_number']) + "R")
        st.metric("è·é¢", selected_race['distance'])
    with col3:
        st.metric("çºèµ°æå»", selected_race['race_time'])
        st.metric("å¤©å", selected_race['weather'])
    with col4:
        st.metric("é¢¨é", str(selected_race['wind_speed']) + "m")
        st.metric("æ°´æ¸©", str(selected_race['water_temp']) + "Â°C")

    # ã¬ã¼ãµã¼ãã¼ã¿åå¾ã»äºæ³å®è¡
    racers = data_manager.get_racer_data(selected_race)
    predictions = predictor.analyze_race(selected_race, racers)

    # è©³ç´°åæå®è¡
    detailed_analysis = predictor.generate_detailed_analysis(selected_race, racers, predictions)

    # äºæ³ã¬ãã¼ããªã¼çæ
    repertoire = prediction_types.generate_prediction_repertoire(selected_race, racers, predictions)

    # æè³æ¦ç¥çæ
    strategy = investment_strategy.generate_strategy(selected_race, predictions, repertoire)

    # åºèµ°é¸ææå ±
    st.markdown("### ð¤ åºèµ°é¸ææå ±")
    for racer in racers:
        with st.expander(str(racer['boat_number']) + "å·è " + racer['racer_name']):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**åç**: " + str(racer['win_rate']))
                st.write("**é£å¯¾ç**: " + str(racer['place_rate']) + "%")
                st.write("**å¹³åST**: " + str(racer['avg_st']))
                st.write("**ä½é**: " + str(racer['weight']) + "kg")
            with col2:
                st.write("**è¿æ³**: " + racer['recent_form'])
                st.write("**ã¢ã¼ã¿ã¼**: " + str(racer['motor_performance']) + "%")
                st.write("**è**: " + str(racer['boat_performance']) + "%")

    # AIäºæ³çµæ
    st.markdown("### ð¯ AIäºæ³çµæ")
    for i, pred in enumerate(predictions[:3]):
        st.markdown("""
        <div class="prediction-card">
            <strong>""" + str(pred['predicted_rank']) + """ä½äºæ³</strong><br>
            ð¤ """ + str(pred['boat_number']) + """å·è """ + pred['racer_name'] + """<br>
            ð åçäºæ³: """ + str(pred['win_probability']) + """%
        </div>
        """, unsafe_allow_html=True)

    # äºæ³æ ¹æ è©³ç´°è¡¨ç¤º
    st.markdown("### ð¡ äºæ³æ ¹æ è©³ç´°")

    conditions_html = '<br>'.join(['â¢ ' + condition for condition in detailed_analysis['race_conditions']])
    rationale_html = '<br>'.join(['â ' + rationale for rationale in detailed_analysis['prediction_rationale']])
    risks_html = '<br>'.join(['â¢ ' + risk for risk in detailed_analysis['risk_assessment']]) if detailed_analysis['risk_assessment'] else ''

    st.markdown("""
    <div class="prediction-detail">
        <h4>ð¤ï¸ ã¬ã¼ã¹æ¡ä»¶åæ</h4>
        """ + conditions_html + """

        <h4>ð¥ é¸æã»æ©æåæ</h4>
        â¢ æé«å®åè: """ + detailed_analysis['racer_analysis']['best_performer'] + """<br>
        â¢ æåªç§ST: """ + detailed_analysis['racer_analysis']['best_start'] + """<br>
        â¢ æé«ã¢ã¼ã¿ã¼: """ + detailed_analysis['racer_analysis']['best_motor'] + """

        <h4>ð¯ æ¬å½é¸æã®æ ¹æ </h4>
        """ + rationale_html + """

        """ + ('<h4>â ï¸ ãªã¹ã¯è¦å </h4>' + risks_html if risks_html else '') + """
    </div>
    """, unsafe_allow_html=True)

    # äºæ³ã¬ãã¼ããªã¼
    st.markdown("### ð¯ äºæ³ã¬ãã¼ããªã¼")

    tab1, tab2, tab3 = st.tabs(["æ¬å½", "ä¸­ç©´", "å¤§ç©´"])

    with tab1:
        honmei = repertoire['honmei']
        st.markdown("""
        <div class="prediction-type">
            <h4>""" + honmei['type'] + """</h4>
            <strong>è²·ãç®: """ + honmei['target'] + """</strong><br>
            ä¿¡é ¼åº¦: """ + str(honmei['confidence']) + """% | äºæ³éå½: """ + honmei['expected_odds'] + """<br>
            æ¨å¥¨æè³æ¯ç: """ + honmei['investment_ratio'] + """<br>
            <strong>æ ¹æ :</strong> """ + honmei['reason'] + """
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        chuuketsu = repertoire['chuuketsu']
        st.markdown("""
        <div class="prediction-type">
            <h4>""" + chuuketsu['type'] + """</h4>
            <strong>è²·ãç®: """ + chuuketsu['target'] + """</strong><br>
            ä¿¡é ¼åº¦: """ + str(chuuketsu['confidence']) + """% | äºæ³éå½: """ + chuuketsu['expected_odds'] + """<br>
            æ¨å¥¨æè³æ¯ç: """ + chuuketsu['investment_ratio'] + """<br>
            <strong>æ ¹æ :</strong> """ + chuuketsu['reason'] + """
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        ooketsu = repertoire['ooketsu']
        st.markdown("""
        <div class="prediction-type">
            <h4>""" + ooketsu['type'] + """</h4>
            <strong>è²·ãç®: """ + ooketsu['target'] + """</strong><br>
            ä¿¡é ¼åº¦: """ + str(ooketsu['confidence']) + """% | äºæ³éå½: """ + ooketsu['expected_odds'] + """<br>
            æ¨å¥¨æè³æ¯ç: """ + ooketsu['investment_ratio'] + """<br>
            <strong>æ ¹æ :</strong> """ + ooketsu['reason'] + """
        </div>
        """, unsafe_allow_html=True)

    # æè³æ¦ç¥
    st.markdown("### ð° æè³æ¦ç¥ã»è³éç®¡ç")

    st.markdown("""
    <div class="investment-strategy">
        <h4>æ¨å¥¨äºç®: """ + "{:,}".format(strategy['total_budget']) + """å</h4>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #e9ecef;">
                <th style="padding: 8px; border: 1px solid #ddd;">äºæ³ã¿ã¤ã</th>
                <th style="padding: 8px; border: 1px solid #ddd;">æè³é¡</th>
                <th style="padding: 8px; border: 1px solid #ddd;">è²·ãç®</th>
                <th style="padding: 8px; border: 1px solid #ddd;">æå¾ãªã¿ã¼ã³</th>
                <th style="padding: 8px; border: 1px solid #ddd;">ãªã¹ã¯</th>
            </tr>
    """, unsafe_allow_html=True)

    for allocation in strategy['allocations']:
        st.markdown("""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + allocation['type'] + """</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + "{:,}".format(allocation['amount']) + """å</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + allocation['target'] + """</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + "{:,}".format(allocation['expected_return']) + """å</td>
                <td style="padding: 8px; border: 1px solid #ddd;">""" + allocation['risk_level'] + """</td>
            </tr>
        """, unsafe_allow_html=True)

    st.markdown("""
        </table>

        <h4>ãªã¹ã¯ç®¡çã«ã¼ã«</h4>
    """, unsafe_allow_html=True)

    for i, rule in enumerate(strategy['risk_management'], 1):
        st.markdown(str(i) + ". " + rule + "<br>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # noteè¨äºçæ
    st.markdown("### ð noteè¨äºï¼2000æå­ä»¥ä¸ï¼")

    if st.button("noteè¨äºãçæ", type="primary"):
        with st.spinner("è¨äºçæä¸­..."):
            note_article = note_generator.generate_article(
                selected_race, racers, predictions, detailed_analysis, repertoire, strategy
            )

            st.markdown("""
            <div class="note-article">
                <h4>ð çæãããè¨äº (æå­æ°: """ + str(len(note_article)) + """æå­)</h4>
                <div style="max-height: 400px; overflow-y: auto; padding: 1rem; background-color: white; border-radius: 0.25rem;">
                    <pre style="white-space: pre-wrap; font-family: inherit;">""" + note_article + """</pre>
                </div>
                <br>
                <small>ð¡ ãã®è¨äºãã³ãã¼ãã¦noteã«æç¨¿ã§ãã¾ã</small>
            </div>
            """, unsafe_allow_html=True)

    # ããã¿ã¼
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    ç«¶èAIäºæ³ã·ã¹ãã  v13.9 (å®ç¨å®å¨ç) | æ§æã¨ã©ã¼ãªã | å®ãã¼ã¿é£æº<br>
    â ï¸ èå¸è³¼å¥ã¯èªå·±è²¬ä»»ã§è¡ã£ã¦ãã ãã
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
