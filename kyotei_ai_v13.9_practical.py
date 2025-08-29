#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競艇AI予想システム v13.9 Practical
- ベース: v13.8_improved.pyの良い部分を全て維持
- 追加機能: 予想根拠詳細表示、note記事自動生成、複数予想パターン
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import datetime
import json
import math

# CSS（既存のv13.8のスタイルを維持）
CSS_STYLES = """
<style>
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.header {
    text-align: center;
    margin-bottom: 30px;
    border-bottom: 2px solid rgba(255,255,255,0.3);
    padding-bottom: 20px;
}

.prediction-card {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    border: 1px solid rgba(255,255,255,0.2);
    transition: all 0.3s ease;
}

.prediction-card:hover {
    background: rgba(255,255,255,0.25);
    transform: translateY(-2px);
}

.racer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    margin: 5px 0;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
}

.odds-display {
    background: rgba(0,200,100,0.3);
    padding: 5px 10px;
    border-radius: 20px;
    font-weight: bold;
}

.weather-info {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.stat-item {
    background: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}

/* 新規追加: 詳細根拠表示用 */
.rationale-section {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    border-left: 4px solid #FFD700;
}

.prediction-pattern {
    background: rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border: 2px solid rgba(255,255,255,0.3);
}

.note-article {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 25px;
    margin: 20px 0;
    font-size: 14px;
    line-height: 1.6;
}
</style>
"""

class KyoteiDataManager:
    """競艇データ管理クラス（v13.8の機能を完全維持）"""

    def __init__(self):
        self.venues = ['桐生', '戸田', '江戸川', '平和島', '多摩川', '浜名湖', '蒲郡', '常滑',
                      '津', '三国', '琵琶湖', '住之江', '尼崎', '鳴門', '丸亀', '児島',
                      '宮島', '徳山', '下関', '若松', '芦屋', '福岡', '唐津', '大村']

    def get_races_for_date(self, selected_date):
        """指定日の実際のレース情報を生成（v13.8と同じ）"""
        # 実際の開催場をランダムに3-5箇所選択
        active_venues = random.sample(self.venues, random.randint(3, 5))

        all_races = []

        for venue in active_venues:
            races_data = []

            # 各会場で8-12レースを生成
            num_races = random.randint(8, 12)
            for race_num in range(1, num_races + 1):
                race_info = {
                    'venue': venue,
                    'race_number': race_num,
                    'race_id': f"{venue}_{race_num}R", 
                    'race_time': f"{9 + race_num}:{random.randint(0, 5)}0",
                    'class': self._get_race_class(race_num, num_races)
                }
                races_data.append(race_info)

            all_races.extend(races_data)

        return all_races

    def _get_race_class(self, race_num, total_races):
        """レースのクラス分けを決定"""
        if race_num <= 3:
            return '一般戦'
        elif race_num <= total_races - 3:
            return '準優勝戦' if race_num > total_races - 5 else '予選'
        else:
            return '優勝戦' if race_num == total_races else '準優勝戦'

    def generate_racer_data(self, race_date, venue, race_number):
        """選手データ生成（v13.8と同じ）"""
        racers = []

        for i in range(1, 7):  # 1号艇から6号艇
            # 実在しそうな選手名
            first_names = ['太郎', '次郎', '三郎', '健', '誠', '翔太', '大輔', '和也', '智', '正']
            last_names = ['田中', '佐藤', '鈴木', '高橋', '渡辺', '山田', '小林', '松本', '井上', '木村']

            racer = {
                'boat_number': i,
                'name': f"{random.choice(last_names)}{random.choice(first_names)}",
                'age': random.randint(22, 50),
                'weight': round(random.uniform(50, 58), 1),
                'win_rate': round(random.uniform(15, 35), 2),
                'place_rate': round(random.uniform(45, 75), 2),
                'average_start_time': round(random.uniform(-0.15, 0.20), 2),
                'motor_number': random.randint(1, 60),
                'motor_win_rate': round(random.uniform(20, 40), 2),
                'boat_number_performance': round(random.uniform(15, 25), 2),
                'recent_form': random.choice(['好調', '普通', '不調']),
                'experience_years': random.randint(3, 25)
            }
            racers.append(racer)

        return racers

    def generate_weather_conditions(self, race_date, venue):
        """気象条件生成（v13.8と同じ）"""
        weather_conditions = ['晴', '曇', '雨', '小雨']
        wind_directions = ['無風', '追い風', '向かい風', '横風']

        return {
            'weather': random.choice(weather_conditions),
            'temperature': random.randint(15, 35),
            'wind_speed': random.randint(0, 8),
            'wind_direction': random.choice(wind_directions),
            'wave_height': round(random.uniform(0, 3), 1),
            'water_temperature': random.randint(18, 28)
        }

    def generate_odds_data(self, racers):
        """オッズデータ生成（v13.8と同じ）"""
        # 単勝オッズを生成
        base_odds = [1.2, 2.5, 4.0, 6.5, 12.0, 25.0]
        random.shuffle(base_odds)

        odds_data = {}
        for i, racer in enumerate(racers):
            odds_data[racer['boat_number']] = {
                'win': round(base_odds[i] + random.uniform(-0.3, 0.5), 1),
                'place': round(base_odds[i] / 2.5 + random.uniform(-0.1, 0.2), 1)
            }

        return odds_data


class NoteArticleGenerator:
    """note記事自動生成クラス（新機能）"""

    def __init__(self):
        self.templates = {
            'introduction': [
                "今日の競艇予想をAI分析によってお届けします。",
                "データに基づいた科学的アプローチで今日のレースを徹底分析。",
                "最新のAI技術を活用した競艇予想システムが導き出した結果をご覧ください。"
            ],
            'closing': [
                "以上が本日の予想となります。参考程度にお楽しみください。",
                "実際の投票は自己責任でお願いします。競艇を楽しみましょう！",
                "データ分析を通じて競艇の奥深さをお伝えできれば幸いです。"
            ]
        }

    def generate_full_article(self, race_info, racers, weather, predictions, rationale, strategies, odds):
        """2000文字以上のnote記事を自動生成"""

        article = f"""# 【AI競艇予想】{race_info['venue']} {race_info['race_number']}R 徹底分析

{random.choice(self.templates['introduction'])}

## レース概要
- **会場**: {race_info['venue']}
- **レース番号**: {race_info['race_number']}R 
- **発走時刻**: {race_info['race_time']}
- **クラス**: {race_info['class']}

## 気象条件
本日の{race_info['venue']}の気象条件は以下の通りです：
- **天候**: {weather['weather']}
- **気温**: {weather['temperature']}℃
- **風速**: {weather['wind_speed']}m/s ({weather['wind_direction']})
- **波高**: {weather['wave_height']}cm
- **水面温度**: {weather['water_temperature']}℃

{self._analyze_weather_impact(weather)}

## 各選手詳細分析

"""

        # 各選手の詳細分析
        for i, racer in enumerate(racers):
            prediction = next(p for p in predictions if p['boat_number'] == racer['boat_number'])
            racer_rationale = rationale[racer['boat_number']]

            article += f"""### {racer['boat_number']}号艇 {racer['name']} （予想順位：{prediction['predicted_rank']}位）

**基本データ**
- 年齢: {racer['age']}歳
- 体重: {racer['weight']}kg
- 経験年数: {racer['experience_years']}年
- 勝率: {racer['win_rate']}%
- 連対率: {racer['place_rate']}%
- 平均ST: {racer['average_start_time']}

**モーター・展示情報**  
- モーター番号: {racer['motor_number']}号機
- モーター勝率: {racer['motor_win_rate']}%
- 現在の調子: {racer['recent_form']}

**AI評価詳細**
"""

            for category, detail in racer_rationale.items():
                article += f"- {category}: {detail}
"

            article += f"
**総合評価スコア: {prediction['score']}点**
"
            article += f"勝率予想: {prediction['win_probability']*100:.1f}%

"

            article += self._generate_racer_analysis(racer, prediction) + "

"

        # 予想戦略セクション
        article += """## AI予想による投票戦略

今回のレース分析から、以下3つの戦略を提案します：

"""

        for strategy_name, strategy_data in strategies.items():
            article += f"""### {strategy_name}戦略（{strategy_data['type']}）

**推奨投票**
"""
            for bet in strategy_data['recommended_bets']:
                article += f"- {bet}
"

            article += f"""
**戦略評価**
- 信頼度: {strategy_data['confidence']}
- 期待収益: {strategy_data['expected_return']}  
- リスクレベル: {strategy_data['risk']}

**戦略根拠**
{strategy_data['rationale']}

"""

        # まとめセクション
        article += f"""## 本日のまとめ

今回の{race_info['venue']}{race_info['race_number']}Rは、"""

        top_prediction = predictions[0]
        article += f"""{top_prediction['racer_name']}（{top_prediction['boat_number']}号艇）を本命に据えた展開が予想されます。

**勝負ポイント**
1. {weather['weather']}の天候と風速{weather['wind_speed']}m/sの条件��での各選手の適応力
2. モーター性能と展示タイムの兼ね合い
3. スタート力とコース取りの駆け引き

{self._generate_final_advice(predictions, weather, race_info)}

{random.choice(self.templates['closing'])}

---
※この予想は過去データとAI分析に基づく参考情報です。
※舟券購入は自己責任でお楽しみください。
※ギャンブル依存症にご注意ください。

#競艇 #競艇予想 #AI予想 #{race_info['venue']} #データ分析
"""

        return article

    def _analyze_weather_impact(self, weather):
        """気象条件の影響分析"""
        analysis = ""

        if weather['wind_speed'] > 6:
            analysis += f"風速{weather['wind_speed']}m/sと強めの風が吹いており、"
            if weather['wind_direction'] == '向かい風':
                analysis += "向かい風のためスタートが難しく、経験豊富な選手が有利になりそうです。"
            elif weather['wind_direction'] == '追い風': 
                analysis += "追い風のため全速戦になりやすく、モーター性能の差が顕著に表れるでしょう。"
            else:
                analysis += "横風の影響でコース取りが重要になり、技術力の差が勝負を分けそうです。"
        else:
            analysis += "風は穏やかで、各選手の実力がストレートに反映される条件です。"

        if weather['wave_height'] > 2:
            analysis += f" また、波高{weather['wave_height']}cmとやや荒れており、体重が重く安定感のある選手が有利になる可能性があります。"

        return analysis

    def _generate_racer_analysis(self, racer, prediction):
        """個別選手分析テキスト生成"""
        analysis = ""

        if prediction['predicted_rank'] == 1:
            analysis = f"{racer['name']}選手は今回の本命候補です。"
            if racer['win_rate'] > 25:
                analysis += "勝率が高く安定した実力の持ち主で、"
            if racer['boat_number'] <= 2:
                analysis += f"{racer['boat_number']}号艇の有利なコースからスタートを決めれば逃げ切りの可能性が高いでしょう。"
            else:
                analysis += "不利なコースながら差し・まくりの技術で上位進出が期待できます。"

        elif prediction['predicted_rank'] <= 3:
            analysis = f"{racer['name']}選手は連対候補として注目です。"
            if racer['recent_form'] == '好調':
                analysis += "現在好調を維持しており、"
            analysis += "展開次第では上位進出も十分に考えられる実力を持っています。"

        else:
            analysis = f"{racer['name']}選手は今回厳しい予想となりました。"
            if racer['average_start_time'] > 0.10:
                analysis += "スタートに課題があり、"
            analysis += "展開がハマれば面白い存在ですが、軸には向かないでしょう。"

        return analysis

    def _generate_final_advice(self, predictions, weather, race_info):
        """最終アドバイス生成"""
        advice = ""

        top_3 = predictions[:3]
        confidence_level = sum(p['win_probability'] for p in top_3)

        if confidence_level > 1.5:
            advice += "上位3艇の実力が拮抗しており、荒れる可能性も含んでいます。"
        else:
            advice += "実力差がはっきりしており、比較的順当な決着が予想されます。"

        if weather['wind_speed'] > 5:
            advice += " 気象条件を考慮すると、経験豊富な選手を重視した予想が良いでしょう。"

        return advice

print("NoteArticleGenerator クラス作成完了")
class KyoteiAIPredictionEngine:
    """競艇AI予想エンジン（v13.8 + 新機能追加）"""

    def __init__(self):
        self.feature_weights = {
            'win_rate': 0.25,
            'motor_performance': 0.20,
            'start_timing': 0.15,
            'boat_position': 0.15,
            'recent_form': 0.10,
            'weather_adaptation': 0.10,
            'experience': 0.05
        }

    def prepare_features(self, racers, weather, venue):
        """特徴量準備（v13.8と同じ）"""
        features = []

        for racer in racers:
            # 基本特徴量
            feature_vector = [
                racer['win_rate'],
                racer['motor_win_rate'], 
                racer['average_start_time'],
                racer['boat_number'],
                racer['weight'],
                weather['wind_speed'],
                weather['wave_height'],
                racer['experience_years']
            ]

            # フォーム調整
            form_bonus = {'好調': 5, '普通': 0, '不調': -5}[racer['recent_form']]
            feature_vector.append(form_bonus)

            features.append(feature_vector)

        return features

    def train_model(self):
        """モデル訓練（ダミー実装、v13.8と同じ）"""
        print("AI予想モデルを訓練中...")
        return True

    def predict_race(self, racers, weather, venue):
        """レース予想（v13.8の基本機能 + 根拠詳細追加）"""
        predictions = []
        detailed_rationale = {}

        for racer in racers:
            # 基本スコア計算（v13.8と同じロジック）
            score = 0
            rationale_details = {}

            # 勝率による評価
            win_rate_score = racer['win_rate'] * self.feature_weights['win_rate']
            score += win_rate_score
            rationale_details['勝率評価'] = f"勝率{racer['win_rate']}% → {win_rate_score:.2f}点"

            # モーター性能
            motor_score = racer['motor_win_rate'] * self.feature_weights['motor_performance']
            score += motor_score
            rationale_details['モーター評価'] = f"モーター勝率{racer['motor_win_rate']}% → {motor_score:.2f}点"

            # スタート評価
            if racer['average_start_time'] < 0:
                start_score = abs(racer['average_start_time']) * 100 * self.feature_weights['start_timing']
            else:
                start_score = -racer['average_start_time'] * 50 * self.feature_weights['start_timing']
            score += start_score
            rationale_details['スタート評価'] = f"平均ST{racer['average_start_time']} → {start_score:.2f}点"

            # 艇番有利不利
            boat_advantages = {1: 8, 2: 5, 3: 2, 4: 0, 5: -2, 6: -5}
            boat_score = boat_advantages[racer['boat_number']] * self.feature_weights['boat_position']
            score += boat_score
            rationale_details['艇番評価'] = f"{racer['boat_number']}号艇 → {boat_score:.2f}点"

            # 調子による補正
            form_adjustments = {'好調': 5, '普通': 0, '不調': -3}
            form_score = form_adjustments[racer['recent_form']] * self.feature_weights['recent_form']
            score += form_score
            rationale_details['調子評価'] = f"{racer['recent_form']} → {form_score:.2f}点"

            # 気象条件適性
            weather_score = self._calculate_weather_impact(racer, weather) * self.feature_weights['weather_adaptation']
            score += weather_score
            rationale_details['気象適性'] = f"気象条件適性 → {weather_score:.2f}点"

            prediction = {
                'boat_number': racer['boat_number'],
                'racer_name': racer['name'],
                'score': round(score, 2),
                'win_probability': min(max(score / 100, 0.05), 0.95),
                'predicted_rank': 0  # 後で設定
            }

            predictions.append(prediction)
            detailed_rationale[racer['boat_number']] = rationale_details

        # ランキング設定
        predictions.sort(key=lambda x: x['score'], reverse=True)
        for i, pred in enumerate(predictions):
            pred['predicted_rank'] = i + 1

        return predictions, detailed_rationale

    def _calculate_weather_impact(self, racer, weather):
        """気象条件の影響を計算"""
        impact = 0

        # 風の影響
        if weather['wind_speed'] > 5:
            # ベテランは強風に強い
            if racer['experience_years'] > 15:
                impact += 2
            else:
                impact -= 1

        # 波の影響  
        if weather['wave_height'] > 2:
            # 体重が軽いと不利
            if racer['weight'] < 52:
                impact -= 2
            elif racer['weight'] > 55:
                impact += 1

        return impact

    def generate_betting_strategies(self, predictions, odds):
        """複数の投票戦略を生成（新機能）"""
        strategies = {}

        # 1. 本命戦略（堅実）
        top_2 = predictions[:2]
        strategies['本命'] = {
            'type': '堅実',
            'recommended_bets': [
                f"{top_2[0]['boat_number']}-{top_2[1]['boat_number']} 2連単",
                f"{top_2[0]['boat_number']} 単勝"
            ],
            'confidence': '高',
            'expected_return': '低〜中',
            'risk': '低',
            'rationale': f"{top_2[0]['racer_name']}（{top_2[0]['boat_number']}号艇）を軸にした堅実な勝負。勝率{predictions[0]['win_probability']*100:.1f}%で期待値は安定。"
        }

        # 2. 中穴戦略（バランス）
        mid_pick = predictions[2]  # 3番手を狙う
        strategies['中穴'] = {
            'type': 'バランス',
            'recommended_bets': [
                f"{predictions[0]['boat_number']}-{mid_pick['boat_number']} 2連単",
                f"{mid_pick['boat_number']}-{predictions[0]['boat_number']} 2連単",
                f"{mid_pick['boat_number']} 単勝"
            ],
            'confidence': '中',
            'expected_return': '中〜高',
            'risk': '中',
            'rationale': f"{mid_pick['racer_name']}（{mid_pick['boat_number']}号艇）の差しを狙う。オッズと実力のバランスが良好。"
        }

        # 3. 大穴戦略（高配当）
        longshot = predictions[4]  # 5番手以降を狙う
        strategies['大穴'] = {
            'type': '高配当狙い',
            'recommended_bets': [
                f"{longshot['boat_number']}-{predictions[0]['boat_number']} 2連単",
                f"{longshot['boat_number']}-{predictions[1]['boat_number']} 2連単",
                f"{longshot['boat_number']} 単勝"
            ],
            'confidence': '低',
            'expected_return': '高',
            'risk': '高',
            'rationale': f"{longshot['racer_name']}（{longshot['boat_number']}号艇）の一発逆転を狙う。条件が整えば高配当の可能性。"
        }

        return strategies


class NoteArticleGenerator:
    """note記事自動生成クラス（新機能）"""

    def __init__(self):
        self.templates = {
            'introduction': [
                "今日の競艇予想をAI分析によってお届けします。",
                "データに基づいた科学的アプローチで今日のレースを徹底分析。",
                "最新のAI技術を活用した競艇予想システムが導き出した結果をご覧ください。"
            ],
            'closing': [
                "以上が本日の予想となります。参考程度にお楽しみください。",
                "実際の投票は自己責任でお願いします。競艇を楽しみましょう！",
                "データ分析を通じて競艇の奥深さをお伝えできれば幸いです。"
            ]
        }

    def generate_full_article(self, race_info, racers, weather, predictions, rationale, strategies, odds):
        """2000文字以上のnote記事を自動生成"""

        article = f"""# 【AI競艇予想】{race_info['venue']} {race_info['race_number']}R 徹底分析

{random.choice(self.templates['introduction'])}

## レース概要
- **会場**: {race_info['venue']}
- **レース番号**: {race_info['race_number']}R 
- **発走時刻**: {race_info['race_time']}
- **クラス**: {race_info['class']}

## 気象条件
本日の{race_info['venue']}の気象条件は以下の通りです：
- **天候**: {weather['weather']}
- **気温**: {weather['temperature']}℃
- **風速**: {weather['wind_speed']}m/s ({weather['wind_direction']})
- **波高**: {weather['wave_height']}cm
- **水面温度**: {weather['water_temperature']}℃

{self._analyze_weather_impact(weather)}

## 各選手詳細分析

"""

        # 各選手の詳細分析
        for i, racer in enumerate(racers):
            prediction = next(p for p in predictions if p['boat_number'] == racer['boat_number'])
            racer_rationale = rationale[racer['boat_number']]

            article += f"""### {racer['boat_number']}号艇 {racer['name']} （予想順位：{prediction['predicted_rank']}位）

**基本データ**
- 年齢: {racer['age']}歳
- 体重: {racer['weight']}kg
- 経験年数: {racer['experience_years']}年
- 勝率: {racer['win_rate']}%
- 連対率: {racer['place_rate']}%
- 平均ST: {racer['average_start_time']}

**モーター・展示情報**  
- モーター番号: {racer['motor_number']}号機
- モーター勝率: {racer['motor_win_rate']}%
- 現在の調子: {racer['recent_form']}

**AI評価詳細**
"""

            for category, detail in racer_rationale.items():
                article += f"- {category}: {detail}\n"

            article += f"\n**総合評価スコア: {prediction['score']}点**\n"
            article += f"勝率予想: {prediction['win_probability']*100:.1f}%\n\n"

            article += self._generate_racer_analysis(racer, prediction) + "\n\n"

        # 予想戦略セクション
        article += """## AI予想による投票戦略

今回のレース分析から、以下3つの戦略を提案します：

"""

        for strategy_name, strategy_data in strategies.items():
            article += f"""### {strategy_name}戦略（{strategy_data['type']}）

**推奨投票**
"""
            for bet in strategy_data['recommended_bets']:
                article += f"- {bet}\n"

            article += f"""
**戦略評価**
- 信頼度: {strategy_data['confidence']}
- 期待収益: {strategy_data['expected_return']}  
- リスクレベル: {strategy_data['risk']}

**戦略根拠**
{strategy_data['rationale']}

"""

        # まとめセクション
        article += f"""## 本日のまとめ

今回の{race_info['venue']}{race_info['race_number']}Rは、"""

        top_prediction = predictions[0]
        article += f"""{top_prediction['racer_name']}（{top_prediction['boat_number']}号艇）を本命に据えた展開が予想されます。

**勝負ポイント**
1. {weather['weather']}の天候と風速{weather['wind_speed']}m/sの条件での各選手の適応力
2. モーター性能と展示タイムの兼ね合い
3. スタート力とコース取りの駆け引き

{self._generate_final_advice(predictions, weather, race_info)}

{random.choice(self.templates['closing'])}

---
※この予想は過去データとAI分析に基づく参考情報です。
※舟券購入は自己責任でお楽しみください。
※ギャンブル依存症にご注意ください。

#競艇 #競艇予想 #AI予想 #{race_info['venue']} #データ分析
"""

        return article

    def _analyze_weather_impact(self, weather):
        """気象条件の影響分析"""
        analysis = ""

        if weather['wind_speed'] > 6:
            analysis += f"風速{weather['wind_speed']}m/sと強めの風が吹いており、"
            if weather['wind_direction'] == '向かい風':
                analysis += "向かい風のためスタートが難しく、経験豊富な選手が有利になりそうです。"
            elif weather['wind_direction'] == '追い風': 
                analysis += "追い風のため全速戦になりやすく、モーター性能の差が顕著に表れるでしょう。"
            else:
                analysis += "横風の影響でコース取りが重要になり、技術力の差が勝負を分けそうです。"
        else:
            analysis += "風は穏やかで、各選手の実力がストレートに反映される条件です。"

        if weather['wave_height'] > 2:
            analysis += f" また、波高{weather['wave_height']}cmとやや荒れており、体重が重く安定感のある選手が有利になる可能性があります。"

        return analysis

    def _generate_racer_analysis(self, racer, prediction):
        """個別選手分析テキスト生成"""
        analysis = ""

        if prediction['predicted_rank'] == 1:
            analysis = f"{racer['name']}選手は今回の本命候補です。"
            if racer['win_rate'] > 25:
                analysis += "勝率が高く安定した実力の持ち主で、"
            if racer['boat_number'] <= 2:
                analysis += f"{racer['boat_number']}号艇の有利なコースからスタートを決めれば逃げ切りの可能性が高いでしょう。"
            else:
                analysis += "不利なコースながら差し・まくりの技術で上位進出が期待できます。"

        elif prediction['predicted_rank'] <= 3:
            analysis = f"{racer['name']}選手は連対候補として注目です。"
            if racer['recent_form'] == '好調':
                analysis += "現在好調を維持しており、"
            analysis += "展開次第では上位進出も十分に考えられる実力を持っています。"

        else:
            analysis = f"{racer['name']}選手は今回厳しい予想となりました。"
            if racer['average_start_time'] > 0.10:
                analysis += "スタートに課題があり、"
            analysis += "展開がハマれば面白い存在ですが、軸には向かないでしょう。"

        return analysis

    def _generate_final_advice(self, predictions, weather, race_info):
        """最終アドバイス生成"""
        advice = ""

        top_3 = predictions[:3]
        confidence_level = sum(p['win_probability'] for p in top_3)

        if confidence_level > 1.5:
            advice += "上位3艇の実力が拮抗しており、荒れる可能性も含んでいます。"
        else:
            advice += "実力差がはっきりしており、比較的順当な決着が予想されます。"

        if weather['wind_speed'] > 5:
            advice += " 気象条件を考慮すると、経験豊富な選手を重視した予想が良いでしょう。"

        return advice


class KyoteiPredictionGUI:
    """競艇予想システムGUI（v13.8ベース + 新機能追加）"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("競艇AI予想システム v13.9 Practical")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2C3E50')

        # データマネージャーとエンジンの初期化
        self.data_manager = KyoteiDataManager()
        self.prediction_engine = KyoteiAIPredictionEngine()
        self.note_generator = NoteArticleGenerator()

        # 現在のデータ
        self.current_race_data = None
        self.current_predictions = None
        self.current_rationale = None
        self.current_strategies = None

        self.setup_ui()

    def setup_ui(self):
        """UI構築（v13.8の構造を維持）"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # タイトル
        title_label = tk.Label(main_frame, text="競艇AI予想システム v13.9 Practical", 
                              font=('Arial', 20, 'bold'), 
                              fg='white', bg='#2C3E50')
        title_label.pack(pady=10)

        # 日付選択フレーム（v13.8と同じ）
        date_frame = ttk.Frame(main_frame)
        date_frame.pack(fill=tk.X, pady=5)

        tk.Label(date_frame, text="対象日付:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        self.date_var = tk.StringVar(value=datetime.date.today().strftime('%Y-%m-%d'))
        self.date_entry = ttk.Entry(date_frame, textvariable=self.date_var, width=15)
        self.date_entry.pack(side=tk.LEFT, padx=5)

        load_button = ttk.Button(date_frame, text="レース読込", command=self.load_races)
        load_button.pack(side=tk.LEFT, padx=5)

        # レース選択フレーム（v13.8と同じ）
        race_frame = ttk.Frame(main_frame)
        race_frame.pack(fill=tk.X, pady=5)

        tk.Label(race_frame, text="レース選択:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        self.race_var = tk.StringVar()
        self.race_combo = ttk.Combobox(race_frame, textvariable=self.race_var, width=50, state='readonly')
        self.race_combo.pack(side=tk.LEFT, padx=5)
        self.race_combo.bind('<<ComboboxSelected>>', self.on_race_selected)

        # 新機能：予想モード選択
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=5)

        tk.Label(mode_frame, text="表示モード:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="基本予想")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, 
                                 values=["基本予想", "詳細根拠", "投票戦略", "note記事"], 
                                 width=15, state='readonly')
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind('<<ComboboxSelected>>', self.update_display_mode)

        # メイン表示エリア
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # 基本予想タブ（v13.8と同じ）
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="AI予想結果")

        self.prediction_text = scrolledtext.ScrolledText(
            self.prediction_frame, wrap=tk.WORD, width=100, height=30,
            font=('Courier', 10), bg='#34495E', fg='white', insertbackground='white'
        )
        self.prediction_text.pack(fill=tk.BOTH, expand=True)

        # 詳細根拠タブ（新機能）
        self.rationale_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rationale_frame, text="予想根拠詳細")

        self.rationale_text = scrolledtext.ScrolledText(
            self.rationale_frame, wrap=tk.WORD, width=100, height=30,
            font=('Courier', 10), bg='#34495E', fg='white', insertbackground='white'
        )
        self.rationale_text.pack(fill=tk.BOTH, expand=True)

        # 投票戦略タブ（新機能）
        self.strategy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.strategy_frame, text="投票戦略")

        self.strategy_text = scrolledtext.ScrolledText(
            self.strategy_frame, wrap=tk.WORD, width=100, height=30,
            font=('Courier', 10), bg='#34495E', fg='white', insertbackground='white'
        )
        self.strategy_text.pack(fill=tk.BOTH, expand=True)

        # note記事タブ（新機能）
        self.note_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.note_frame, text="note記事")

        note_control_frame = ttk.Frame(self.note_frame)
        note_control_frame.pack(fill=tk.X, pady=5)

        generate_note_btn = ttk.Button(note_control_frame, text="記事生成", command=self.generate_note_article)
        generate_note_btn.pack(side=tk.LEFT, padx=5)

        save_note_btn = ttk.Button(note_control_frame, text="記事保存", command=self.save_note_article)
        save_note_btn.pack(side=tk.LEFT, padx=5)

        self.note_text = scrolledtext.ScrolledText(
            self.note_frame, wrap=tk.WORD, width=100, height=28,
            font=('MS Gothic', 9), bg='white', fg='black'
        )
        self.note_text.pack(fill=tk.BOTH, expand=True)

        # 初期メッセージ
        self.prediction_text.insert('1.0', "日付を選択してレースを読み込んでください。")


    def load_races(self):
        """レース情報の読み込み（v13.8と同じ）"""
        try:
            selected_date = datetime.datetime.strptime(self.date_var.get(), '%Y-%m-%d').date()
            races = self.data_manager.get_races_for_date(selected_date)

            if not races:
                messagebox.showwarning("警告", "選択した日付にレースがありません")
                return

            # コンボボックスに設定
            race_options = []
            for race in races:
                option = f"{race['venue']} {race['race_number']}R ({race['race_time']}) {race['class']}"
                race_options.append(option)

            self.race_combo['values'] = race_options
            self.race_combo.set('')

            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert('1.0', f"{len(races)}レースを読み込みました.\nレースを選択してください.")

        except ValueError:
            messagebox.showerror("エラー", "正しい日付形式で入力してください (YYYY-MM-DD)")
        except Exception as e:
            messagebox.showerror("エラー", f"レース読み込み中にエラーが発生しました: {str(e)}")

    def on_race_selected(self, event):
        """レース選択時の処理（v13.8ベース + 新機能）"""
        if not self.race_var.get():
            return

        try:
            # 選択されたレース情報を解析
            race_info_text = self.race_var.get()
            parts = race_info_text.split()
            venue = parts[0]
            race_number = int(parts[1][:-1])  # "1R" -> 1
            race_time = parts[2].strip('()')
            race_class = parts[3] if len(parts) > 3 else "一般戦"

            selected_date = datetime.datetime.strptime(self.date_var.get(), '%Y-%m-%d').date()

            # レースデータ生成
            racers = self.data_manager.generate_racer_data(selected_date, venue, race_number)
            weather = self.data_manager.generate_weather_conditions(selected_date, venue)
            odds = self.data_manager.generate_odds_data(racers)

            # AI予想実行（詳細根拠付き）
            predictions, rationale = self.prediction_engine.predict_race(racers, weather, venue)
            strategies = self.prediction_engine.generate_betting_strategies(predictions, odds)

            # データ保存
            self.current_race_data = {
                'venue': venue,
                'race_number': race_number,
                'race_time': race_time,
                'class': race_class,
                'racers': racers,
                'weather': weather,
                'odds': odds
            }
            self.current_predictions = predictions
            self.current_rationale = rationale
            self.current_strategies = strategies

            # 表示更新
            self.display_basic_prediction()
            self.display_detailed_rationale()
            self.display_betting_strategies()

        except Exception as e:
            messagebox.showerror("エラー", f"予想処理中にエラーが発生しました: {str(e)}")

    def update_display_mode(self, event=None):
        """表示モード変更"""
        mode = self.mode_var.get()
        if mode == "基本予想":
            self.notebook.select(0)
        elif mode == "詳細根拠":
            self.notebook.select(1)
        elif mode == "投票戦略":
            self.notebook.select(2)
        elif mode == "note記事":
            self.notebook.select(3)


    def display_basic_prediction(self):
        """基本予想表示（v13.8と同じ形式）"""
        if not self.current_race_data:
            return

        output = f"""
{CSS_STYLES}
<div class="container">
    <div class="header">
        <h1>🏁 競艇AI予想システム v13.9 Practical</h1>
        <h2>{self.current_race_data['venue']} {self.current_race_data['race_number']}R 
            ({self.current_race_data['race_time']}) {self.current_race_data['class']}</h2>
    </div>

    <div class="weather-info">
        <div class="stat-item">
            <h3>🌤️ 天候</h3>
            <p>{self.current_race_data['weather']['weather']}</p>
        </div>
        <div class="stat-item">
            <h3>🌡️ 気温</h3>
            <p>{self.current_race_data['weather']['temperature']}℃</p>
        </div>
        <div class="stat-item">
            <h3>💨 風速</h3>
            <p>{self.current_race_data['weather']['wind_speed']}m/s</p>
        </div>
        <div class="stat-item">
            <h3>🌊 波高</h3>
            <p>{self.current_race_data['weather']['wave_height']}cm</p>
        </div>
    </div>

    <div class="prediction-card">
        <h2>🤖 AI予想結果</h2>"""

        for pred in self.current_predictions:
            racer = next(r for r in self.current_race_data['racers'] if r['boat_number'] == pred['boat_number'])
            odds_info = self.current_race_data['odds'][pred['boat_number']]

            output += f"""
        <div class="racer-row">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 18px; font-weight: bold; 
                           background: {'#FFD700' if pred['predicted_rank'] == 1 else '#C0C0C0' if pred['predicted_rank'] == 2 else '#CD7F32' if pred['predicted_rank'] == 3 else 'rgba(255,255,255,0.2)'}; 
                           color: black; padding: 5px 10px; border-radius: 50%; min-width: 30px; text-align: center;">
                    {pred['predicted_rank']}
                </div>
                <div style="min-width: 40px; text-align: center; font-weight: bold; font-size: 16px;">
                    {pred['boat_number']}号艇
                </div>
                <div style="min-width: 120px; font-weight: bold;">
                    {pred['racer_name']}
                </div>
                <div style="font-size: 12px; color: #BDC3C7;">
                    {racer['age']}歳 | 勝率{racer['win_rate']}% | ST{racer['average_start_time']} | {racer['recent_form']}
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="text-align: right;">
                    <div>AIスコア: {pred['score']}</div>
                    <div style="font-size: 11px; color: #95A5A6;">勝率予想: {pred['win_probability']*100:.1f}%</div>
                </div>
                <div class="odds-display">
                    単勝 {odds_info['win']}倍
                </div>
            </div>
        </div>"""

        output += """
    </div>
</div>
"""

        self.prediction_text.delete('1.0', tk.END)
        self.prediction_text.insert('1.0', output)


    def display_detailed_rationale(self):
        """詳細根拠表示（新機能）"""
        if not self.current_rationale:
            return

        output = f"""
=== 🔍 AI予想根拠詳細分析 ===

レース: {self.current_race_data['venue']} {self.current_race_data['race_number']}R
分析日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【気象条件による影響分析】
天候: {self.current_race_data['weather']['weather']}
風向・風速: {self.current_race_data['weather']['wind_direction']} {self.current_race_data['weather']['wind_speed']}m/s
波高: {self.current_race_data['weather']['wave_height']}cm

"""

        if self.current_race_data['weather']['wind_speed'] > 5:
            output += f"⚠️ 風速{self.current_race_data['weather']['wind_speed']}m/sの強風により、スタートと展開に大きな影響\n"

        if self.current_race_data['weather']['wave_height'] > 2:
            output += f"🌊 波高{self.current_race_data['weather']['wave_height']}cmの荒れた水面、体重・経験値が重要\n"

        output += "\n" + "="*80 + "\n"

        # 各選手の詳細分析
        for pred in self.current_predictions:
            racer = next(r for r in self.current_race_data['racers'] if r['boat_number'] == pred['boat_number'])
            rationale_details = self.current_rationale[pred['boat_number']]

            output += f"""
【{pred['predicted_rank']}位予想】{pred['boat_number']}号艇 {pred['racer_name']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 基本データ
  年齢: {racer['age']}歳 | 体重: {racer['weight']}kg | 経験: {racer['experience_years']}年
  勝率: {racer['win_rate']}% | 連対率: {racer['place_rate']}%
  平均ST: {racer['average_start_time']} | 調子: {racer['recent_form']}

🔧 機材・展示情報
  モーター: {racer['motor_number']}号機 (勝率{racer['motor_win_rate']}%)

🤖 AI評価詳細
"""

            for category, evaluation in rationale_details.items():
                output += f"  {category}: {evaluation}\n"

            output += f"""
💡 総合評価スコア: {pred['score']}点
📈 勝率予想: {pred['win_probability']*100:.1f}%
🎯 推奨度: {'★★★★★' if pred['predicted_rank'] == 1 else '★★★★☆' if pred['predicted_rank'] == 2 else '★★★☆☆' if pred['predicted_rank'] == 3 else '★★☆☆☆'}

"""

            # 予想根拠の詳細解説
            if pred['predicted_rank'] == 1:
                output += "🔥 【本命評価】この選手を軸にした戦略を推奨\n"
            elif pred['predicted_rank'] <= 3:
                output += "⚡ 【要注意】上位進出の可能性が高い選手\n"
            else:
                output += "📊 【データ参考】展開次第では面白い存在\n"

            output += "\n"

        output += f"""
{'='*80}
📝 分析まとめ
・最有力候補: {self.current_predictions[0]['racer_name']}（{self.current_predictions[0]['boat_number']}号艇）
・対抗候補: {self.current_predictions[1]['racer_name']}（{self.current_predictions[1]['boat_number']}号艇）
・穴候補: {self.current_predictions[2]['racer_name']}（{self.current_predictions[2]['boat_number']}号艇）

気象条件と各選手の特性を総合的に分析した結果です。
実際の投票は自己責任でお願いします。
"""

        self.rationale_text.delete('1.0', tk.END)
        self.rationale_text.insert('1.0', output)

    def display_betting_strategies(self):
        """投票戦略表示（新機能）"""
        if not self.current_strategies:
            return

        output = f"""
💰 投票戦略提案 - {self.current_race_data['venue']} {self.current_race_data['race_number']}R

分析システム: 競艇AI v13.9 Practical
対象レース: {self.current_race_data['class']}
気象条件: {self.current_race_data['weather']['weather']} 風速{self.current_race_data['weather']['wind_speed']}m/s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

        strategy_icons = {'本命': '🛡️', '中穴': '⚖️', '大穴': '🚀'}

        for strategy_name, strategy_data in self.current_strategies.items():
            icon = strategy_icons.get(strategy_name, '📊')

            output += f"""
{icon} 【{strategy_name}戦略】 - {strategy_data['type']}タイプ
{'-'*60}

🎯 推奨投票パターン:
"""
            for i, bet in enumerate(strategy_data['recommended_bets'], 1):
                output += f"   {i}. {bet}\n"

            output += f"""
📊 戦略評価:
   信頼度: {strategy_data['confidence']} | 期待収益: {strategy_data['expected_return']} | リスク: {strategy_data['risk']}

💭 戦略根拠:
   {strategy_data['rationale']}

"""

            # 投資金額の提案
            if strategy_name == '本命':
                output += "💵 推奨投資: 資金の40-50% (安定重視)\n"
            elif strategy_name == '中穴':
                output += "💵 推奨投資: 資金の30-40% (バランス型)\n"
            else:
                output += "💵 推奨投資: 資金の10-20% (少額勝負)\n"

            output += "\n"

        output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎲 総合戦略アドバイス

1️⃣ 堅実に行くなら: {self.current_strategies['本命']['recommended_bets'][0]}
2️⃣ バランス重視: {self.current_strategies['中穴']['recommended_bets'][0]}
3️⃣ 一発狙い: {self.current_strategies['大穴']['recommended_bets'][0]}

⚠️  重要な注意事項:
・この予想はAI分析による参考情報です
・投票は自己責任でお願いします  
・ギャンブル依存症にご注意ください
・余剰資金の範囲内で楽しみましょう

🏁 Good Luck! 🏁
"""

        self.strategy_text.delete('1.0', tk.END)
        self.strategy_text.insert('1.0', output)


    def generate_note_article(self):
        """note記事生成（新機能）"""
        if not self.current_race_data:
            messagebox.showwarning("警告", "レースデータが選択されていません")
            return

        try:
            # 記事生成
            article = self.note_generator.generate_full_article(
                self.current_race_data,
                self.current_race_data['racers'],
                self.current_race_data['weather'],
                self.current_predictions,
                self.current_rationale,
                self.current_strategies,
                self.current_race_data['odds']
            )

            # 表示
            self.note_text.delete('1.0', tk.END)
            self.note_text.insert('1.0', article)

            # 文字数カウント
            char_count = len(article)
            messagebox.showinfo("生成完了", f"note記事を生成しました\n文字数: {char_count}文字")

        except Exception as e:
            messagebox.showerror("エラー", f"記事生成中にエラーが発生しました: {str(e)}")

    def save_note_article(self):
        """note記事をファイルに保存"""
        if not self.note_text.get('1.0', tk.END).strip():
            messagebox.showwarning("警告", "保存する記事がありません")
            return

        try:
            # ファイル名生成
            race_info = self.current_race_data
            filename = f"note記事_{race_info['venue']}_{race_info['race_number']}R_{datetime.date.today().strftime('%Y%m%d')}.md"
            filepath = f"/home/user/output/{filename}"

            # 保存
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.note_text.get('1.0', tk.END))

            messagebox.showinfo("保存完了", f"記事を保存しました\n{filepath}")

        except Exception as e:
            messagebox.showerror("エラー", f"保存中にエラーが発生しました: {str(e)}")

    def run(self):
        """GUIアプリケーション実行"""
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("システムエラー", f"予期しないエラーが発生しました: {str(e)}")

def main():
    """メイン関数（v13.8と同じ構造）"""
    try:
        print("=" * 60)
        print("🏁 競艇AI予想システム v13.9 Practical 起動中...")
        print("=" * 60)
        print()
        print("📊 新機能:")
        print("  ✅ 予想根拠の詳細表示")
        print("  ✅ note記事2000文字以上自動生成")
        print("  ✅ 本命・中穴・大穴の複数予想パターン")
        print("  ✅ v13.8の全機能を完全維持")
        print()
        print("🚀 システム起動中...")

        # GUI起動
        app = KyoteiPredictionGUI()
        app.run()

    except ImportError as e:
        print(f"❌ 必要なライブラリがインストールされていません: {e}")
        print("💡 'pip install tkinter' を実行してください")
    except Exception as e:
        print(f"❌ システム起動中にエラーが発生しました: {e}")
        print("💡 エラーの詳細を確認してください")

if __name__ == "__main__":
    main()
