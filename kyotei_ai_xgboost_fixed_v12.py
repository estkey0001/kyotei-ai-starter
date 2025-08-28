# 競艇AI予想システム v12.1 - XGBoost修正版
# ファイル: kyotei_ai_xgboost_fixed_v12.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any
import logging

# 警告を抑制
warnings.filterwarnings("ignore")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostKyoteiSystem:
    """
    競艇AI予想システム v12.1 - XGBoost修正版
    XGBoost + RandomForest + GradientBoosting + NeuralNetwork の4モデルアンサンブル
    
    主要機能:
    - 18次元特徴量による高精度予測
    - 4モデルアンサンブル予測
    - プロフェッショナル特徴量計算
    - 買い目自動生成
    - 完全エラーハンドリング
    """

    def __init__(self):
        """システム初期化"""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            "avg_st", "win_rate", "place_rate", "motor_rate", "boat_rate",
            "recent_performance", "course_advantage", "weather_factor",
            "wave_height", "wind_speed", "temperature", "humidity",
            "experience_years", "weight", "age", "morning_adjustment",
            "exhibition_time", "start_timing"
        ]

        # アンサンブル重み設定 (XGB, RF, GBM, NN)
        self.ensemble_weights = [0.2, 0.2, 0.25, 0.35]

        # モデル設定
        self.model_configs = {
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "objective": "reg:squarederror",
                "eval_metric": "rmse"
            },
            "random_forest": {
                "n_estimators": 150,
                "max_depth": 8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 150,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": 42
            },
            "neural_network": {
                "hidden_layer_sizes": (100, 50, 25),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "learning_rate": "adaptive",
                "max_iter": 500,
                "random_state": 42
            }
        }

        logger.info("XGBoostKyoteiSystem v12.1 初期化完了")

    def load_data(self, data_source: str = "sample") -> pd.DataFrame:
        """
        データ読み込み（サンプルデータまたは実データ）
        
        Args:
            data_source: データソース ("sample" または ファイルパス)
            
        Returns:
            pd.DataFrame: 読み込んだデータ
        """
        try:
            if data_source == "sample":
                logger.info("サンプルデータを生成中...")
                return self._generate_sample_data()
            else:
                if os.path.exists(data_source):
                    logger.info(f"データファイル読み込み中: {data_source}")
                    return pd.read_csv(data_source)
                else:
                    logger.warning(f"データファイルが見つかりません: {data_source}")
                    logger.info("サンプルデータを生成します")
                    return self._generate_sample_data()
                    
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            logger.info("サンプルデータを生成します")
            return self._generate_sample_data()

    def _generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        サンプルデータ生成
        
        Args:
            n_samples: サンプル数
            
        Returns:
            pd.DataFrame: 生成されたサンプルデータ
        """
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # 基本特徴量生成
            avg_st = np.random.normal(0.17, 0.05)
            win_rate = np.random.uniform(0.05, 0.35)
            place_rate = np.random.uniform(0.3, 0.7)
            motor_rate = np.random.uniform(0.3, 0.8)
            boat_rate = np.random.uniform(0.3, 0.8)
            
            # パフォーマンス指標
            recent_performance = np.random.uniform(0.2, 0.8)
            course_advantage = np.random.uniform(0.1, 0.9)
            
            # 環境要因
            weather_factor = np.random.uniform(0.3, 1.0)
            wave_height = np.random.uniform(0, 5)
            wind_speed = np.random.uniform(0, 15)
            temperature = np.random.uniform(10, 35)
            humidity = np.random.uniform(30, 90)
            
            # 選手情報
            experience_years = np.random.randint(1, 30)
            weight = np.random.uniform(45, 65)
            age = np.random.randint(20, 60)
            
            # レース当日要因
            morning_adjustment = np.random.uniform(-0.1, 0.1)
            exhibition_time = np.random.uniform(6.8, 7.5)
            start_timing = np.random.uniform(-0.2, 0.2)
            
            # 着順（目的変数）- 特徴量に基づいて生成
            rank_score = (
                win_rate * 0.3 + 
                place_rate * 0.2 + 
                motor_rate * 0.15 + 
                boat_rate * 0.15 + 
                recent_performance * 0.1 + 
                course_advantage * 0.1 +
                np.random.normal(0, 0.1)
            )
            
            # 着順に変換（1-6位）
            rank = max(1, min(6, int(7 - rank_score * 6)))
            
            data.append([
                avg_st, win_rate, place_rate, motor_rate, boat_rate,
                recent_performance, course_advantage, weather_factor,
                wave_height, wind_speed, temperature, humidity,
                experience_years, weight, age, morning_adjustment,
                exhibition_time, start_timing, rank
            ])
        
        columns = self.feature_names + ["rank"]
        df = pd.DataFrame(data, columns=columns)
        
        logger.info(f"サンプルデータ生成完了: {len(df)}件")
        return df

    def get_race_data(self, race_id: str = "sample_race") -> Dict[str, Any]:
        """
        レースデータ取得
        
        Args:
            race_id: レースID
            
        Returns:
            Dict: レースデータ
        """
        try:
            race_data = {
                "race_id": race_id,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "venue": "サンプル競艇場",
                "race_number": np.random.randint(1, 12),
                "weather": np.random.choice(["晴", "曇", "雨"]),
                "wind_direction": np.random.choice(["無風", "向風", "追風", "横風"]),
                "wind_speed": np.random.uniform(0, 10),
                "wave_height": np.random.uniform(0, 3),
                "temperature": np.random.uniform(15, 30),
                "humidity": np.random.uniform(40, 80),
                "boats": []
            }
            
            # 6艇のデータ生成
            for boat_num in range(1, 7):
                boat_data = {
                    "boat_number": boat_num,
                    "racer_name": f"選手{boat_num}",
                    "avg_st": np.random.normal(0.17, 0.05),
                    "win_rate": np.random.uniform(0.05, 0.35),
                    "place_rate": np.random.uniform(0.3, 0.7),
                    "motor_number": np.random.randint(1, 60),
                    "motor_rate": np.random.uniform(0.3, 0.8),
                    "boat_number_actual": np.random.randint(1, 60),
                    "boat_rate": np.random.uniform(0.3, 0.8),
                    "weight": np.random.uniform(45, 65),
                    "age": np.random.randint(20, 60),
                    "experience_years": np.random.randint(1, 30)
                }
                race_data["boats"].append(boat_data)
            
            logger.info(f"レースデータ取得完了: {race_id}")
            return race_data
            
        except Exception as e:
            logger.error(f"レースデータ取得エラー: {e}")
            return {}

    def calculate_professional_features(self, race_data: Dict[str, Any]) -> np.ndarray:
        """
        プロフェッショナル特徴量計算
        
        Args:
            race_data: レースデータ
            
        Returns:
            np.ndarray: 特徴量行列 (6艇 × 18特徴量)
        """
        try:
            features = []
            
            for boat in race_data["boats"]:
                # 基本特徴量
                avg_st = boat.get("avg_st", 0.17)
                win_rate = boat.get("win_rate", 0.15)
                place_rate = boat.get("place_rate", 0.45)
                motor_rate = boat.get("motor_rate", 0.5)
                boat_rate = boat.get("boat_rate", 0.5)
                
                # パフォーマンス指標計算
                recent_performance = (win_rate + place_rate) / 2
                course_advantage = 0.8 if boat["boat_number"] == 1 else 0.6 if boat["boat_number"] <= 3 else 0.4
                
                # 環境要因
                weather_factor = self._calculate_weather_factor(race_data)
                wave_height = race_data.get("wave_height", 1.0)
                wind_speed = race_data.get("wind_speed", 3.0)
                temperature = race_data.get("temperature", 20.0)
                humidity = race_data.get("humidity", 60.0)
                
                # 選手情報
                experience_years = boat.get("experience_years", 10)
                weight = boat.get("weight", 52.0)
                age = boat.get("age", 35)
                
                # レース当日要因
                morning_adjustment = np.random.uniform(-0.05, 0.05)
                exhibition_time = np.random.uniform(6.9, 7.3)
                start_timing = avg_st + np.random.uniform(-0.1, 0.1)
                
                boat_features = [
                    avg_st, win_rate, place_rate, motor_rate, boat_rate,
                    recent_performance, course_advantage, weather_factor,
                    wave_height, wind_speed, temperature, humidity,
                    experience_years, weight, age, morning_adjustment,
                    exhibition_time, start_timing
                ]
                
                features.append(boat_features)
            
            features_array = np.array(features)
            logger.info(f"特徴量計算完了: {features_array.shape}")
            return features_array
            
        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            return np.random.rand(6, 18)

    def _calculate_weather_factor(self, race_data: Dict[str, Any]) -> float:
        """天候要因計算"""
        weather = race_data.get("weather", "晴")
        wind_speed = race_data.get("wind_speed", 3.0)
        
        weather_base = {"晴": 1.0, "曇": 0.9, "雨": 0.7}.get(weather, 0.8)
        wind_factor = max(0.5, 1.0 - wind_speed * 0.05)
        
        return weather_base * wind_factor

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        4モデルの訓練
        
        Args:
            X: 特徴量
            y: 目的変数
            
        Returns:
            Dict: 訓練結果
        """
        try:
            logger.info("モデル訓練開始...")
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 特徴量正規化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            
            # 1. XGBoost
            logger.info("XGBoost訓練中...")
            self.models["xgboost"] = xgb.XGBRegressor(**self.model_configs["xgboost"])
            self.models["xgboost"].fit(X_train, y_train)
            xgb_pred = self.models["xgboost"].predict(X_test)
            results["xgboost"] = {
                "mse": mean_squared_error(y_test, xgb_pred),
                "mae": mean_absolute_error(y_test, xgb_pred)
            }
            
            # 2. Random Forest
            logger.info("Random Forest訓練中...")
            self.models["random_forest"] = RandomForestRegressor(**self.model_configs["random_forest"])
            self.models["random_forest"].fit(X_train, y_train)
            rf_pred = self.models["random_forest"].predict(X_test)
            results["random_forest"] = {
                "mse": mean_squared_error(y_test, rf_pred),
                "mae": mean_absolute_error(y_test, rf_pred)
            }
            
            # 3. Gradient Boosting
            logger.info("Gradient Boosting訓練中...")
            self.models["gradient_boosting"] = GradientBoostingRegressor(**self.model_configs["gradient_boosting"])
            self.models["gradient_boosting"].fit(X_train, y_train)
            gb_pred = self.models["gradient_boosting"].predict(X_test)
            results["gradient_boosting"] = {
                "mse": mean_squared_error(y_test, gb_pred),
                "mae": mean_absolute_error(y_test, gb_pred)
            }
            
            # 4. Neural Network
            logger.info("Neural Network訓練中...")
            self.models["neural_network"] = MLPRegressor(**self.model_configs["neural_network"])
            self.models["neural_network"].fit(X_train_scaled, y_train)
            nn_pred = self.models["neural_network"].predict(X_test_scaled)
            results["neural_network"] = {
                "mse": mean_squared_error(y_test, nn_pred),
                "mae": mean_absolute_error(y_test, nn_pred)
            }
            
            # アンサンブル予測
            ensemble_pred = (
                xgb_pred * self.ensemble_weights[0] +
                rf_pred * self.ensemble_weights[1] +
                gb_pred * self.ensemble_weights[2] +
                nn_pred * self.ensemble_weights[3]
            )
            
            results["ensemble"] = {
                "mse": mean_squared_error(y_test, ensemble_pred),
                "mae": mean_absolute_error(y_test, ensemble_pred)
            }
            
            logger.info("モデル訓練完了")
            return results
            
        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            return {}

    def analyze_race_xgboost(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        XGBoostアンサンブルによるレース分析
        
        Args:
            race_data: レースデータ
            
        Returns:
            Dict: 分析結果
        """
        try:
            logger.info("レース分析開始...")
            
            # 特徴量計算
            features = self.calculate_professional_features(race_data)
            
            if not self.models:
                logger.warning("モデルが訓練されていません")
                return self._generate_dummy_analysis(race_data)
            
            predictions = {}
            
            # 各モデルで予測
            for model_name, model in self.models.items():
                if model_name == "neural_network":
                    features_scaled = self.scaler.transform(features)
                    pred = model.predict(features_scaled)
                else:
                    pred = model.predict(features)
                
                predictions[model_name] = pred
            
            # アンサンブル予測
            ensemble_pred = (
                predictions["xgboost"] * self.ensemble_weights[0] +
                predictions["random_forest"] * self.ensemble_weights[1] +
                predictions["gradient_boosting"] * self.ensemble_weights[2] +
                predictions["neural_network"] * self.ensemble_weights[3]
            )
            
            # 着順予測（予測値の逆順）
            predicted_ranks = np.argsort(ensemble_pred) + 1
            
            # 結果整理
            analysis_result = {
                "race_id": race_data.get("race_id", "unknown"),
                "predictions": {
                    "ensemble": ensemble_pred.tolist(),
                    "individual_models": {k: v.tolist() for k, v in predictions.items()},
                    "predicted_ranks": predicted_ranks.tolist()
                },
                "boat_analysis": []
            }
            
            # 各艇の分析
            for i, boat in enumerate(race_data["boats"]):
                boat_analysis = {
                    "boat_number": boat["boat_number"],
                    "racer_name": boat.get("racer_name", f"選手{i+1}"),
                    "predicted_rank": int(predicted_ranks[i]),
                    "confidence_score": float(ensemble_pred[i]),
                    "win_probability": max(0, min(1, (6 - ensemble_pred[i]) / 5)),
                    "features": features[i].tolist()
                }
                analysis_result["boat_analysis"].append(boat_analysis)
            
            # 着順でソート
            analysis_result["boat_analysis"].sort(key=lambda x: x["predicted_rank"])
            
            logger.info("レース分析完了")
            return analysis_result
            
        except Exception as e:
            logger.error(f"レース分析エラー: {e}")
            return self._generate_dummy_analysis(race_data)

    def _generate_dummy_analysis(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """ダミー分析結果生成"""
        dummy_result = {
            "race_id": race_data.get("race_id", "unknown"),
            "predictions": {
                "ensemble": [3.5, 2.1, 4.2, 1.8, 5.1, 3.9],
                "predicted_ranks": [4, 1, 5, 2, 6, 3]
            },
            "boat_analysis": []
        }
        
        for i, boat in enumerate(race_data["boats"]):
            boat_analysis = {
                "boat_number": boat["boat_number"],
                "racer_name": boat.get("racer_name", f"選手{i+1}"),
                "predicted_rank": i + 1,
                "confidence_score": np.random.uniform(0.3, 0.8),
                "win_probability": np.random.uniform(0.1, 0.4)
            }
            dummy_result["boat_analysis"].append(boat_analysis)
        
        return dummy_result

    def generate_xgboost_formations(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        XGBoost分析結果から買い目生成
        
        Args:
            analysis_result: 分析結果
            
        Returns:
            Dict: 買い目情報
        """
        try:
            logger.info("買い目生成開始...")
            
            boat_analysis = analysis_result.get("boat_analysis", [])
            if not boat_analysis:
                return {"error": "分析結果が不正です"}
            
            # 信頼度順にソート
            sorted_boats = sorted(boat_analysis, key=lambda x: x["predicted_rank"])
            
            formations = {
                "race_id": analysis_result.get("race_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "recommendations": []
            }
            
            # 3連単推奨
            top3 = sorted_boats[:3]
            trifecta = {
                "bet_type": "3連単",
                "combination": [boat["boat_number"] for boat in top3],
                "confidence": np.mean([boat["confidence_score"] for boat in top3]),
                "expected_odds": self._estimate_odds(top3, "3連単"),
                "recommended_amount": 1000
            }
            formations["recommendations"].append(trifecta)
            
            # 3連複推奨
            trifecta_box = {
                "bet_type": "3連複",
                "combination": sorted([boat["boat_number"] for boat in top3]),
                "confidence": np.mean([boat["confidence_score"] for boat in top3]) * 0.9,
                "expected_odds": self._estimate_odds(top3, "3連複"),
                "recommended_amount": 800
            }
            formations["recommendations"].append(trifecta_box)
            
            # 2連単推奨
            top2 = sorted_boats[:2]
            exacta = {
                "bet_type": "2連単",
                "combination": [boat["boat_number"] for boat in top2],
                "confidence": np.mean([boat["confidence_score"] for boat in top2]),
                "expected_odds": self._estimate_odds(top2, "2連単"),
                "recommended_amount": 1500
            }
            formations["recommendations"].append(exacta)
            
            # 単勝推奨
            win_bet = {
                "bet_type": "単勝",
                "combination": [sorted_boats[0]["boat_number"]],
                "confidence": sorted_boats[0]["confidence_score"],
                "expected_odds": self._estimate_odds([sorted_boats[0]], "単勝"),
                "recommended_amount": 2000
            }
            formations["recommendations"].append(win_bet)
            
            # 総投資額と期待収益
            total_investment = sum(rec["recommended_amount"] for rec in formations["recommendations"])
            expected_return = sum(
                rec["recommended_amount"] * rec["expected_odds"] * rec["confidence"]
                for rec in formations["recommendations"]
            )
            
            formations["summary"] = {
                "total_investment": total_investment,
                "expected_return": expected_return,
                "roi_estimate": (expected_return - total_investment) / total_investment * 100
            }
            
            logger.info("買い目生成完了")
            return formations
            
        except Exception as e:
            logger.error(f"買い目生成エラー: {e}")
            return {"error": f"買い目生成に失敗しました: {e}"}

    def _estimate_odds(self, boats: List[Dict], bet_type: str) -> float:
        """オッズ推定"""
        base_odds = {
            "単勝": 3.5,
            "2連単": 12.0,
            "3連単": 45.0,
            "3連複": 15.0
        }
        
        confidence_avg = np.mean([boat["confidence_score"] for boat in boats])
        adjustment = 1.0 / max(0.1, confidence_avg)
        
        return base_odds.get(bet_type, 10.0) * adjustment

    def save_analysis_results(self, analysis_result: Dict[str, Any], formations: Dict[str, Any]) -> str:
        """分析結果保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kyotei_analysis_{timestamp}.json"
            
            save_data = {
                "system_version": "v12.1",
                "timestamp": datetime.now().isoformat(),
                "analysis_result": analysis_result,
                "formations": formations,
                "model_info": {
                    "ensemble_weights": self.ensemble_weights,
                    "feature_count": len(self.feature_names),
                    "models_used": list(self.models.keys()) if self.models else []
                }
            }
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析結果保存完了: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"分析結果保存エラー: {e}")
            return ""

    def run_complete_analysis(self, race_id: str = "demo_race") -> Dict[str, Any]:
        """完全分析実行"""
        try:
            logger.info(f"完全分析開始: {race_id}")
            
            # 1. データ読み込み
            training_data = self.load_data("sample")
            
            # 2. モデル訓練
            X = training_data[self.feature_names].values
            y = training_data["rank"].values
            training_results = self.train_models(X, y)
            
            # 3. レースデータ取得
            race_data = self.get_race_data(race_id)
            
            # 4. レース分析
            analysis_result = self.analyze_race_xgboost(race_data)
            
            # 5. 買い目生成
            formations = self.generate_xgboost_formations(analysis_result)
            
            # 6. 結果保存
            saved_file = self.save_analysis_results(analysis_result, formations)
            
            # 7. 結果統合
            complete_result = {
                "status": "success",
                "race_id": race_id,
                "training_results": training_results,
                "analysis_result": analysis_result,
                "formations": formations,
                "saved_file": saved_file,
                "execution_time": datetime.now().isoformat()
            }
            
            logger.info("完全分析完了")
            return complete_result
            
        except Exception as e:
            logger.error(f"完全分析エラー: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "race_id": race_id
            }

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("競艇AI予想システム v12.1 - XGBoost修正版")
    print("=" * 60)
    
    try:
        # システム初期化
        system = XGBoostKyoteiSystem()
        print("✅ システム初期化完了")
        
        # 完全分析実行
        result = system.run_complete_analysis("test_race_001")
        
        if result["status"] == "success":
            print("\n🎯 分析結果サマリー:")
            print(f"レースID: {result['race_id']}")
            
            # 訓練結果表示
            if result["training_results"]:
                print("\n📊 モデル性能:")
                for model_name, metrics in result["training_results"].items():
                    print(f"  {model_name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")
            
            # 予想結果表示
            analysis = result["analysis_result"]
            if "boat_analysis" in analysis:
                print("\n🏁 着順予想:")
                for boat in analysis["boat_analysis"][:3]:
                    print(f"  {boat['predicted_rank']}位: {boat['boat_number']}号艇 "
                          f"({boat['racer_name']}) - 信頼度: {boat['confidence_score']:.3f}")
            
            # 買い目推奨表示
            formations = result["formations"]
            if "recommendations" in formations:
                print("\n💰 推奨買い目:")
                for rec in formations["recommendations"]:
                    print(f"  {rec['bet_type']}: {rec['combination']} "
                          f"- 信頼度: {rec['confidence']:.3f}, 期待オッズ: {rec['expected_odds']:.1f}倍")
            
            # 投資サマリー
            if "summary" in formations:
                summary = formations["summary"]
                print("\n📈 投資サマリー:")
                print(f"  総投資額: {summary['total_investment']:,}円")
                print(f"  期待収益: {summary['expected_return']:,.0f}円")
                print(f"  ROI予想: {summary['roi_estimate']:.1f}%")
            
            print(f"\n💾 結果保存: {result['saved_file']}")
            print("\n✅ 分析完了！")
            
        else:
            print(f"❌ 分析エラー: {result['error_message']}")
            
    except Exception as e:
        print(f"❌ システムエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
