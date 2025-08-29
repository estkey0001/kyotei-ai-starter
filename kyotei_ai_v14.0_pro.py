# -*- coding: utf-8 -*-
# kyotei_ai_v14.0_pro.py
# 商用レベル 競艇AI予想システム（単一ファイル）
# - LightGBM/CatBoost モデル学習
# - 包括的特徴量設計（選手・モーター・展示・進入・気象・会場・時間帯・オッズ乖離・派生）
# - 1～3着確率、3連単フォーメーション、EV計算、過大/過小評価艇
# - 予想根拠の詳細説明（各艇の重要特徴）
# - note記事（2000文字以上）自動生成
# - SQLite（選手名DB）連携
# - サイドバー廃止・1画面統合UI
# - エラーハンドリング強化、レスポンシブUI

import os
import sys
import io
import json
import math
import time
import sqlite3
import itertools
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st

# ML libraries
LGBM_AVAILABLE = True
CAT_AVAILABLE = True
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, Pool
except Exception:
    CAT_AVAILABLE = False

# =========================
# パス・環境変数
# =========================
APP_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATA_DIRS = [
    os.getenv("KYOTEI_DATA_DIR", ""),                              # 優先: 環境変数
    os.path.join(APP_DIR, "kyotei_data"),                          # v13系互換
    os.path.join(APP_DIR, "data", "coconala_2024"),                # 既存実データ
]
DEFAULT_DB_PATHS = [
    os.path.join(APP_DIR, "kyotei_racer_master.db"),
    os.path.join(APP_DIR, "data", "kyotei_racer_master.db"),
]

VENUE_CANON = {
    "edogawa": ["江戸川", "edogawa"],
    "heiwajima": ["平和島", "heiwajima"],
    "suminoe": ["住之江", "suminoe"],
    "toda": ["戸田", "toda"],
    "omura": ["大村", "omura"],
}

# =========================
# ユーティリティ
# =========================
def find_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None

def to_datetime_safe(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)):
        # yyyymmdd など
        try:
            s = str(int(x))
            return datetime.strptime(s, "%Y%m%d")
        except Exception:
            pass
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def to_numeric_safe(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan]*len(s))

def odds_to_implied_prob(odds):
    # オッズ → 期待確率（単純逆数）
    try:
        f = float(odds)
        if f <= 0:
            return np.nan
        return 1.0 / f
    except Exception:
        return np.nan

def normalize_probs(arr: np.ndarray) -> np.ndarray:
    s = np.nansum(arr)
    if s <= 0 or np.isnan(s):
        return np.array([np.nan if np.isnan(x) else 0.0 for x in arr])
    return arr / s

# =========================
# 選手名DB
# =========================
class RacerMasterDB:
    def __init__(self, db_paths: List[str]):
        self.db_path = find_existing_path(db_paths)
        self.conn = None
        if self.db_path and os.path.exists(self.db_path):
            try:
                self.conn = sqlite3.connect(self.db_path)
            except Exception:
                self.conn = None

    def get_names(self, racer_ids: List[str]) -> Dict[str, str]:
        out = {str(r): str(r) for r in racer_ids}
        if not self.conn:
            return out
        try:
            qmarks = ",".join("?"*len(racer_ids))
            df = pd.read_sql_query(
                f"SELECT racer_id, racer_name FROM racers WHERE racer_id IN ({qmarks})",
                self.conn,
                params=[str(r) for r in racer_ids]
            )
            for _, row in df.iterrows():
                out[str(row["racer_id"])] = str(row["racer_name"])
        except Exception:
            pass
        return out

# =========================
# データローダ
# =========================
class KyoteiDataLoader:
    def __init__(self, data_dirs: List[str]):
        self.data_dir = find_existing_path(data_dirs)

    def _venue_key(self, s: str) -> Optional[str]:
        s = (s or "").lower()
        for k, aliases in VENUE_CANON.items():
            for a in aliases:
                if a in s:
                    return k
        return None

    def list_venue_files(self) -> Dict[str, str]:
        """
        既知5場のCSVファイル候補を探索
        優先: edogawa_*.csv 等
        """
        out = {}
        if not self.data_dir:
            return out
        for fn in os.listdir(self.data_dir):
            if not fn.lower().endswith(".csv"):
                continue
            key = self._venue_key(fn)
            if key and key not in out:
                out[key] = os.path.join(self.data_dir, fn)
        return out

    def load_venue_df(self, venue_key: str) -> Optional[pd.DataFrame]:
        files = self.list_venue_files()
        path = files.get(venue_key)
        if not path:
            return None
        df = safe_read_csv(path)
        return df

# =========================
# スキーマ正規化
# =========================
class SchemaNormalizer:
    def __init__(self):
        pass

    def normalize(self, df: pd.DataFrame, venue_key: str) -> pd.DataFrame:
        if df is None or len(df)==0:
            return pd.DataFrame()

        # 列名を英字寄りに標準化（内部カラム）
        col_map = {
            "登録番号": "racer_id",
            "選手ID": "racer_id",
            "racer_id": "racer_id",

            "年齢": "age",
            "級別": "grade",
            "体重": "weight",
            "勝率": "racer_winrate",
            "事故率": "accident_rate",

            "モーター番号": "motor_no",
            "モーター勝率": "motor_winrate",
            "2連対率": "motor_2rate",
            "3連対率": "motor_3rate",

            "枠番": "lane",
            "コース": "course_in",
            "展示進入順": "ex_course_order",

            "展示タイム": "ex_time",
            "直前気配": "ex_mood",
            "スタート展示": "ex_start",

            "天候": "weather",
            "気温": "temp",
            "風速": "wind_speed",
            "風向": "wind_dir",
            "波高": "wave",
            "湿度": "humidity",

            "競艇場名": "venue_name",
            "場": "venue_name",
            "水質": "water_type",
            "干満差": "tide_diff",

            "レース番号": "race_no",
            "ナイター": "is_nighter",
            "日付": "date",

            "オッズ": "odds",
            "人気": "popularity",

            "着順": "rank",
        }

        # 既知カラムを変換（存在分のみ）
        working = df.copy()
        for c in list(working.columns):
            if c in col_map:
                working.rename(columns={c: col_map[c]}, inplace=True)

        # venue_name を補完
        if "venue_name" not in working.columns:
            working["venue_name"] = VENUE_CANON.get(venue_key, [venue_key])[0]

        # date 正規化
        if "date" in working.columns:
            working["date"] = working["date"].apply(to_datetime_safe)
        else:
            # CSVにない場合は推定不能→NaT
            working["date"] = pd.NaT

        # race_no 正規化
        if "race_no" in working.columns:
            working["race_no"] = to_numeric_safe(working["race_no"]).astype("Int64")

        # 数値化候補
        for num_col in [
            "age","weight","racer_winrate","accident_rate","motor_no",
            "motor_winrate","motor_2rate","motor_3rate",
            "lane","course_in","ex_course_order",
            "ex_time","temp","wind_speed","wave","humidity",
            "odds","popularity",
        ]:
            if num_col in working.columns:
                working[num_col] = pd.to_numeric(working[num_col], errors="coerce")

        # grade 正規化
        if "grade" in working.columns:
            working["grade"] = working["grade"].astype(str).str.upper().str.strip()

        # wind_dir 正規化（方位を角度化）
        if "wind_dir" in working.columns:
            wd = working["wind_dir"].astype(str)
            # 東西南北→角度（例: 北=0, 東=90, 南=180, 西=270）
            compass = {"北":0, "東":90, "南":180, "西":270}
            working["wind_deg"] = wd.map(compass).fillna(np.nan)
        else:
            working["wind_deg"] = np.nan

        # is_rainy
        if "weather" in working.columns:
            w = working["weather"].astype(str)
            working["is_rainy"] = w.str.contains("雨", na=False).astype(int)
        else:
            working["is_rainy"] = 0

        # is_nighter
        if "is_nighter" not in working.columns:
            working["is_nighter"] = 0

        # rank 正規化（F, L, K 等は着外扱い）
        if "rank" in working.columns:
            r = working["rank"].astype(str).str.extract(r"(\d+)")[0]
            working["rank"] = pd.to_numeric(r, errors="coerce")
        else:
            working["rank"] = np.nan

        # racer_id 文字列化
        if "racer_id" in working.columns:
            working["racer_id"] = working["racer_id"].astype(str).str.replace(".0","", regex=False)
        else:
            working["racer_id"] = ""

        # lane 補完（なければ course_in）
        if "lane" not in working.columns and "course_in" in working.columns:
            working["lane"] = working["course_in"]

        # 必須最低限
        base_cols = [
            "date","venue_name","race_no","racer_id","lane","grade",
            "racer_winrate","accident_rate","ex_time",
            "motor_no","motor_winrate","motor_2rate","motor_3rate",
            "course_in","ex_course_order","ex_mood","ex_start",
            "weather","temp","wind_speed","wave","wind_deg","humidity",
            "water_type","tide_diff","is_rainy","is_nighter",
            "odds","popularity","rank","age","weight"
        ]
        for c in base_cols:
            if c not in working.columns:
                working[c] = np.nan

        # ソート
        working = working.sort_values(["date","venue_name","race_no","lane"], kind="stable")
        working.reset_index(drop=True, inplace=True)
        return working

# =========================
# 特徴量生成
# =========================
class FeatureBuilder:
    def __init__(self):
        pass

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df)==0:
            return pd.DataFrame()

        work = df.copy()

        # 基本欠損埋め
        for c in ["racer_winrate","accident_rate","ex_time","motor_winrate","motor_2rate","motor_3rate","wind_speed","wave","temp","humidity","wind_deg"]:
            if c in work.columns:
                work[c] = work[c].fillna(work[c].median())

        # カテゴリ→数値
        # grade: A1=3, A2=2, B1=1, B2=0
        grade_map = {"A1":3,"A2":2,"B1":1,"B2":0}
        work["grade_code"] = work["grade"].map(grade_map).fillna(1)

        # 水質: 海水=1, 淡水=0 それ以外はNA→0.5
        water_map = {"海水":1.0,"淡水":0.0}
        work["water_code"] = work["water_type"].map(water_map).fillna(0.5)

        # 風の強さ分類
        work["wind_cat"] = pd.cut(work["wind_speed"].fillna(0), bins=[-1,1,3,5,100], labels=[0,1,2,3]).astype(int)

        # 天候特徴
        work["is_windy"] = (work["wind_speed"]>=4).astype(int)
        work["is_high_wave"] = (work["wave"]>=3).astype(int)

        # コース関連
        work["lane"] = pd.to_numeric(work["lane"], errors="coerce")
        work["lane"].fillna(3, inplace=True)
        for l in range(1,7):
            work[f"is_lane_{l}"] = (work["lane"]==l).astype(int)

        # 直近成績（リーク防止で時系列順にshift）
        work = work.sort_values(["racer_id","date","race_no","lane"]).reset_index(drop=True)
        def add_rolling(g):
            g = g.copy()
            # rank_small: 1~6、着外は7扱い
            r = g["rank"].copy()
            r = r.where(~r.isna(), 7)
            g["lag_rank"] = r.shift(1)
            g["avg_rank_3"] = r.shift(1).rolling(3, min_periods=1).mean()
            g["top3_rate_10"] = (r.shift(1)<=3).rolling(10, min_periods=3).mean()
            g["dnf_rate_10"] = (r.shift(1)>=7).rolling(10, min_periods=3).mean()

            # コース別ローリング
            for l in range(1,7):
                mask = (g["lane"]==l)
                series = r.where(mask, np.nan)
                g[f"lane{l}_avg_rank_6"] = series.shift(1).rolling(6, min_periods=1).mean()
                g[f"lane{l}_top3_6"] = (series.shift(1)<=3).rolling(6, min_periods=1).mean()
            return g

        work = work.groupby("racer_id", group_keys=False).apply(add_rolling)

        # 会場×選手の適性（会場平均着順、上位率）
        def add_venue_roll(g):
            g = g.copy()
            r = g["rank"].copy()
            r = r.where(~r.isna(), 7)
            g["venue_avg_rank_10"] = r.shift(1).rolling(10, min_periods=1).mean()
            g["venue_top3_10"] = (r.shift(1)<=3).rolling(10, min_periods=1).mean()
            return g
        work = work.sort_values(["racer_id","venue_name","date","race_no"])
        work = work.groupby(["racer_id","venue_name"], group_keys=False).apply(add_venue_roll)

        # 気象適性: 雨・高風・高波の時の上位率（選手別）
        def add_weather_roll(g):
            g = g.copy()
            r = g["rank"].copy().where(~g["rank"].isna(), 7)
            # 雨
            rainy = g["is_rainy"]==1
            g["rain_top3_10"] = (r.shift(1).where(rainy, np.nan)<=3).rolling(10, min_periods=1).mean()
            # 強風
            windy = g["is_windy"]==1
            g["windy_top3_10"] = (r.shift(1).where(windy, np.nan)<=3).rolling(10, min_periods=1).mean()
            # 高波
            hw = g["is_high_wave"]==1
            g["highwave_top3_10"] = (r.shift(1).where(hw, np.nan)<=3).rolling(10, min_periods=1).mean()
            return g
        work = work.groupby("racer_id", group_keys=False).apply(add_weather_roll)

        # モーター関連
        # その大会・節内のモーター番号が同一で近似
        # 簡易的に motor_winrate/2rate/3rate を一次特徴として採用（欠損は中央値）
        for c in ["avg_rank_3","top3_rate_10","dnf_rate_10",
                  "venue_avg_rank_10","venue_top3_10",
                  "rain_top3_10","windy_top3_10","highwave_top3_10"]:
            if c in work.columns:
                work[c] = work[c].fillna(work[c].median())

        # オッズ乖離（単勝があれば→期待確率）
        if "odds" in work.columns:
            work["implied_prob"] = work["odds"].apply(odds_to_implied_prob)
        else:
            work["implied_prob"] = np.nan

        # 相互作用（例: lane×is_windy）
        work["lane_windy"] = work["lane"] * work["is_windy"]

        # 目的変数の準備（1着/2着/3着）
        y1 = (work["rank"]==1).astype(int)
        y2 = (work["rank"]==2).astype(int)
        y3 = (work["rank"]==3).astype(int)

        # 特徴量カラム
        feature_cols = [
            # 選手基本
            "grade_code","racer_winrate","accident_rate","age","weight",
            # 展示
            "ex_time",
            # 進入・コース
            "lane","course_in","ex_course_order",
            "is_lane_1","is_lane_2","is_lane_3","is_lane_4","is_lane_5","is_lane_6",
            # モーター
            "motor_winrate","motor_2rate","motor_3rate",
            # 気象
            "temp","wind_speed","wind_deg","wave","humidity",
            "is_rainy","is_windy","is_high_wave","wind_cat",
            # 会場・時間帯
            "water_code","is_nighter",
            # ローリング
            "lag_rank","avg_rank_3","top3_rate_10","dnf_rate_10",
            "venue_avg_rank_10","venue_top3_10",
            "rain_top3_10","windy_top3_10","highwave_top3_10",
            # オッズ関連（単勝）
            "implied_prob",
            # 交互作用
            "lane_windy",
        ]
        # 存在しない列は落とす
        feature_cols = [c for c in feature_cols if c in work.columns]

        # 型と欠損
        X = work[feature_cols].copy()
        X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)

        work["_y1"] = y1
        work["_y2"] = y2
        work["_y3"] = y3
        work["_is_label"] = (~work["rank"].isna()).astype(int)
        work["_features"] = [feature_cols]*len(work)
        for c in feature_cols:
            work[c] = X[c]

        return work

# =========================
# モデル学習・推論
# =========================
class MLModel:
    def __init__(self, algo="lightgbm", random_state=42):
        self.algo = "catboost" if (algo=="catboost" and CAT_AVAILABLE) else ("lightgbm" if LGBM_AVAILABLE else None)
        if self.algo is None and CAT_AVAILABLE:
            self.algo = "catboost"
        self.random_state = random_state
        self.models = {}  # {"y1": model, "y2": model, "y3": model}
        self.feature_names_ = None

    def _fit_one(self, X, y):
        if self.algo == "lightgbm":
            model = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X, y)
            return model
        else:
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                verbose=False,
                random_seed=self.random_state
            )
            model.fit(X, y)
            return model

    def fit(self, df_feat: pd.DataFrame, cutoff_date: Optional[datetime]=None, venue_filter: Optional[str]=None):
        d = df_feat.copy()
        if cutoff_date is not None:
            d = d[(~d["date"].isna()) & (d["date"] < cutoff_date)]
        if venue_filter:
            d = d[d["venue_name"].astype(str).str.contains(venue_filter, na=False)]

        d = d[d["_is_label"]==1]
        if len(d) < 500:
