# main.py
# FastAPI สำหรับพยากรณ์อัตราการว่างงานด้วย CatBoost
# ผู้ใช้ "ไม่ต้อง" ใส่ y_lag1 / y_lag4 / y_diff1 / y_roll4
# ระบบจะคำนวณให้จากประวัติ (history) ที่ส่งมา หรือจาก CSV ในโหมด batch

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import pandas as pd
import numpy as np
import os, json, io
from io import BytesIO
from catboost import CatBoostRegressor, Pool
import joblib

app = FastAPI(title="Unemployment Forecast API (Auto-Lag)")

# --------- โหลดโมเดล & ฟีเจอร์ ---------
def load_model():
    if os.path.exists("catboost_model_with_lags1.pkl"):
        return joblib.load("catboost_model_with_lags1.pkl")
    elif os.path.exists("catboost_model.cbm"):
        m = CatBoostRegressor()
        m.load_model("catboost_model.cbm")
        return m
    raise FileNotFoundError("ไม่พบโมเดล catboost_model.pkl หรือ catboost_model.cbm")

def load_features():
    if os.path.exists("features.json"):
        with open("features.json","r",encoding="utf-8") as f:
            feats = json.load(f)
        return [c for c in feats if c != "value_pct"]
    # ถ้าไม่มีไฟล์ ให้ใช้ fallback (ควรมี features.json จะดีที่สุด)
    return [
        "year","quarter_num","time_index_q",
        "sex_rec","age_rec","cohort_key",
        "y_lag1","y_lag4","y_diff1","y_roll4",
        "sex_clean_all","sex_clean_female","sex_clean_male",
        "age_group_clean_15-19","age_group_clean_20-24","age_group_clean_25-29",
        "age_group_clean_30-34","age_group_clean_35-39","age_group_clean_40-49",
        "age_group_clean_50-59","age_group_clean_60","age_group_clean_all"
    ]

MODEL = load_model()
FEATURES = load_features()

# ลำดับจริงที่โมเดลคาดหวัง
def get_model_feature_order(mdl, fallback):
    order = getattr(mdl, "feature_names_", None)
    return list(order) if order and len(order)>0 else list(fallback)

MODEL_FEATURE_ORDER = get_model_feature_order(MODEL, FEATURES)

# คอลัมน์ประเภทหมวดหมู่ (ต้องตรงกับตอนเทรน)
CATEGORICAL_NAMES = [c for c in ["sex_rec","age_rec","cohort_key"] if c in MODEL_FEATURE_ORDER]
NUMERIC_MAYBE     = [c for c in MODEL_FEATURE_ORDER if c not in CATEGORICAL_NAMES]

# --------- ตัวช่วย ----------
AGE_ONEHOTS = [
    "age_group_clean_15-19","age_group_clean_20-24","age_group_clean_25-29",
    "age_group_clean_30-34","age_group_clean_35-39","age_group_clean_40-49",
    "age_group_clean_50-59","age_group_clean_60","age_group_clean_all"
]
SEX_ONEHOTS = ["sex_clean_all","sex_clean_female","sex_clean_male"]

def onehot_from_choice(gender:str, age_label:str) -> dict:
    row = {k:False for k in SEX_ONEHOTS + AGE_ONEHOTS}
    if gender == "All":    row["sex_clean_all"] = True
    elif gender == "Male": row["sex_clean_male"] = True
    else:                  row["sex_clean_female"] = True
    age_map = {
        "15-19":"age_group_clean_15-19","20-24":"age_group_clean_20-24","25-29":"age_group_clean_25-29",
        "30-34":"age_group_clean_30-34","35-39":"age_group_clean_35-39","40-49":"age_group_clean_40-49",
        "50-59":"age_group_clean_50-59","60":"age_group_clean_60","All":"age_group_clean_all"
    }
    row[age_map[age_label]] = True
    return row

def recover_labels_from_onehot(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    age_cols = [c for c in d.columns if c.startswith("age_group_clean_")]
    def _sex(r):
        if r.get("sex_clean_female", False): return "female"
        if r.get("sex_clean_male",   False): return "male"
        return "all"
    def _age(r):
        for c in age_cols:
            if r.get(c, False): return c.replace("age_group_clean_", "")
        return "all"
    d["sex_rec"] = d.apply(_sex, axis=1)
    d["age_rec"] = d.apply(_age, axis=1)
    d["cohort_key"] = d["sex_rec"].astype(str) + "|" + d["age_rec"].astype(str)
    return d

def fe_add_group_time_features(d: pd.DataFrame) -> pd.DataFrame:
    """สร้าง y_lag1, y_lag4, y_diff1, y_roll4 ต่อ cohort จาก value_pct (อดีต)"""
    def _fn(g):
        g = g.sort_values("time_index_q").copy()
        g["y_lag1"]  = g["value_pct"].shift(1)
        g["y_lag4"]  = g["value_pct"].shift(4)
        g["y_diff1"] = g["value_pct"].diff(1)
        g["y_roll4"] = g["value_pct"].shift(1).rolling(4, min_periods=1).mean()
        return g
    try:
        return d.groupby("cohort_key", group_keys=False).apply(_fn, include_groups=False)
    except TypeError:  # pandas เก่า
        return d.groupby("cohort_key", group_keys=False).apply(_fn)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    if NUMERIC_MAYBE:
        df[NUMERIC_MAYBE] = df[NUMERIC_MAYBE].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

def predict_df(df: pd.DataFrame) -> np.ndarray:
    X = df[MODEL_FEATURE_ORDER]
    pool = Pool(X, feature_names=MODEL_FEATURE_ORDER, cat_features=CATEGORICAL_NAMES)
    return MODEL.predict(pool)

# --------- Schemas (ผู้ใช้ไม่ต้องส่ง lag) ---------
Gender   = Literal["All","Male","Female"]
AgeGroup = Literal["15-19","20-24","25-29","30-34","35-39","40-49","50-59","60","All"]

class HistPoint(BaseModel):
    time_index_q: int
    value_pct: Optional[float] = None  # อดีตที่ทราบ (ส่งได้บางจุด)

class SingleAutoIn(BaseModel):
    year: int = Field(..., ge=2000)
    quarter_num: int = Field(..., ge=1, le=4)
    time_index_q: int = Field(..., ge=0)      # จุดที่จะพยากรณ์
    gender: Gender
    age_group: AgeGroup
    history: List[HistPoint]                  # ประวัติของ cohort (time_index_q < target)

# --------- Endpoints ---------
@app.get("/health")
def health():
    return {"status":"ok", "n_features": len(MODEL_FEATURE_ORDER)}

@app.post("/predict/single")  # คำนวณ lag จาก history อัตโนมัติ
def predict_single(inp: SingleAutoIn):
    # เตรียม one-hot + labels + cohort key
    onehots  = onehot_from_choice(inp.gender, inp.age_group)
    sex_rec  = "female" if inp.gender=="Female" else ("male" if inp.gender=="Male" else "all")
    age_rec  = inp.age_group if inp.age_group!="All" else "all"
    cohort_key = f"{sex_rec}|{age_rec}"

    # ---- ดึง history เฉพาะอดีตก่อนเป้าหมาย
    if not inp.history:
        raise HTTPException(400, "history is empty")
    hist_df = pd.DataFrame([h.dict() for h in inp.history])
    if "time_index_q" not in hist_df.columns:
        raise HTTPException(400, "history must include time_index_q")
    hist_df = hist_df.sort_values("time_index_q")
    hist_df = hist_df[hist_df["time_index_q"] < inp.time_index_q]

    if hist_df.empty or ("value_pct" not in hist_df.columns) or hist_df["value_pct"].isna().all():
        raise HTTPException(400, "No valid past 'value_pct' in history before target time_index_q")

    # ---- คำนวณ lag อัตโนมัติ (สูตรเดียวกับตอนเทรน)
    last_vals = hist_df["value_pct"].dropna().values
    y_lag1  = float(last_vals[-1]) if len(last_vals)>=1 else float("nan")
    y_lag4  = float(last_vals[-4]) if len(last_vals)>=4 else float("nan")
    y_diff1 = float(last_vals[-1] - last_vals[-2]) if len(last_vals)>=2 else float("nan")
    y_roll4 = float(pd.Series(last_vals[:-0]).tail(4).mean()) if len(last_vals)>=1 else float("nan")

    # fallback (กันกรณีอดีตมีไม่พอ)
    fallback = float(pd.Series(last_vals).mean())
    if np.isnan(y_lag1):  y_lag1  = fallback
    if np.isnan(y_lag4):  y_lag4  = fallback
    if np.isnan(y_diff1): y_diff1 = 0.0
    if np.isnan(y_roll4): y_roll4 = fallback

    # ---- สร้างแถวฟีเจอร์ตามลำดับที่โมเดลต้องการ
    row = {
        "year": inp.year, "quarter_num": inp.quarter_num, "time_index_q": inp.time_index_q,
        "sex_rec": sex_rec, "age_rec": age_rec, "cohort_key": cohort_key,
        "y_lag1": y_lag1, "y_lag4": y_lag4, "y_diff1": y_diff1, "y_roll4": y_roll4,
        **onehots
    }
    for c in MODEL_FEATURE_ORDER:
        if c not in row: row[c] = 0
    X = pd.DataFrame([row])[MODEL_FEATURE_ORDER]

    # ---- ทำนาย
    pred = float(predict_df(X)[0])
    return {
        "prediction_value_pct": pred,
        "cohort_key": cohort_key,
        "computed_lags": {"y_lag1": y_lag1, "y_lag4": y_lag4, "y_diff1": y_diff1, "y_roll4": y_roll4}
    }

@app.post("/predict/batch")   # อัปโหลด CSV แล้วระบบสร้าง lag ให้เองตาม cohort
def predict_batch(file: UploadFile = File(...)):
    content = file.file.read()
    df_raw = pd.read_csv(BytesIO(content))

    # ต้องมีอย่างน้อย: year, quarter_num, time_index_q, value_pct + one-hot เพศ/อายุ
    must_cols = ["year","quarter_num","time_index_q","value_pct"] + SEX_ONEHOTS + AGE_ONEHOTS
    missing = [c for c in must_cols if c not in df_raw.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    # กู้ labels + cohort และคำนวณ lag ตามสูตรเทรน
    df = recover_labels_from_onehot(df_raw)
    df = fe_add_group_time_features(df)
    df = ensure_columns(df, MODEL_FEATURE_ORDER)

    preds = predict_df(df)
    out = df_raw.copy()
    out["prediction_value_pct"] = preds

    # ส่งตัวอย่างกลับ
    return {"rows": len(out), "preview": out.head(12).to_dict(orient="records")}
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# เสิร์ฟโฟลเดอร์ /public เป็น static ที่ /web
app.mount("/web", StaticFiles(directory="public", html=True), name="web")

# ทำให้เปิด root แล้วพาไปหน้าเว็บ
@app.get("/")
def root():
    return RedirectResponse(url="/web")

