# main.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, conlist
import joblib

# ---------- Load artifacts ----------
MODEL_PATH = "insurance_top_k_recommendation_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
LABELS_PATH = "label_names.pkl"
SELECTED_FEATURES_PATH = "selected_features.pkl"  # <-- You should have saved this

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_names: np.ndarray = joblib.load(LABELS_PATH)
    selected_features: List[str] = joblib.load(SELECTED_FEATURES_PATH)  # order matters
except Exception as e:
    raise RuntimeError(
        f"Failed to load artifacts: {e}. "
        f"Ensure {MODEL_PATH}, {SCALER_PATH}, {LABELS_PATH}, and {SELECTED_FEATURES_PATH} exist."
    )

app = FastAPI(title="Insurance Top‑K Recommender", version="1.0")

# ---------- Pydantic schema (raw payload) ----------
class RecommendationRequest(BaseModel):
    # A) Demographics & status
    ClientGender: Optional[str] = None
    Nationality: Optional[str] = None
    SpokenLanguage: Optional[str] = None     # e.g. "English,Mandarin"
    WrittenLanguage: Optional[str] = None
    Education: Optional[str] = None
    EmploymentStatus: Optional[str] = None
    Occupation: Optional[str] = None
    MaritalStatus: Optional[str] = None
    IncomeRange: Optional[str] = None
    RiskProfile: Optional[str] = None
    CKAProfile: Optional[str] = None
    CARProfile: Optional[str] = None
    ClientResidentialStatus: Optional[str] = None
    CountryOfBirth: Optional[str] = None
    Race: Optional[str] = None
    ClientAge: Optional[float] = None
    CurrencyCode: Optional[str] = "SGD"

    # B) Temporal & session
    ClientInvitedDate: Optional[datetime] = None
    EMFCSubmitDate: Optional[datetime] = None
    EMFC_Count: Optional[float] = 0

    # C) Assets & balances
    SavingsAccounts: Optional[float] = 0
    FixedDepositsAccount: Optional[float] = 0
    HomeAsset: Optional[float] = 0
    MotorAsset: Optional[float] = 0
    InsuranceCashValues: Optional[float] = 0
    StocksPortofolio: Optional[float] = 0
    BondPortofolio: Optional[float] = 0
    UTFEquityAsset: Optional[float] = 0
    ETFs: Optional[float] = 0
    InvestmentProperties: Optional[float] = 0
    CPFOABalance: Optional[float] = 0
    CPFSABalance: Optional[float] = 0
    CPFMABalance: Optional[float] = 0
    SRSEquityAsset: Optional[float] = 0

    # D) Coverage & premium totals
    Total_Life_Coverage: Optional[float] = 0
    Total_CI_Coverage: Optional[float] = 0
    Total_Hospital_Income: Optional[float] = 0
    Total_LTC_Coverage: Optional[float] = 0
    Total_Annual_Premium: Optional[float] = 0

    # E) Portfolio composition
    Plan_Types: Optional[List[str]] = Field(default=None, description="e.g. ['Term','Integrated Shield']")
    Insurance_Companies: Optional[List[str]] = None

# ---------- Utilities: feature engineering (mirror Phase‑1) ----------
def _normalize_lang(val: Optional[str]) -> str:
    if val is None or str(val).strip() == "":
        return "Unknown"
    langs = sorted([s.strip().title() for s in str(val).split(",") if s.strip()])
    return "|".join(langs) if langs else "Unknown"

def _income_numeric(income_range: Optional[str]) -> float:
    mapping = {
        "No Income": 0,
        "Below S$30,000": 15000,
        "S$30,000 - S$49,999": 40000,
        "S$50,000 - S$99,999": 75000,
        "S$100,000 and above": 150000,
    }
    return float(mapping.get(income_range or "", 0))

def _income_category(income_range: Optional[str]) -> str:
    cat_map = {
        "No Income": "Low",
        "Below S$30,000": "Low",
        "S$30,000 - S$49,999": "Medium",
        "S$50,000 - S$99,999": "Medium",
        "S$100,000 and above": "High",
    }
    return cat_map.get(income_range or "", "Unknown")

def _age_group(age: Optional[float]) -> str:
    if age is None: return "Unknown"
    try:
        a = float(age)
    except:
        return "Unknown"
    bins = [(0,25,"Under_25"),(25,35,"25-35"),(35,45,"35-45"),(45,55,"45-55"),(55,65,"55-65"),(65,1e9,"Over_65")]
    for lo, hi, label in bins:
        if lo <= a < hi: return label
    return "Unknown"

def _life_stage(age_group: str, marital: Optional[str]) -> str:
    if age_group == "Unknown" or marital is None: return "Unknown"
    m = (marital or "").lower()
    if age_group in ["Under_25","25-35"]:
        return "Young_Single" if "single" in m else "Young_Family"
    if age_group in ["35-45","45-55"]:
        return "Mid_Career_Single" if "single" in m else "Mid_Career_Family"
    return "Pre_Retirement"

def _fin_sophistication(education: Optional[str], inv_ratio: float) -> str:
    edu = (education or "").lower()
    score = 0
    if "university" in edu or "degree" in edu: score += 2
    elif "diploma" in edu: score += 1
    if inv_ratio > 0.3: score += 2
    elif inv_ratio > 0.1: score += 1
    return "High" if score >= 3 else ("Medium" if score >= 1 else "Low")

def _temporal_features(now: datetime, invited: Optional[datetime], submit: Optional[datetime], emfc_count: float):
    if invited is None:
        tenure_days = 0.0
    else:
        tenure_days = max(0.0, (now - invited).total_seconds()/86400.0)
    tenure_years = tenure_days / 365.25

    if submit is None:
        days_since = 0.0
    else:
        days_since = max(0.0, (now - submit).total_seconds()/86400.0)
    months_since = days_since / 30.44

    fna_freq = float(emfc_count or 0) / (tenure_years + 1.0)
    return tenure_days, tenure_years, days_since, months_since, fna_freq

def _portfolio_affinities(plan_types: Optional[List[str]]) -> Dict[str, int]:
    pt = [p.strip() for p in (plan_types or [])]
    has = lambda name: int(any(name == x or name in str(x) for x in pt))
    feats = {
        "Affinity_Term_to_Whole_Life": has("Term"),
        "Affinity_Term_to_Investment-Linked": has("Term"),
        "Affinity_Term_to_Universal_Life": has("Term"),
        "Affinity_Whole_Life_to_Investment-Linked": has("Whole Life"),
        "Affinity_Whole_Life_to_Endowment": has("Whole Life"),
        "Affinity_Whole_Life_to_Annuity": has("Whole Life"),
        "Affinity_Endowment_to_Investment-Linked": has("Endowment"),
        "Affinity_Endowment_to_Annuity": has("Endowment"),
        "Affinity_Critical_Illness_to_Early_Critical_Illness": has("Critical Illness"),
        "Affinity_Critical_Illness_to_Long_Term_Care": has("Critical Illness"),
    }
    # Diversity & evolution stage
    diversity = len(set(pt))
    # Evolution: 0 none; 1 basic; 2 developing; 3 intermediate; 4 advanced
    plan_str = " ".join(pt)
    if not pt:
        stage = 0
    elif ("Investment-Linked" in plan_str) or ("Annuity" in plan_str):
        stage = 4
    elif ("Whole Life" in plan_str) or ("Endowment" in plan_str):
        stage = 3
    elif ("Critical Illness" in plan_str) or ("Disability" in plan_str):
        stage = 2
    else:
        stage = 1

    feats["Product_Diversity_Score"] = diversity
    feats["Insurance_Evolution_Stage"] = stage
    return feats

def _coverage_flags_and_ratios(total_life: float, total_ci: float, total_hosp: float, total_ltc: float,
                               total_prem: float, income_numeric: float, total_policies_known: Optional[float]=None):
    has_insurance = int((total_policies_known or 0) > 0 or any([total_life, total_ci, total_hosp, total_ltc]))
    has_life = int((total_life or 0) > 0)
    has_ci = int((total_ci or 0) > 0)
    has_hosp = int((total_hosp or 0) > 0)
    has_ltc = int((total_ltc or 0) > 0)

    life_multiple = (total_life / income_numeric) if income_numeric and income_numeric > 0 else 0.0
    prem_to_income = (total_prem / income_numeric) if income_numeric and income_numeric > 0 else 0.0

    life_gap = int(has_insurance == 1 and has_life == 0)
    ci_gap = int(has_insurance == 1 and has_ci == 0)

    # sophistication by coverage count
    cov_count = has_life + has_ci + has_hosp + has_ltc
    if has_insurance == 0:
        sophistication = "No_Insurance"
    elif cov_count >= 3:
        sophistication = "Comprehensive"
    elif cov_count >= 2:
        sophistication = "Moderate"
    else:
        sophistication = "Basic"

    return dict(
        Has_Insurance=has_insurance,
        Has_Life_Coverage=has_life,
        Has_CI_Coverage=has_ci,
        Has_Hospital_Coverage=has_hosp,
        Has_LTC_Coverage=has_ltc,
        Life_Coverage_Multiple=life_multiple,
        Premium_to_Income_Ratio=prem_to_income,
        Life_Coverage_Gap=life_gap,
        CI_Coverage_Gap=ci_gap,
        Insurance_Sophistication=sophistication,
    )

# ---------- Core: build model-ready row ----------
def build_feature_row(req: RecommendationRequest) -> pd.DataFrame:
    now = datetime.now(timezone.utc)

    # Normalize languages
    spoken_norm = _normalize_lang(req.SpokenLanguage)
    written_norm = _normalize_lang(req.WrittenLanguage)

    # Income transforms
    income_num = _income_numeric(req.IncomeRange)
    income_cat = _income_category(req.IncomeRange)

    # Asset totals
    total_liquid = float(req.SavingsAccounts or 0) + float(req.FixedDepositsAccount or 0)
    total_inv = float(req.StocksPortofolio or 0) + float(req.BondPortofolio or 0) + float(req.UTFEquityAsset or 0) + float(req.ETFs or 0)
    total_cpf = float(req.CPFOABalance or 0) + float(req.CPFSABalance or 0) + float(req.CPFMABalance or 0)
    est_net_worth = total_liquid + total_inv + total_cpf + float(req.InvestmentProperties or 0)
    denom = total_liquid + total_inv
    inv_ratio = (total_inv / denom) if denom > 0 else 0.0

    # Temporal features
    tenure_days, tenure_years, days_since, months_since, fna_freq = _temporal_features(
        now, req.ClientInvitedDate, req.EMFCSubmitDate, req.EMFC_Count or 0
    )

    # Affinities, diversity, evolution
    aff = _portfolio_affinities(req.Plan_Types or [])

    # Coverage flags/ratios
    cov = _coverage_flags_and_ratios(
        float(req.Total_Life_Coverage or 0),
        float(req.Total_CI_Coverage or 0),
        float(req.Total_Hospital_Income or 0),
        float(req.Total_LTC_Coverage or 0),
        float(req.Total_Annual_Premium or 0),
        income_num,
        None,  # Total_Policies not provided in payload; flags infer from totals
    )

    # Age group & life stage & financial sophistication
    age_grp = _age_group(req.ClientAge)
    life_stage = _life_stage(age_grp, req.MaritalStatus)
    fin_soph = _fin_sophistication(req.Education, inv_ratio)

    # Build a single-row DataFrame containing ALL expected features.
    # Any missing columns will be added later (with default).
    row: Dict[str, Any] = {
        # direct passthroughs
        "ClientGender": req.ClientGender,
        "Nationality": req.Nationality,
        "SpokenLanguage": spoken_norm,
        "WrittenLanguage": written_norm,
        "Education": req.Education,
        "EmploymentStatus": req.EmploymentStatus,
        "Occupation": req.Occupation,
        "MaritalStatus": req.MaritalStatus,
        "IncomeRange": req.IncomeRange,
        "RiskProfile": req.RiskProfile,
        "CKAProfile": req.CKAProfile,
        "CARProfile": req.CARProfile,
        "ClientResidentialStatus": req.ClientResidentialStatus,
        "CountryOfBirth": req.CountryOfBirth,
        "Race": req.Race,
        "ClientAge": req.ClientAge,
        "CurrencyCode": req.CurrencyCode,
        # temporal
        "Client_Tenure_Days": tenure_days,
        "Client_Tenure_Years": tenure_years,
        "Days_Since_Last_FNA": days_since,
        "Months_Since_Last_FNA": months_since,
        "FNA_Frequency": fna_freq,
        "EMFC_Count": req.EMFC_Count or 0,
        # assets
        "SavingsAccounts": req.SavingsAccounts or 0,
        "FixedDepositsAccount": req.FixedDepositsAccount or 0,
        "HomeAsset": req.HomeAsset or 0,
        "MotorAsset": req.MotorAsset or 0,
        "InsuranceCashValues": req.InsuranceCashValues or 0,
        "StocksPortofolio": req.StocksPortofolio or 0,
        "BondPortofolio": req.BondPortofolio or 0,
        "UTFEquityAsset": req.UTFEquityAsset or 0,
        "ETFs": req.ETFs or 0,
        "InvestmentProperties": req.InvestmentProperties or 0,
        "CPFOABalance": req.CPFOABalance or 0,
        "CPFSABalance": req.CPFSABalance or 0,
        "CPFMABalance": req.CPFMABalance or 0,
        "SRSEquityAsset": req.SRSEquityAsset or 0,
        # portfolio companies (kept as categorical)
        "Insurance_Companies": "|".join(req.Insurance_Companies) if req.Insurance_Companies else "Unknown",
        # income & buckets
        "Income_Numeric": income_num,
        "Income_Category": income_cat,
        "Total_Liquid_Assets": total_liquid,
        "Total_Investments": total_inv,
        "Total_CPF": total_cpf,
        "Estimated_Net_Worth": est_net_worth,
        "Investment_Ratio": inv_ratio,
        "Age_Group": age_grp,
        "Life_Stage": life_stage,
        "Financial_Sophistication": fin_soph,
        # coverage-derived
        **cov,
        # affinity & composition
        **aff,
    }

    df = pd.DataFrame([row])

    # Ensure ALL selected_features columns exist, fill missing with sensible defaults
    for col in selected_features:
        if col not in df.columns:
            # default numeric 0, else "Unknown"
            df[col] = 0 if col not in (
                "ClientGender","Nationality","SpokenLanguage","WrittenLanguage","Education",
                "EmploymentStatus","Occupation","MaritalStatus","IncomeRange","RiskProfile",
                "CKAProfile","CARProfile","ClientResidentialStatus","CountryOfBirth","Race",
                "Insurance_Companies","Income_Category","Age_Group","Life_Stage",
                "Financial_Sophistication","CurrencyCode"
            ) else "Unknown"

    # Reorder to match training
    df = df[selected_features]

    # Factorize categoricals exactly like training (simple factorize; same process at serve-time)
    # NOTE: For fully stable behavior, persist encoders per column. This mirrors your current training flow.
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        df[c] = df[c].fillna("Unknown")
        df[c] = pd.factorize(df[c])[0]

    return df

# ---------- Predict helpers ----------
def _predict_proba_multioutput(multi_model, X_np: np.ndarray) -> np.ndarray:
    """Return (n_samples, n_labels) probability matrix."""
    y_pred_proba = multi_model.predict_proba(X_np)
    if isinstance(y_pred_proba, list):
        # list of length n_labels, each (n_samples, 2)
        probs = np.column_stack([p[:, 1] for p in y_pred_proba])
    else:
        # Some wrappers may return array directly
        probs = y_pred_proba
    return probs

def _topk_from_probs(probs_row: np.ndarray, k: int) -> List[Dict[str, Any]]:
    idx = np.argsort(probs_row)[-k:][::-1]
    return [
        {"rank": i+1, "label": str(label_names[j]), "probability": float(probs_row[j])}
        for i, j in enumerate(idx)
    ]

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "n_features_expected": len(selected_features), "n_labels": len(label_names)}

@app.get("/meta/features")
def meta_features():
    return {"selected_features": selected_features, "labels": list(map(str, label_names))}

@app.post("/recommend/topk")
def recommend_topk(req: RecommendationRequest, k: int = Query(3, ge=1, le=10)):
    try:
        df = build_feature_row(req)
        X_scaled = scaler.transform(df.values)
        probs = _predict_proba_multioutput(model, X_scaled)
        topk = _topk_from_probs(probs[0], k)
        return {
            "top_k": topk,
            "debug": {
                "feature_vector_shape": list(df.shape),
                "feature_order": selected_features,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
