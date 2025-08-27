from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException, Query, Body
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
import joblib
from fastapi.middleware.cors import CORSMiddleware

# ---------- Load artifacts ----------
MODEL_PATH = "insurance_top_k_recommendation_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
LABELS_PATH = "label_names.pkl"
SELECTED_FEATURES_PATH = "selected_features.pkl"  

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],
)

PROTECTION_EXAMPLE: Dict[str, Any] = {
    "ClientGender": "Male",
    "Nationality": "Singaporean",
    "SpokenLanguage": "English,Mandarin",
    "WrittenLanguage": "English",
    "Education": "Tertiary",
    "EmploymentStatus": "Full-time",
    "Occupation": "Manager",
    "MaritalStatus": "Married",
    "IncomeRange": "S$50,000 - S$99,999",
    "RiskProfile": "Balanced",
    "CKAProfile": "Not Assessed",
    "CARProfile": "Not Assessed",
    "ClientResidentialStatus": "Singapore Citizen",
    "Race": "Chinese",
    "ClientAge": 42,
    "EMFCSubmitDate": "2025-03-15T00:00:00Z",
    "EMFC_Count": 3,
    "SavingsAccounts": 20000,
    "FixedDepositsAccount": 10000,
    "StocksPortofolio": 15000,
    "BondPortofolio": 2000,
    "UTFEquityAsset": 3000,
    "ETFs": 1000,
    "InvestmentProperties": 0,
    "CPFOABalance": 25000,
    "CPFSABalance": 10000,
    "CPFMABalance": 8000,
    "Total_Life_Coverage": 250000,
    "Total_CI_Coverage": 100000,
    "Total_Annual_Premium": 2400,
    "Plan_Types": ["Term","Integrated Shield"],
    "Insurance_Companies": ["AIA","Singlife"]
}

WEALTH_EXAMPLE: Dict[str, Any] = {
    "ClientGender":"Female","Nationality":"Singaporean","SpokenLanguage":"English",
    "WrittenLanguage":"English","Education":"Post Graduate","EmploymentStatus":"Full-time",
    "MaritalStatus":"Married","IncomeRange":"S$100,000 and above","RiskProfile":"Moderately Aggressive",
    "CKAProfile":"Pass","CARProfile":"Pass","ClientAge":37,
    "EMFCSubmitDate":"2025-01-10T00:00:00Z","EMFC_Count":5,
    "SavingsAccounts":40000,"FixedDepositsAccount":60000,"StocksPortofolio":120000,"BondPortofolio":25000,
    "UTFEquityAsset":45000,"ETFs":30000,"InvestmentProperties":0,
    "CPFOABalance":45000,"CPFSABalance":80000,"CPFMABalance":15000,
    "Total_Life_Coverage":400000,"Total_CI_Coverage":150000,"Total_Annual_Premium":6000,
    "Plan_Types":["Whole Life","Endowment","Integrated Shield"],"Insurance_Companies":["AIA","Prudential"]
}


# ---------- Pydantic schema (raw payload) ----------
# ---------- Pydantic schema (raw payload) ----------
class RecommendationRequest(BaseModel):
    # A) Demographics & status
    ClientGender: Optional[Literal["Male","Female","Other"]] = Field(
        None,
        description='Gender. One of: "Male", "Female", "Other".'
    )
    Nationality: Optional[str] = Field(
        None,
        description='Client nationality. Expected common country name, e.g. "Singaporean", "Indonesian".'
    )
    SpokenLanguage: Optional[str] = Field(
        None,
        description='Comma-separated languages. Title-cased names. E.g. "English,Mandarin" or "English".'
    )
    WrittenLanguage: Optional[str] = Field(
        None,
        description='Comma-separated languages client can write. E.g. "English" or "English,Bahasa".'
    )
    Education: Optional[str] = Field(
        None,
        description='Highest education. E.g. "Secondary", "Diploma", "Tertiary", "Post Graduate".'
    )
    EmploymentStatus: Optional[str] = Field(
        None,
        description='Employment status. E.g. "Full-time", "Part-time", "Self-employed", "Unemployed", "Retired", "Student".'
    )
    Occupation: Optional[str] = Field(
        None,
        description='Free-text occupation. E.g. "Manager", "Engineer", "Teacher", "Retired", etc.'
    )
    MaritalStatus: Optional[str] = Field(
        None,
        description='Marital status. E.g. "Single", "Married", "Divorced", "Widowed".'
    )
    IncomeRange: Optional[Literal[
        "No Income",
        "Below S$30,000",
        "S$30,000 - S$49,999",
        "S$50,000 - S$99,999",
        "S$100,000 and above"
    ]] = Field(
        None,
        description='Household/annual income band. One of the listed literals: "No Income","Below S$30,000","S$30,000 - S$49,999","S$50,000 - S$99,999","S$100,000 and above"'
    )
    RiskProfile: Optional[str] = Field(
        None,
        description='Investment risk profile. E.g. "Conservative", "Moderately Conservative", "Balanced", "Moderately Aggressive", "Aggressive", or "Not Assessed".'
    )
    CKAProfile: Optional[Literal["Pass","Not Pass","Not Assessed"]] = Field(
        None,
        description='Customer Knowledge Assessment (CKA) result. One of: "Pass", "Not Pass", "Not Assessed".'
    )
    CARProfile: Optional[Literal["Pass","Not Pass","Not Assessed"]] = Field(
        None,
        description='Customer Account Review (CAR) result. One of: "Pass", "Not Pass", "Not Assessed".'
    )
    ClientResidentialStatus: Optional[str] = Field(
        None,
        description='Residency. E.g. "Singapore Citizen", "Permanent Resident", "Work Pass", "Foreigner".'
    )
    Race: Optional[str] = Field(
        None,
        description='Self-declared race/ethnicity. E.g. "Chinese", "Malay", "Indian", "Others".'
    )
    ClientAge: Optional[float] = Field(
        None,
        ge=0,
        description='Client age, fill with integer'        
    )

    # B) Temporal & session
    EMFCSubmitDate: Optional[datetime] = Field(
        None,
        description='ISO 8601 datetime of latest FNA/EMFC submission. E.g. "2025-03-15T00:00:00Z".'
    )
    EMFC_Count: Optional[float] = Field(
        0, 
        ge=0,
        description='Number of FNA sessions client did with One Synergy'        
    )

    # C) Assets & balances 
    SavingsAccounts: Optional[float] = Field(0, ge=0)
    FixedDepositsAccount: Optional[float] = Field(0, ge=0)
    HomeAsset: Optional[float] = Field(0, ge=0)
    MotorAsset: Optional[float] = Field(0, ge=0)
    InsuranceCashValues: Optional[float] = Field(0, ge=0)
    StocksPortofolio: Optional[float] = Field(0, ge=0)
    BondPortofolio: Optional[float] = Field(0, ge=0)
    UTFEquityAsset: Optional[float] = Field(0, ge=0)
    ETFs: Optional[float] = Field(0, ge=0)
    InvestmentProperties: Optional[float] = Field(0, ge=0)
    CPFOABalance: Optional[float] = Field(0, ge=0)
    CPFSABalance: Optional[float] = Field(0, ge=0)
    CPFMABalance: Optional[float] = Field(0, ge=0)
    SRSEquityAsset: Optional[float] = Field(0, ge=0)

    # D) Coverage & premium totals (numeric)
    Total_Life_Coverage: Optional[float] = Field(0, ge=0)
    Total_CI_Coverage: Optional[float] = Field(0, ge=0)
    Total_Hospital_Income: Optional[float] = Field(0, ge=0)
    Total_LTC_Coverage: Optional[float] = Field(0, ge=0)
    Total_Annual_Premium: Optional[float] = Field(0, ge=0)

    # E) Portfolio composition
    Plan_Types: Optional[List[str]] = Field(
        default=None,
        description='List of existing plan types (case-sensitive matching not required). E.g. ["Term","Whole Life","Endowment","Integrated Shield","Investment-Linked","Annuity","Critical Illness","Early Stage Critical Illness","Disability","Long Term Care"].'
    )
    Insurance_Companies: Optional[List[str]] = Field(
        default=None,
        description='List of insurer names. E.g. ["AIA","Prudential","Singlife","Income","Allianz","FWD","Etiqa","Great Eastern","AIG"].'
    )

    model_config = ConfigDict(json_schema_extra={
        "examples": [PROTECTION_EXAMPLE, WEALTH_EXAMPLE]
    })


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

def _temporal_features(now: datetime,
                       submit: Optional[datetime]
                       ) -> Tuple[float, float]:
    """
    Returns:
        days_since_last_fna, months_since_last_fna
    """
    if submit is None:
        days_since = 0.0
    else:
        days_since = max(0.0, (now - submit).total_seconds() / 86400.0)

    months_since = days_since / 30.44
    return days_since, months_since


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
    days_since, months_since = _temporal_features(
        now, req.EMFCSubmitDate
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
        "Race": req.Race,
        "ClientAge": req.ClientAge,
        # temporal
        "Days_Since_Last_FNA": days_since,
        "Months_Since_Last_FNA": months_since,
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
                "CKAProfile","CARProfile","ClientResidentialStatus","Race",
                "Insurance_Companies","Income_Category","Age_Group","Life_Stage",
                "Financial_Sophistication"
            ) else "Unknown"

    # Reorder to match training
    df = df[selected_features]

    # Factorize categoricals exactly like training (simple factorize; same process at serve-time)
    # NOTE: For fully stable behavior, persist encoders per column. 
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

def build_body_descriptions() -> str:
    lines = []
    for name, field in RecommendationRequest.model_fields.items():
        desc = field.description or ""
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)

@app.post(
    "/recommend/topk",
    summary="Get top‑K product recommendations",
    description=(
        """
Consumes raw client profile/temporal/assets/coverage/portfolio, recomputes engineered features, and returns top‑K product labels with probabilities.
# Request Body Fields 

## Identity & Demographics
- **ClientGender** *(string)*: One of "Male", "Female", "Other".
- **Nationality** *(string)*: Common country name, e.g. "Singaporean", "Indonesian".
- **Race** *(string)*: e.g. "Chinese", "Malay", "Indian", "Others".
- **ClientAge** *(integer)*: Age in years.

## Languages & Communication
- **SpokenLanguage** *(string)*: Comma‑separated; title‑cased tokens. E.g. "English,Mandarin".
- **WrittenLanguage** *(string)*: Comma‑separated; e.g. "English" or "English,Bahasa".

## Education & Employment
- **Education** *(string)*: "Secondary", "Diploma", "Tertiary", "Post Graduate", etc.
- **EmploymentStatus** *(string)*: "Full-time", "Part-time", "Self-employed", "Unemployed", "Retired", "Student".
- **Occupation** *(string)*: Free text, e.g. "Manager", "Engineer", "Teacher".

## Residency & Currency
- **ClientResidentialStatus** *(string)*: "Singapore Citizen", "Permanent Resident", "Work Pass", "Foreigner".

## Financial Profile
- **IncomeRange** *(string)*: One of "No Income", "Below S$30,000", "S$30,000 - S$49,999", "S$50,000 - S$99,999", "S$100,000 and above".

### Assets (numbers; use request currency)
- **SavingsAccounts**, **FixedDepositsAccount**, **HomeAsset**, **MotorAsset**, **InsuranceCashValues**,  
  **StocksPortofolio**, **BondPortofolio**, **UTFEquityAsset**, **ETFs**, **InvestmentProperties**,  
  **CPFOABalance**, **CPFSABalance**, **CPFMABalance**, **SRSEquityAsset** *(number)*.

## Regulatory / Suitability Profiles
- **RiskProfile** *(string)*: "Conservative", "Moderately Conservative", "Balanced", "Moderately Aggressive", "Aggressive", or "Not Assessed".
- **CKAProfile** *(string)*: "Pass", "Not Pass", "Not Assessed".
- **CARProfile** *(string)*: "Pass", "Not Pass", "Not Assessed".

## Engagement / Temporal
- **EMFCSubmitDate** *(ISO 8601 datetime)*: Latest FNA/EMFC submission, e.g. "2025-03-15T00:00:00Z".
- **EMFC_Count** *(integer)*: Total completed FNA sessions.

## Existing Portfolio (Coverage & Insurers)
- **Total_Life_Coverage** *(number)*: Sum of life coverage (policy rows).
- **Total_CI_Coverage** *(number)*: Sum of critical illness coverage.
- **Total_Hospital_Income** *(number)*: Sum of hospital income coverage.
- **Total_LTC_Coverage** *(number)*: Sum of long‑term care coverage.
- **Total_Annual_Premium** *(number)*: Sum of annual cash premiums.
- **Plan_Types** *(array of string)*: Existing plan types, e.g.  
  ["Term","Whole Life","Endowment","Integrated Shield","Investment-Linked","Annuity","Critical Illness","Early Stage Critical Illness","Disability","Long Term Care"].
- **Insurance_Companies** *(array of string)*: Unique insurer company names present in the portfolio.

---
Notes:
- If a field is unknown, omit it or send null. The service applies training‑consistent defaults.
"""
    ),
    tags=["Recommendations"]
)
def recommend_topk(
    req: RecommendationRequest = Body(
        ...,
        examples={
            "protection_user": {
                "summary": "Protection‑heavy client",
                "description": "Client with basic protection and Shield; often gets CI or Term.",
                "value": PROTECTION_EXAMPLE
            },
            "wealth_user": {
                "summary": "Wealth/ILP‑oriented client",
                "description": "Higher income, investment holdings; tends to get ILP/Endowment/Retirement.",
                "value": WEALTH_EXAMPLE
            }
        }
    ),
    k: int = Query(3, ge=1, le=10, description="Number of recommendations to return.")
):
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
