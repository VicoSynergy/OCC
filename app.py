"""
FastAPI Product Subcategory Recommendation Service
Processes raw client data and returns top-3 product subcategory recommendations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Product Subcategory Recommendation API",
    description="ML-powered product subcategory recommendations for financial advisors",
    version="1.0.0"
)

# Global model variable
model_package = None

# ===== DATA MODELS (REQUEST/RESPONSE SCHEMAS) =====

class ClientData(BaseModel):
    """Raw client data input schema - matches your preprocessing pipeline"""
    
    # Core Demographics
    ClientId: str
    ClientAge: Optional[int] = Field(None, ge=18, le=100, description="Client age in years")
    ClientGender: Optional[str] = Field(None, description="Male/Female")
    Nationality: Optional[str] = Field(None, description="Client nationality")
    MaritalStatus: Optional[str] = Field(None, description="Single/Married/Divorced/Widowed")
    
    # Socioeconomic
    IncomeRange: Optional[str] = Field(None, description="No Income/Below S$30,000/S$30,000 - S$49,999/S$50,000 - S$99,999/S$100,000 and above")
    Education: Optional[str] = Field(None, description="Education level")
    EmploymentStatus: Optional[str] = Field(None, description="Employment status")
    RiskProfile: Optional[str] = Field(None, description="Conservative/Moderate/Aggressive/etc")
    
    # Financial Assets (raw values before aggregation)
    SavingsAccounts: Optional[float] = Field(None, ge=0, description="Savings account balance")
    FixedDepositsAccount: Optional[float] = Field(None, ge=0, description="Fixed deposit balance")
    StocksPortofolio: Optional[float] = Field(None, ge=0, description="Stock portfolio value")
    BondPortofolio: Optional[float] = Field(None, ge=0, description="Bond portfolio value")
    UTFEquityAsset: Optional[float] = Field(None, ge=0, description="Unit trust equity assets")
    ETFs: Optional[float] = Field(None, ge=0, description="ETF holdings")
    InvestmentProperties: Optional[float] = Field(None, ge=0, description="Investment property value")
    CPFOABalance: Optional[float] = Field(None, ge=0, description="CPF Ordinary Account")
    CPFSABalance: Optional[float] = Field(None, ge=0, description="CPF Special Account")
    CPFMABalance: Optional[float] = Field(None, ge=0, description="CPF Medisave Account")
    
    # Insurance Portfolio (raw values)
    Total_Policies: Optional[int] = Field(None, ge=0, description="Number of insurance policies")
    Total_Life_Coverage: Optional[float] = Field(None, ge=0, description="Total life insurance coverage")
    Total_CI_Coverage: Optional[float] = Field(None, ge=0, description="Total critical illness coverage")
    Total_Annual_Premium: Optional[float] = Field(None, ge=0, description="Total annual insurance premium")

class ProductRecommendation(BaseModel):
    """Individual product recommendation"""
    Id: str = Field(description="Product subcategory ID")
    Label: str = Field(description="Product subcategory name")
    Confidence: float = Field(description="Confidence score (0-1)")

class RecommendationResponse(BaseModel):
    """API response schema"""
    clientId: str
    recProductSubCatId: List[str] = Field(description="List of recommended subcategory IDs")
    recProductSubCatLabel: List[str] = Field(description="List of recommended subcategory labels")
    recommendations: List[ProductRecommendation] = Field(description="Detailed recommendations with confidence")
    processingTime: float = Field(description="Processing time in seconds")
    timestamp: str = Field(description="Prediction timestamp")

# ===== FEATURE ENGINEERING FUNCTIONS =====

def create_derived_features(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create derived features from raw client data
    Matches your preprocessing pipeline exactly
    """
    
    features = client_data.copy()
    
    # Income processing (numeric conversion)
    income_mapping = {
        'No Income': 0,
        'Below S$30,000': 15000,
        'S$30,000 - S$49,999': 40000,
        'S$50,000 - S$99,999': 75000,
        'S$100,000 and above': 150000
    }
    
    income_range = features.get('IncomeRange')
    features['Income_Numeric'] = income_mapping.get(income_range, 0)
    
    # Asset aggregations (handle missing values with 0)
    liquid_assets = [
        features.get('SavingsAccounts', 0) or 0,
        features.get('FixedDepositsAccount', 0) or 0
    ]
    features['Total_Liquid_Assets'] = sum(liquid_assets)
    
    investment_assets = [
        features.get('StocksPortofolio', 0) or 0,
        features.get('BondPortofolio', 0) or 0,
        features.get('UTFEquityAsset', 0) or 0,
        features.get('ETFs', 0) or 0
    ]
    features['Total_Investments'] = sum(investment_assets)
    
    cpf_assets = [
        features.get('CPFOABalance', 0) or 0,
        features.get('CPFSABalance', 0) or 0,
        features.get('CPFMABalance', 0) or 0
    ]
    features['Total_CPF'] = sum(cpf_assets)
    
    # Net worth calculation
    wealth_components = [
        features['Total_Liquid_Assets'],
        features['Total_Investments'],
        features['Total_CPF'],
        features.get('InvestmentProperties', 0) or 0
    ]
    features['Estimated_Net_Worth'] = sum(wealth_components)
    
    # Investment ratio
    total_financial = features['Total_Liquid_Assets'] + features['Total_Investments']
    if total_financial > 0:
        features['Investment_Ratio'] = features['Total_Investments'] / total_financial
    else:
        features['Investment_Ratio'] = 0
    
    # Insurance features
    features['Has_Insurance'] = 1 if (features.get('Total_Policies', 0) or 0) > 0 else 0
    
    # Life coverage multiple
    if features['Income_Numeric'] > 0:
        features['Life_Coverage_Multiple'] = (features.get('Total_Life_Coverage', 0) or 0) / features['Income_Numeric']
    else:
        features['Life_Coverage_Multiple'] = 0
    
    # Premium to income ratio
    if features['Income_Numeric'] > 0:
        features['Premium_to_Income_Ratio'] = (features.get('Total_Annual_Premium', 0) or 0) / features['Income_Numeric']
    else:
        features['Premium_to_Income_Ratio'] = 0
    
    # Coverage indicators
    features['Has_Life_Coverage'] = 1 if (features.get('Total_Life_Coverage', 0) or 0) > 0 else 0
    features['Has_CI_Coverage'] = 1 if (features.get('Total_CI_Coverage', 0) or 0) > 0 else 0
    
    # Coverage gaps
    if features['Has_Insurance'] == 1 and features['Has_CI_Coverage'] == 0:
        features['CI_Coverage_Gap'] = 1
    else:
        features['CI_Coverage_Gap'] = 0
    
    # Insurance sophistication
    policies = features.get('Total_Policies', 0) or 0
    life_cov = features['Has_Life_Coverage']
    ci_cov = features['Has_CI_Coverage']
    
    if policies == 0:
        features['Insurance_Sophistication'] = 'No_Insurance'
    else:
        coverage_types = life_cov + ci_cov
        if coverage_types >= 2:
            features['Insurance_Sophistication'] = 'Moderate'
        else:
            features['Insurance_Sophistication'] = 'Basic'
    
    # Life stage (simplified logic)
    age = features.get('ClientAge')
    marital = features.get('MaritalStatus', '').lower()
    
    if age and age < 35:
        features['Life_Stage'] = 'Young_Single' if 'single' in marital else 'Young_Family'
    elif age and age < 55:
        features['Life_Stage'] = 'Mid_Career_Single' if 'single' in marital else 'Mid_Career_Family'
    else:
        features['Life_Stage'] = 'Pre_Retirement'
    
    return features

def prepare_model_input(client_data: Dict[str, Any], model_package: Dict) -> pd.DataFrame:
    """
    Prepare model input features with proper encoding
    """
    
    # Create derived features
    features = create_derived_features(client_data)
    
    # Get required features
    top_features = model_package['top_features']
    feature_encoders = model_package['feature_encoders']
    feature_info = model_package['feature_info']
    
    # Create feature matrix
    X = pd.DataFrame(index=[0])
    
    for feature in top_features:
        if feature in features:
            value = features[feature]
            
            # Handle encoding based on feature type
            if feature in feature_info:
                feature_type = feature_info[feature]['type']
                
                if feature_type == 'categorical':
                    # Use label encoder
                    encoder = feature_encoders.get(feature)
                    if encoder and value is not None:
                        try:
                            # Handle unknown categories
                            if str(value) in encoder.classes_:
                                X[feature] = encoder.transform([str(value)])[0]
                            else:
                                # Use most common class or 0
                                X[feature] = 0
                        except:
                            X[feature] = 0
                    else:
                        X[feature] = 0
                
                elif feature_type == 'numerical':
                    # Use median for missing values
                    if value is not None:
                        X[feature] = float(value)
                    else:
                        median_val = feature_info[feature].get('median', 0)
                        X[feature] = median_val
                
                elif feature_type == 'binary':
                    X[feature] = int(value) if value is not None else 0
                
                else:
                    X[feature] = value if value is not None else 0
            else:
                # Default handling
                X[feature] = value if value is not None else 0
        else:
            # Feature not provided, use default
            X[feature] = 0
    
    return X

# ===== SUBCATEGORY MAPPING =====
# This should match your actual subcategory to ID mapping
SUBCATEGORY_ID_MAPPING = {
    'Long_Term_Care_Plans': 'LTC001',
    'SHIELD': 'SHD001', 
    'Term': 'TRM001',
    'Investment_Linked_Plan___Accumulation': 'ILP001',
    'Whole_Life': 'WHL001',
    'Accident_and_Health_Plans': 'AHP001',
    'Critical_Illness_Plans': 'CIP001',
    'ENDOW_NP': 'END001',
    'Retirement': 'RET001',
    'Policy_Servicing_Non_Shield_Plans': 'PSV001',
    'Add_Rider_to_existing_shield_plans': 'RDR001',
    'Endowment': 'EDW001',
    'Whole_Life_Income': 'WLI001',
    'Universal_Life_Protection': 'ULP001',
    'Discretionary_Managed_Account': 'DMA001'
}

def get_subcategory_id(subcategory_label: str) -> str:
    """Convert subcategory label to ID"""
    return SUBCATEGORY_ID_MAPPING.get(subcategory_label, f"UNK_{hash(subcategory_label) % 1000}")

# ===== PREDICTION FUNCTION =====

def predict_top_subcategories(client_features: pd.DataFrame, model_package: Dict, k: int = 3) -> List[tuple]:
    """
    Predict top-k subcategories with confidence scores
    """
    
    model = model_package['model']
    target_encoder = model_package['target_encoder']
    top_features = model_package['top_features']
    
    # Ensure features are in correct order
    X_pred = client_features[top_features]
    
    # Get prediction probabilities
    pred_proba = model.predict_proba(X_pred)[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(pred_proba)[-k:][::-1]
    top_k_subcategories = target_encoder.inverse_transform(top_k_indices)
    top_k_confidences = pred_proba[top_k_indices]
    
    return [(subcat, conf) for subcat, conf in zip(top_k_subcategories, top_k_confidences)]

# ===== API STARTUP =====

@app.on_event("startup")
async def load_model():
    """Load the ML model on startup"""
    global model_package
    
    try:
        model_package = joblib.load('SUBCATEGORY_PREDICTION_MODEL_V1.pkl')
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model type: {model_package.get('model_name', 'Unknown')}")
        logger.info(f"Features: {len(model_package['top_features'])}")
        logger.info(f"Classes: {len(model_package['target_encoder'].classes_)}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Product Subcategory Recommendation API",
        "status": "healthy",
        "model_loaded": model_package is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_package is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_package.get('model_name', 'Unknown'),
        "training_date": model_package.get('training_date', 'Unknown'),
        "performance": model_package.get('model_performance', {}),
        "features": model_package['top_features'],
        "target_classes": list(model_package['target_encoder'].classes_),
        "total_classes": len(model_package['target_encoder'].classes_)
    }

@app.post("/predict", response_model=RecommendationResponse)
async def predict_subcategories(client_data: ClientData):
    """
    Main prediction endpoint - returns top-3 subcategory recommendations
    """
    
    start_time = datetime.now()
    
    if model_package is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict
        client_dict = client_data.dict()
        
        # Prepare model input
        X_pred = prepare_model_input(client_dict, model_package)
        
        # Make predictions
        predictions = predict_top_subcategories(X_pred, model_package, k=3)
        
        # Format response
        rec_ids = []
        rec_labels = []
        detailed_recommendations = []
        
        for subcategory, confidence in predictions:
            subcategory_id = get_subcategory_id(subcategory)
            
            rec_ids.append(subcategory_id)
            rec_labels.append(subcategory)
            
            detailed_recommendations.append(ProductRecommendation(
                Id=subcategory_id,
                Label=subcategory,
                Confidence=round(float(confidence), 4)
            ))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = RecommendationResponse(
            clientId=client_data.ClientId,
            recProductSubCatId=rec_ids,
            recProductSubCatLabel=rec_labels,
            recommendations=detailed_recommendations,
            processingTime=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Prediction completed for client {client_data.ClientId} in {processing_time:.3f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction failed for client {client_data.ClientId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(clients: List[ClientData]):
    """
    Batch prediction endpoint for multiple clients
    """
    
    if model_package is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for client_data in clients:
        try:
            # Reuse single prediction logic
            result = await predict_subcategories(client_data)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed prediction for client {client_data.ClientId}: {e}")
            # Add error result
            results.append({
                "clientId": client_data.ClientId,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    return {
        "total_clients": len(clients),
        "successful_predictions": len([r for r in results if "error" not in r]),
        "failed_predictions": len([r for r in results if "error" in r]),
        "results": results
    }

# ===== EXAMPLE USAGE =====

@app.get("/example")
async def get_example_request():
    """Get example client data for testing"""
    return {
        "example_client": {
            "ClientId": "CLIENT_12345",
            "ClientAge": 35,
            "ClientGender": "Male",
            "Nationality": "Singapore",
            "MaritalStatus": "Married",
            "IncomeRange": "S$50,000 - S$99,999",
            "Education": "University",
            "EmploymentStatus": "Employed",
            "RiskProfile": "Moderate",
            "SavingsAccounts": 50000,
            "FixedDepositsAccount": 30000,
            "StocksPortofolio": 25000,
            "CPFOABalance": 40000,
            "CPFSABalance": 20000,
            "CPFMABalance": 15000,
            "Total_Policies": 2,
            "Total_Life_Coverage": 500000,
            "Total_CI_Coverage": 200000,
            "Total_Annual_Premium": 5000
        }
    }

# ===== RUN SERVER =====

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )