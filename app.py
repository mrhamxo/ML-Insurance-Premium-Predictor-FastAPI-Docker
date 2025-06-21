from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, MODEL_VERSION, model
    
# Create FastAPI instance
app = FastAPI()

# -------------------- Define City Tiers --------------------

# Tier 1 cities (metro)
tier_1_cities = ['Islamabad', 'Karachi', 'Lahore', 'Peshawar', 'Quetta', 'Rawalpindi', 'Faisalabad']

# Tier 2 cities (developed but smaller than Tier 1)
tier_2_cities = [
    'Multan', 'Gujranwala', 'Hyderabad', 'Sialkot', 'Bahawalpur', 'Sargodha', 'Sukkur', 'Larkana', 'Sheikhupura',
    'Abbottabad', 'Jhelum', 'Gujrat', 'Mardan', 'Kasur', 'Okara', 'Sahiwal', 'Turbat', 'Mingora', 'Nawabshah', 
    'Chiniot', 'Kohat', 'Muzaffarabad', 'Gilgit', 'Kotli', 'Skardu', 'Khuzdar', 'Bannu', 'Gwadar', 'Jhang', 'Hafizabad',
    'Kamoke', 'Jacobabad', 'Shikarpur', 'Charsadda', 'Mansehra', 'Narowal', 'Vehari', 'Layyah', 'Attock', 'Lodhran',
    'Badin', 'Khanewal', 'Bhakkar', 'Haripur', 'Swabi', 'Jamshoro', 'Gojra', 'Chakwal', 'Jaranwala', 'Khanpur', 'Kamalia',
    'Daska', 'Nowshera', 'Thatta', 'Pakpattan', 'Jaccobabad', 'Samundri', 'Muridke', 'Mianwali', 'Kandhkot', 'Shahdadpur', 
    'Shahkot', 'Arifwala', 'Pattoki', 'Shikarpur', 'Hangu', 'Charsadda', 'Burewala', 'Jatoi',
]

# -------------------- Prediction Endpoint --------------------

# human readable endpoints
@app.get("/")
def home():
    return JSONResponse(status_code=200, content={"message": "Welcome to the Health Insurance Premium Prediction API!"})

# machine readable endpoints
@app.get("/health-check", name="Health Check Endpoint")
def health_check():
    return JSONResponse(status_code=200, content={
        "satus": "API Health check is OK!",
        "version": MODEL_VERSION,
        "model_loaded": model is not None,
    })

@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    # Construct DataFrame with engineered features
    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'city': data.city,
        'occupation': data.occupation 
    }
    
    try:
        # Make prediction using the ML model
        prediction = predict_output(user_input)
        
        # Return prediction as JSON response
        return JSONResponse(status_code=200, content={'response': prediction})
    
    except Exception as e:
        
        return JSONResponse(status_code=500, content=str(e))
