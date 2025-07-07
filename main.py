import os
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality API",
    description="API for air quality prediction and anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create model directories if they don't exist
models_dir = Path("Saved Models")
models_dir.mkdir(exist_ok=True)


# Define Pydantic models for request/response

# Model 1 - AQI Prediction Input schema
class AQIPredictionRequest(BaseModel):
    """Input schema for Air Quality Index prediction model
    
    Attributes:
        location: Location identifier (city or station name)
        date: Date of measurement (format: YYYY-MM-DD)
        time: Time of measurement (format: HH:MM)
        CO_GT: Carbon Monoxide ground truth value (mg/m³)
        C6H6_GT: Benzene ground truth value (µg/m³)
        PT08_S2_NMHC: Tin oxide sensor response for Non-Methanic HydroCarbons
        T: Temperature in °C
        RH: Relative Humidity %
        AH: Absolute Humidity g/m³
    """
    location: str = Field(..., description="Location identifier (city or station name)")
    date: str = Field(..., description="Date of measurement (format: YYYY-MM-DD)")
    time: str = Field(..., description="Time of measurement (format: HH:MM)")
    CO_GT: float = Field(..., ge=0, description="Carbon Monoxide ground truth value (mg/m³)")
    C6H6_GT: float = Field(..., ge=0, description="Benzene ground truth value (µg/m³)")
    PT08_S2_NMHC: float = Field(..., ge=0, description="Tin oxide sensor response for Non-Methanic HydroCarbons")
    T: float = Field(..., ge=-50, le=60, description="Temperature in °C")
    RH: float = Field(..., ge=0, le=100, description="Relative Humidity %")
    AH: float = Field(..., ge=0, le=30, description="Absolute Humidity g/m³")

# Model 1 - AQI Prediction Response schema
class AQIPredictionResponse(BaseModel):
    """Response schema for AQI predictions"""
    air_quality_index: float
    pollutant_levels: Dict[str, float]
    health_recommendation: str
    forecast_values: Optional[List[Dict[str, float]]] = None

# Model 2 - NO2 Prediction Input schema
class NO2PredictionRequest(BaseModel):
    """Input schema for NO2 prediction model
    
    Attributes:
        CO_GT: Carbon Monoxide ground truth value (mg/m³)
        C6H6_GT: Benzene ground truth value (µg/m³)
        PT08_S2_NMHC: Tin oxide sensor response for Non-Methanic HydroCarbons
        NOx_GT: Nitrogen oxides ground truth value (ppb)
        PT08_S3_NOx: Tungsten oxide sensor response for NOx
        PT08_S4_NO2: Tungsten oxide sensor response for NO2
        PT08_S5_O3: Indium oxide sensor response for O3
        T: Temperature in °C
        RH: Relative Humidity %
        AH: Absolute Humidity g/m³
    """
    CO_GT: float = Field(..., ge=0, description="Carbon Monoxide ground truth value (mg/m³)")
    C6H6_GT: float = Field(..., ge=0, description="Benzene ground truth value (µg/m³)")
    PT08_S2_NMHC: float = Field(..., ge=0, description="Tin oxide sensor response for Non-Methanic HydroCarbons")
    NOx_GT: float = Field(..., ge=0, description="Nitrogen oxides ground truth value (ppb)")
    PT08_S3_NOx: float = Field(..., ge=0, description="Tungsten oxide sensor response for NOx")
    PT08_S4_NO2: float = Field(..., ge=0, description="Tungsten oxide sensor response for NO2")
    PT08_S5_O3: float = Field(..., ge=0, description="Indium oxide sensor response for O3")
    T: float = Field(..., ge=-50, le=60, description="Temperature in °C")
    RH: float = Field(..., ge=0, le=100, description="Relative Humidity %")
    AH: float = Field(..., ge=0, le=30, description="Absolute Humidity g/m³")

# Model 2 - NO2 Prediction Response schema
class NO2PredictionResponse(BaseModel):
    no2_prediction: float
    model_details: Dict

# Model 3 - Anomaly Detection Input schema
class AnomalyDetectionRequest(BaseModel):
    """Input schema for air quality anomaly detection
    
    Attributes:
        air_quality_data: List of air quality measurements
        detection_method: Algorithm to use for anomaly detection
    """
    air_quality_data: List[Dict[str, float]]
    detection_method: str = Field(
        "isolation_forest", 
        description="Detection method: 'isolation_forest' or 'dbscan'"
    )

# Model 3 - Anomaly Detection Response schema
class AnomalyDetectionResponse(BaseModel):
    """Response schema for anomaly detection results"""
    anomalies: List[bool]
    anomaly_indices: List[int]
    anomaly_percentage: float
    pca_visualization: Optional[List[Dict[str, float]]] = None


# API Routes

# Model 1 - AQI Prediction endpoints
@app.post("/predict/aqi", response_model=AQIPredictionResponse, tags=["Model 1 - AQI Prediction"])
async def predict_aqi(request: AQIPredictionRequest):
    """
    Predict air quality index and pollutant levels.
    
    This model uses location, time, and basic air quality metrics to predict overall air quality.
    
    Note: This is a mock implementation for testing purposes only.
    """
    try:
        # Calculate mock AQI based on input values (simple formula for testing)
        mock_aqi = (10 * request.CO_GT + 5 * request.C6H6_GT - 0.2 * request.T + 0.1 * request.RH + 20)
        
        # Generate health recommendation based on AQI
        health_rec = "Good air quality, no special precautions needed."
        if mock_aqi > 100:
            health_rec = "Moderate air quality. Sensitive individuals should limit prolonged outdoor activity."
        if mock_aqi > 150:
            health_rec = "Unhealthy air quality. Everyone should reduce outdoor activities."
        
        # Create mock pollutant predictions
        pollutants = {
            "PM2.5": round(request.C6H6_GT * 2.5 + 5, 2),
            "PM10": round(request.C6H6_GT * 4 + 10, 2),
            "O3": round(request.T * 2 - request.RH / 10 + 30, 2),
            "NO2": round(request.CO_GT * 10 + 5, 2),
            "SO2": round(request.C6H6_GT * 1.5 + 2, 2)
        }
        
        # Create mock forecast (next 24 hours, at 6-hour intervals)
        forecasts = []
        for hour in [6, 12, 18, 24]:
            # Create simple variations of current values for the forecast
            variation = (hour / 24.0) * 0.2  # 0-20% variation
            forecasts.append({
                "hours_ahead": hour,
                "aqi": round(mock_aqi * (1 + np.sin(hour/4) * variation), 2),
                "PM2.5": round(pollutants["PM2.5"] * (1 + np.cos(hour/5) * variation), 2),
                "O3": round(pollutants["O3"] * (1 + np.sin(hour/3) * variation), 2),
            })
        
        return AQIPredictionResponse(
            air_quality_index=float(mock_aqi),
            pollutant_levels=pollutants,
            health_recommendation=health_rec,
            forecast_values=forecasts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AQI prediction error: {str(e)}")

@app.get("/predict/aqi/result", response_model=AQIPredictionResponse, tags=["Model 1 - AQI Prediction"])
async def get_aqi_result():
    """
    Get the most recent AQI prediction result.
    
    This endpoint returns a sample AQI prediction for demonstration purposes.
    In a real implementation, it would return the most recent prediction from storage.
    """
    try:
        # Generate sample data for demonstration
        mock_aqi = 85.5
        
        # Generate health recommendation based on AQI
        health_rec = "Good air quality, no special precautions needed."
        if mock_aqi > 100:
            health_rec = "Moderate air quality. Sensitive individuals should limit prolonged outdoor activity."
        if mock_aqi > 150:
            health_rec = "Unhealthy air quality. Everyone should reduce outdoor activities."
        
        # Create sample pollutant predictions
        pollutants = {
            "PM2.5": 24.75,
            "PM10": 38.0,
            "O3": 54.5,
            "NO2": 32.5,
            "SO2": 12.25
        }
        
        # Create mock forecast
        forecasts = []
        for hour in [6, 12, 18, 24]:
            variation = (hour / 24.0) * 0.2
            forecasts.append({
                "hours_ahead": hour,
                "aqi": round(mock_aqi * (1 + np.sin(hour/4) * variation), 2),
                "PM2.5": round(pollutants["PM2.5"] * (1 + np.cos(hour/5) * variation), 2),
                "O3": round(pollutants["O3"] * (1 + np.sin(hour/3) * variation), 2),
            })
        
        return AQIPredictionResponse(
            air_quality_index=float(mock_aqi),
            pollutant_levels=pollutants,
            health_recommendation=health_rec,
            forecast_values=forecasts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving AQI result: {str(e)}")

# Model 2 - NO2 Prediction endpoints
@app.post("/predict/no2", response_model=NO2PredictionResponse, tags=["Model 2 - NO2 Prediction"])
async def predict_no2(request: NO2PredictionRequest):
    """
    Predict NO2 concentration using mock implementation.
    
    Requires air quality metrics as input and returns predicted NO2 value.
    
    Note: This is a mock implementation for testing purposes only.
    """
    try:
        # Create a mock prediction based on input features
        # In a real implementation, this would use a trained model
        mock_prediction = (
            0.1 * request.CO_GT + 
            0.05 * request.C6H6_GT + 
            0.001 * request.PT08_S2_NMHC + 
            0.02 * request.NOx_GT + 
            0.003 * request.PT08_S3_NOx - 
            0.002 * request.PT08_S4_NO2 + 
            0.001 * request.PT08_S5_O3 - 
            0.5 * request.T + 
            0.1 * request.RH + 
            5 * request.AH + 
            random.normalvariate(0, 5)
        )
        
        return NO2PredictionResponse(
            no2_prediction=float(mock_prediction),
            model_details={
                "model_type": "Linear Regression (Mock)",
                "target_variable": "NO2(GT)"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/predict/no2/result", response_model=NO2PredictionResponse, tags=["Model 2 - NO2 Prediction"])
async def get_no2_result():
    """
    Get the most recent NO2 prediction result.
    
    This endpoint returns a sample NO2 prediction for demonstration purposes.
    In a real implementation, it would return the most recent prediction from storage.
    """
    try:
        # Generate a sample NO2 prediction
        mock_prediction = 52.7
        
        return NO2PredictionResponse(
            no2_prediction=float(mock_prediction),
            model_details={
                "model_type": "Linear Regression (Mock)",
                "target_variable": "NO2(GT)",
                "confidence": 0.85,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving NO2 result: {str(e)}")

# Model 3 - Anomaly Detection endpoints
@app.post("/detect/anomalies", response_model=AnomalyDetectionResponse, tags=["Model 3 - Anomaly Detection"])
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalies in air quality data using mock implementation.
    
    Returns anomaly flags and visualization data.
    
    Note: This is a mock implementation for testing purposes only.
    """
    try:
        # Generate mock anomaly detection results
        # In a real implementation, this would use a trained model
        
        # Count data points
        n_samples = len(request.air_quality_data)
        
        # Mark approximately 10% of the data as anomalies
        # In a more realistic implementation, this would be based on actual anomaly detection
        anomalies = [False] * n_samples
        anomaly_count = max(1, int(n_samples * 0.1))  # At least 1 anomaly
        
        # Select random indices for anomalies
        anomaly_indices = random.sample(range(n_samples), anomaly_count)
        for idx in anomaly_indices:
            anomalies[idx] = True
        
        # Calculate anomaly percentage
        anomaly_percentage = (sum(anomalies) / len(anomalies)) * 100
        
        # Generate mock PCA visualization
        pca_viz_data = []
        for i in range(n_samples):
            # Generate random 2D coordinates for visualization
            # In a real implementation, this would be from actual PCA
            is_anomaly = anomalies[i]
            
            # Make anomalies stand out in the visualization
            if is_anomaly:
                pc1 = random.uniform(3, 5)  # Anomalies placed far from center
                pc2 = random.uniform(3, 5)
                if random.random() > 0.5:
                    pc1 *= -1
                if random.random() > 0.5:
                    pc2 *= -1
            else:
                # Normal points clustered near center
                pc1 = random.normalvariate(0, 1)
                pc2 = random.normalvariate(0, 1)
            
            pca_viz_data.append({
                "PC1": pc1,
                "PC2": pc2,
                "is_anomaly": is_anomaly
            })
        
        return AnomalyDetectionResponse(
            anomalies=anomalies,
            anomaly_indices=anomaly_indices,
            anomaly_percentage=anomaly_percentage,
            pca_visualization=pca_viz_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")

@app.get("/detect/anomalies/result", response_model=AnomalyDetectionResponse, tags=["Model 3 - Anomaly Detection"])
async def get_anomaly_results():
    """
    Get the most recent anomaly detection results.
    
    This endpoint returns sample anomaly detection results for demonstration purposes.
    In a real implementation, it would return the most recent analysis from storage.
    """
    try:
        # Generate sample data
        n_samples = 20
        
        # Create anomalies (approximately 15% of data points)
        anomalies = [False] * n_samples
        anomaly_indices = [3, 12, 17]  # Fixed anomaly positions for consistent results
        
        for idx in anomaly_indices:
            anomalies[idx] = True
        
        # Calculate anomaly percentage
        anomaly_percentage = (sum(anomalies) / len(anomalies)) * 100
        
        # Generate PCA visualization
        pca_viz_data = []
        for i in range(n_samples):
            is_anomaly = anomalies[i]
            
            if is_anomaly:
                # Anomalies are placed far from center
                pc1 = 4.0 if i % 2 == 0 else -4.0
                pc2 = 3.5 if i % 3 == 0 else -3.5
            else:
                # Normal points clustered near center with some variance
                pc1 = (i % 5) * 0.2
                pc2 = (i % 7) * 0.15
            
            pca_viz_data.append({
                "PC1": pc1,
                "PC2": pc2,
                "is_anomaly": is_anomaly
            })
        
        return AnomalyDetectionResponse(
            anomalies=anomalies,
            anomaly_indices=anomaly_indices,
            anomaly_percentage=anomaly_percentage,
            pca_visualization=pca_viz_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving anomaly results: {str(e)}")

# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Global handler for validation errors that provides more context in the response.
    """
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": error_details,
            "body": exc.body,
            "message": "Input validation error. Please check your request data."
        },
    )

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - displays basic API information and available endpoints
    """
    return {
        "message": "Welcome to the Air Quality API",
        "documentation": "/docs",
        "endpoints": {
            "Model 1 - AQI Prediction (POST)": "/predict/aqi",
            "Model 1 - AQI Results (GET)": "/predict/aqi/result",
            "Model 2 - NO2 Prediction (POST)": "/predict/no2",
            "Model 2 - NO2 Results (GET)": "/predict/no2/result",
            "Model 3 - Anomaly Detection (POST)": "/detect/anomalies",
            "Model 3 - Anomaly Results (GET)": "/detect/anomalies/result"
        }
    }

# Run the application with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    print("Starting Air Quality API Server...")
    print("API documentation will be available at http://localhost:8000/docs")
    print("Server is accessible from other devices on the network at http://<your-ip-address>:8000")
    print("Using host 0.0.0.0 to allow external connections")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
