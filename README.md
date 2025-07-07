# Air Quality API with Streamlit UI

This project provides an Air Quality prediction and analysis API built with FastAPI, along with a Streamlit user interface for interaction.

## Features

- **Air Quality Index (AQI) Prediction**: Predict AQI values based on various environmental metrics
- **NO2 Concentration Prediction**: Predict NO2 levels using sensor data and environmental factors
- **Anomaly Detection**: Identify unusual patterns in time-series air quality data
- **Streamlit UI**: Modern, interactive interface for accessing all API functionality
- **API Documentation**: Auto-generated Swagger/ReDoc documentation

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Start the FastAPI Server

```
cd air-quality-prediction
python main.py
```

The server will start on `http://127.0.0.1:8000`. You can access:
- API documentation: `http://127.0.0.1:8000/docs`
- Alternative documentation: `http://127.0.0.1:8000/redoc`

### 2. Start the Streamlit UI

```
cd air-quality-prediction
streamlit run app.py
```

The Streamlit interface will open automatically in your browser at `http://localhost:8501`.

### 3. Running the Complete System

To use the complete system, you need to:
1. Start the FastAPI server in one terminal
2. Start the Streamlit UI in another terminal

This two-step process gives you more control over each component.

### Creating a Public Link

To make your UI accessible from anywhere on the internet:

Run `streamlit run app.py -- --server.enableCORS=false --server.enableXsrfProtection=false`

> **Important**: When using a public link, make sure to also run your FastAPI server with the `host="0.0.0.0"` setting (which is the default in our setup) to allow external connections.

## API Documentation

The FastAPI server automatically generates comprehensive API documentation:

### Swagger UI

Navigate to `http://127.0.0.1:8000/docs` in your browser to access the interactive Swagger UI documentation. You can:
- View all available endpoints and models
- Try out API calls directly from the browser
- See request/response schemas and examples

### ReDoc

For an alternative documentation view, navigate to `http://127.0.0.1:8000/redoc`.

## API Endpoints

The following endpoints are available:

### Model 1 - AQI Prediction
- `POST /predict/aqi` - Predict Air Quality Index

### Model 2 - NO2 Prediction
- `POST /predict/no2` - Predict NO2 concentration

### Model 3 - Anomaly Detection
- `POST /detect/anomalies` - Detect anomalies in air quality data

### Information Endpoints
- `GET /` - Root endpoint with API information

## Using the Streamlit Interface

The Streamlit interface provides an intuitive way to interact with the API:

1. **AQI Prediction Tab**: Predict air quality index and see forecasts
2. **NO2 Prediction Tab**: Predict NO2 concentration with visual gauge
3. **Anomaly Detection Tab**: Upload data to find anomalies with PCA visualization
4. **API Info Tab**: Get information about the API endpoints and models

Each tab has a "Load Sample Data" button to populate the form with example data.

## Notes

- The current implementation uses mock models for demonstration purposes
- In a production environment, actual ML models would replace these mocks
- The Gradio UI requires the FastAPI server to be running

## Troubleshooting

- If you see a "API Server Not Running" warning in Gradio, make sure the FastAPI server is running
- Check that the ports (8000 for FastAPI, 7860 for Gradio) are not already in use
- For any issues with dependencies, make sure to use the correct version of Python
