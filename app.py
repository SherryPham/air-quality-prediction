"""
Streamlit UI for Air Quality API

This file creates a Streamlit interface that connects to the FastAPI server for:
1. AQI Prediction (Model 1)
2. NO2 Prediction (Model 2)
3. Anomaly Detection (Model 3)
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Air Quality API Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL - change this if your FastAPI server runs on a different address
API_BASE_URL = "http://127.0.0.1:8000"

# Uncomment and edit the line below when deploying with a public URL
# API_BASE_URL = "https://your-fastapi-public-url-here"

# Helper function to make API requests
def make_api_request(endpoint, data=None, method="GET"):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        else:  # POST
            response = requests.post(url, json=data)
        
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

# Check if the server is running
def check_api_status():
    try:
        # Use the root endpoint instead of /health
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

#############################################
# MODEL 1: AQI PREDICTION
#############################################

def aqi_prediction_tab():
    st.header("Model 1: Air Quality Index Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Location and time inputs
    with col1:
        location = st.text_input("Location (City/Station)", placeholder="e.g. Delhi")
        date = st.date_input("Date", value=datetime.now())
        time = st.time_input("Time", value=datetime.now())
    
    # Column 2: Pollutant inputs
    with col2:
        co_gt = st.number_input("CO (mg/m³)", value=1.0, format="%.2f", min_value=0.0)
        c6h6_gt = st.number_input("Benzene/C6H6 (μg/m³)", value=5.0, format="%.2f", min_value=0.0)
        pt08_s2_nmhc = st.number_input("PT08.S2(NMHC) Sensor", value=1000, min_value=0)
    
    # Column 3: Environmental inputs
    with col3:
        temperature = st.number_input("Temperature (°C)", value=25.0, format="%.1f")
        rh = st.number_input("Relative Humidity (%)", value=50.0, format="%.1f", min_value=0.0, max_value=100.0)
        ah = st.number_input("Absolute Humidity (g/m³)", value=1.2000, format="%.4f", min_value=0.0, max_value=30.0)
    
    # Submit button
    if st.button("Predict AQI", type="primary"):
        # Format the request data
        data = {
            "location": location,
            "date": date.strftime("%Y-%m-%d"),
            "time": time.strftime("%H:%M"),
            "CO_GT": float(co_gt),
            "C6H6_GT": float(c6h6_gt),
            "PT08_S2_NMHC": float(pt08_s2_nmhc),
            "T": float(temperature),
            "RH": float(rh),
            "AH": float(ah)
        }
        
        # Make API request
        with st.spinner("Predicting AQI..."):
            response = make_api_request("/predict/aqi", data, method="POST")
        
        if "error" in response:
            st.error(response["error"])
        else:
            # Display results
            col1, col2 = st.columns([1, 1])
            
            # Column 1: Results as text
            with col1:
                st.subheader("Air Quality Prediction Results")
                
                # AQI value with color
                aqi_value = response.get('air_quality_index', 0)
                aqi_color = "#4CAF50"  # green
                if aqi_value > 100:
                    aqi_color = "#FFEB3B"  # yellow
                if aqi_value > 150:
                    aqi_color = "#FF5722"  # red
                
                st.markdown(f"<h3 style='color:{aqi_color}'>Air Quality Index: {aqi_value:.1f}</h3>", unsafe_allow_html=True)
                st.markdown(f"**Health Recommendation:** {response.get('health_recommendation', 'N/A')}")
                
                # Pollutant levels
                st.subheader("Pollutant Levels")
                pollutant_df = pd.DataFrame({
                    "Pollutant": list(response.get('pollutant_levels', {}).keys()),
                    "Value": list(response.get('pollutant_levels', {}).values())
                })
                st.dataframe(pollutant_df, use_container_width=True)
            
            # Column 2: Forecast visualization
            with col2:
                if response.get('forecast_values'):
                    st.subheader("Air Quality Forecast (Next 24 Hours)")
                    forecasts = response['forecast_values']
                    forecast_df = pd.DataFrame([
                        {
                            "Hours Ahead": item.get('hours_ahead', i),
                            "AQI": item.get('aqi', 0),
                            "PM2.5": item.get('PM2.5', 0),
                            "O3": item.get('O3', 0),
                            "Type": "AQI"
                        } for i, item in enumerate(forecasts)
                    ])
                    
                    # Add PM2.5 and O3 rows
                    for pollutant in ["PM2.5", "O3"]:
                        for i, item in enumerate(forecasts):
                            forecast_df = pd.concat([forecast_df, pd.DataFrame([{
                                "Hours Ahead": item.get('hours_ahead', i),
                                "AQI": item.get('aqi', 0),
                                "PM2.5": item.get('PM2.5', 0),
                                "O3": item.get('O3', 0),
                                "Type": pollutant
                            }])])
                    
                    # Create and display the chart
                    fig = px.line(
                        forecast_df, 
                        x="Hours Ahead", 
                        y=["AQI", "PM2.5", "O3"],
                        title="Air Quality Forecast",
                        labels={"value": "Measurement", "variable": "Metric"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

#############################################
# MODEL 2: NO2 PREDICTION
#############################################

def no2_prediction_tab():
    st.header("Model 2: NO2 Concentration Prediction")
    
    col1, col2 = st.columns(2)
    
    # Column 1: Pollutant inputs
    with col1:
        co_gt = st.number_input("CO (mg/m³)", value=1.0, format="%.2f", min_value=0.0, key="no2_co")
        c6h6_gt = st.number_input("Benzene/C6H6 (μg/m³)", value=5.0, format="%.2f", min_value=0.0, key="no2_c6h6")
        pt08_s2_nmhc = st.number_input("PT08.S2(NMHC) Sensor", value=1000, min_value=0, key="no2_nmhc")
        nox_gt = st.number_input("NOx (ppb)", value=60.0, format="%.1f", min_value=0.0)
        pt08_s3_nox = st.number_input("PT08.S3(NOx) Sensor", value=1000, min_value=0)
    
    # Column 2: More sensor and environmental inputs
    with col2:
        pt08_s4_no2 = st.number_input("PT08.S4(NO2) Sensor", value=1000, min_value=0)
        pt08_s5_o3 = st.number_input("PT08.S5(O3) Sensor", value=1000, min_value=0)
        temperature = st.number_input("Temperature (°C)", value=25.0, format="%.1f", key="no2_temp")
        rh = st.number_input("Relative Humidity (%)", value=50.0, format="%.1f", min_value=0.0, max_value=100.0, key="no2_rh")
        ah = st.number_input("Absolute Humidity (g/m³)", value=1.2000, format="%.4f", min_value=0.0, max_value=30.0, key="no2_ah")
    
    # Submit button
    if st.button("Predict NO2", type="primary"):
        # Format the request data
        data = {
            "CO_GT": float(co_gt),
            "C6H6_GT": float(c6h6_gt),
            "PT08_S2_NMHC": float(pt08_s2_nmhc),
            "NOx_GT": float(nox_gt),
            "PT08_S3_NOx": float(pt08_s3_nox),
            "PT08_S4_NO2": float(pt08_s4_no2),
            "PT08_S5_O3": float(pt08_s5_o3),
            "T": float(temperature),
            "RH": float(rh),
            "AH": float(ah)
        }
        
        # Make API request
        with st.spinner("Predicting NO2 concentration..."):
            response = make_api_request("/predict/no2", data, method="POST")
        
        if "error" in response:
            st.error(response["error"])
        else:
            # Display results
            col1, col2 = st.columns([1, 1])
            
            # Column 1: Results as text
            with col1:
                st.subheader("NO2 Prediction Results")
                no2_value = response.get('no2_prediction', 0)
                confidence = response.get('confidence', 0.0) or 0.85  # Default confidence if not provided
                
                # NO2 value with color
                no2_color = "#4CAF50"  # green
                if no2_value > 40:
                    no2_color = "#FFEB3B"  # yellow
                if no2_value > 90:
                    no2_color = "#FF9800"  # orange
                if no2_value > 120:
                    no2_color = "#FF5722"  # red
                
                st.markdown(f"<h3 style='color:{no2_color}'>Predicted NO2 Concentration: {no2_value:.2f} μg/m³</h3>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {confidence:.2%}")
                
                # Model information
                st.subheader("Model Information")
                model_info = response.get('model_info', {})
                for key, value in model_info.items():
                    st.markdown(f"**{key}:** {value}")
            
            # Column 2: Gauge visualization
            with col2:
                st.subheader("NO2 Gauge")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = no2_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "NO2 Prediction (μg/m³)"},
                    delta = {'reference': 40, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': 'green'},
                            {'range': [40, 90], 'color': 'yellow'},
                            {'range': [90, 120], 'color': 'orange'},
                            {'range': [120, 200], 'color': 'red'},
                        ],
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

#############################################
# MODEL 3: ANOMALY DETECTION
#############################################

def anomaly_detection_tab():
    st.header("Model 3: Air Quality Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    
    # Column 1: Data input
    with col1:
        data_input = st.text_area(
            "Input Data (JSON or CSV format)", 
            height=200,
            placeholder='[{"datetime": "2023-07-01 10:00", "NO2(GT)": 45.2}, {"datetime": "2023-07-01 11:00", "NO2(GT)": 48.7}]',
            help="Enter data as JSON array or CSV format. Each row should contain a datetime and pollutant values."
        )
    
    # Column 2: Configuration options
    with col2:
        data_type = st.selectbox("Data Type", options=["india", "italy"], index=0)
        pollutant = st.text_input("Pollutant Name", value="PM2.5", placeholder="e.g. PM2.5, NO2(GT), CO(GT)")
        detection_method = st.selectbox(
            "Detection Method", 
            options=["isolation_forest", "dbscan"], 
            index=0
        )
    
    # Submit button
    if st.button("Detect Anomalies", type="primary"):
        # Parse input data
        try:
            # Try to parse as JSON
            input_data = json.loads(data_input)
        except:
            try:
                # Try to parse as CSV
                # Convert CSV-like string to list of dictionaries
                lines = data_input.strip().split('\n')
                headers = lines[0].split(',')
                
                input_data = []
                for line in lines[1:]:
                    values = line.split(',')
                    if len(values) == len(headers):
                        entry = {}
                        for i, header in enumerate(headers):
                            # Try to convert to float if possible
                            try:
                                entry[header.strip()] = float(values[i].strip())
                            except:
                                entry[header.strip()] = values[i].strip()
                        input_data.append(entry)
            except:
                st.error("Error: Could not parse input data. Please provide valid JSON or CSV format.")
                return
        
        # Format request data
        request_data = {
            "data_type": data_type,
            "pollutant": pollutant,
            "detection_method": detection_method,
            "air_quality_data": input_data
        }
        
        # Make API request
        with st.spinner("Detecting anomalies..."):
            response = make_api_request("/detect/anomalies", request_data, method="POST")
        
        if "error" in response:
            st.error(response["error"])
        else:
            # Display results
            col1, col2 = st.columns([1, 1])
            
            # Column 1: Results as text
            with col1:
                st.subheader("Anomaly Detection Results")
                
                # Summary metrics
                st.markdown(f"**Analysis Summary:** {response.get('analysis_summary', 'Analysis completed')}")
                st.markdown(f"**Pollutant:** {response.get('pollutant', pollutant)}")
                st.markdown(f"**Anomalies Found:** {sum(response.get('anomalies', []))}")
                st.markdown(f"**Anomaly Percentage:** {response.get('anomaly_percentage', 0):.2f}%")
                
                # Anomaly points
                st.subheader("Anomaly Points")
                
                # Create a DataFrame with all data points
                anomaly_data = []
                for i, is_anomaly in enumerate(response.get('anomalies', [])):
                    if i < len(input_data):
                        point = input_data[i].copy()
                        point["is_anomaly"] = is_anomaly
                        anomaly_data.append(point)
                
                anomaly_df = pd.DataFrame(anomaly_data)
                if not anomaly_df.empty:
                    # Filter to show only anomalies
                    anomaly_only_df = anomaly_df[anomaly_df["is_anomaly"] == True]
                    if not anomaly_only_df.empty:
                        st.dataframe(anomaly_only_df, use_container_width=True)
                    else:
                        st.info("No anomalies detected in the data.")
                else:
                    st.info("No data points to display.")
            
            # Column 2: PCA Visualization
            with col2:
                if response.get('pca_visualization'):
                    st.subheader("PCA Visualization")
                    
                    # Create PCA scatter plot
                    pca_data = pd.DataFrame(response['pca_visualization'])
                    
                    fig = px.scatter(
                        pca_data, 
                        x="PC1", 
                        y="PC2", 
                        color="is_anomaly",
                        color_discrete_map={True: "red", False: "blue"},
                        title="PCA Visualization of Anomaly Detection",
                        labels={"is_anomaly": "Is Anomaly"},
                        hover_data=["PC1", "PC2", "is_anomaly"]
                    )
                    
                    fig.update_layout(legend_title="Anomaly Status")
                    st.plotly_chart(fig, use_container_width=True)



#############################################
# MAIN STREAMLIT APP
#############################################

def main():
    st.title("Air Quality Prediction Dashboard")
    
    # Check API status
    if not check_api_status():
        st.error("""
        ## ⚠️ API Server Not Running
        
        The FastAPI server doesn't appear to be running at http://127.0.0.1:8000.
        
        Please start the server first with:
        ```
        python main.py
        ```
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "AQI Prediction", 
        "NO2 Prediction", 
        "Anomaly Detection"
    ])
    
    # Set content for each tab
    with tab1:
        aqi_prediction_tab()
    
    with tab2:
        no2_prediction_tab()
    
    with tab3:
        anomaly_detection_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>Air Quality Prediction Dashboard | Powered by FastAPI & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
