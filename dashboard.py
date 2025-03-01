import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import requests
import serial
import time
import threading
import base64
import os
import json

# Arduino Serial Port (Update as needed)
arduino_port = "/dev/ttyACM0"
baud_rate = 9600

# Weather API Configuration
weather_api_key = "06df7bee70374e5f9a8183543252702"
city = "Chennai"  # Change city as needed

# FastAPI Server (YOLO Model Inference)
yolo_api_url = "http://127.0.0.1:8000"

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Global variables to store data
sensor_data = {"moisture": 0, "water_level": 0}
weather_data = {}

# Function to read Arduino sensor data
def read_arduino_data():
    global sensor_data
    try:
        ser = serial.Serial(arduino_port, baud_rate)
        while True:
            line = ser.readline().decode().strip()
            if line:
                try:
                    # Split the values correctly
                    parts = line.split(", ")  # Split at ', ' (comma and space)
                    
                    # Extract numbers from text
                    moisture_value = int(parts[0].split(": ")[1])  # Extracts '568' from 'Moisture: 568'
                    water_level_value = int(parts[1].split(": ")[1])  # Extracts '45' from 'Water Level: 45'
                    
                    sensor_data["moisture"] = moisture_value
                    sensor_data["water_level"] = water_level_value

                except Exception as e:
                    print(f"Error parsing sensor data: {e} | Raw Data: {line}")

            time.sleep(1)

    except serial.SerialException as e:
        print("Serial error:", e)

# Function to fetch weather data
def fetch_weather():
    global weather_data
    url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={city}&days=10&aqi=no"
    try:
        response = requests.get(url)
        weather_data = response.json()
    except Exception as e:
        print("Weather API error:", e)

# Start background threads for fetching data
threading.Thread(target=read_arduino_data, daemon=True).start()
threading.Thread(target=fetch_weather, daemon=True).start()

# Dash Layout with Styling
app.layout = html.Div([
    html.H1("AgriHUB", style={'textAlign': 'center', 'color': '#2C3E50'}),
    
    html.Div([
        # Processed Image (Main Tile)
        html.Div([
            html.H3("Live Feed", style={'textAlign': 'center', 'color': '#34495E'}),
            html.Img(id='processed-image', style={'width': '100%', 'max-width': '600px', 'border': '2px solid #2980B9', 'borderRadius': '10px'})
        ], style={'flex': '2', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ECF0F1', 'borderRadius': '10px'}),
        
        # Sensor Data & Weather (Right Side Panel)
        html.Div([
            html.H3("Sensor Readings", style={'color': '#34495E'}),
            html.Div(id='sensor-info', style={'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#D5DBDB', 'borderRadius': '10px'}),
            html.H3("Weather Information", style={'marginTop': '20px', 'color': '#34495E'}),
            html.Div(id='weather-info', style={'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#D5DBDB', 'borderRadius': '10px'})
        ], style={'flex': '1', 'padding': '20px'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px'}),
    
    # Model Inference Results (Below Image)
    html.Div([
        html.H3("Model Inference Results", style={'textAlign': 'center', 'color': '#34495E'}),
        html.Div(id='yolo-results', style={'fontSize': '18px', 'padding': '10px', 'backgroundColor': '#ECF0F1', 'borderRadius': '10px'})
    ], style={'marginTop': '20px', 'padding': '20px'}),
    
    # Auto-update Interval
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F8F9F9', 'padding': '20px'})

# Callback to update sensor data
@app.callback(
    Output('sensor-info', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_sensor_info(n):
    moisture = sensor_data['moisture']
    water_level = sensor_data['water_level']
    return html.P(f"Moisture: {moisture}, Water Level: {water_level}")

# Callback to update weather data
@app.callback(
    Output('weather-info', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_weather(n):
    fetch_weather()
    if 'forecast' in weather_data:
        forecast_html = []
        for day in weather_data['forecast']['forecastday']:
            date = day['date']
            temp = day['day']['avgtemp_c']
            condition = day['day']['condition']['text']
            forecast_html.append(html.P(f"{date}: {temp}°C, {condition}"))
        return forecast_html
    return "Loading weather data..."

# Callback to update YOLO model results
@app.callback(
    Output('yolo-results', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_yolo_results(n):
    try:
        response = requests.get(f"{yolo_api_url}/get_results")
        if response.status_code == 200:
            data = response.json()
            detections = data.get("detections", [])
            model_name = os.path.basename(data.get("model", "Unknown"))
            result_text = f"Model: {model_name}\n"
            for det in detections:
                result_text += f"Class {det['class']} (Confidence: {det['confidence']:.2f})\n"
            return html.Pre(result_text)
    except Exception as e:
        return "Error fetching model results."
    return "Loading..."

# Callback to update processed image
@app.callback(
    Output('processed-image', 'src'),
    Input('interval-component', 'n_intervals')
)
def update_image(n):
    try:
        image_path = "/home/skkr/Documents/AgriAI/processed.jpg"
        if os.path.exists(image_path):
            encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
            return f"data:image/jpeg;base64,{encoded_image}"
    except Exception as e:
        print("Error loading image:", e)
    return ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)