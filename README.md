# AgriHUB
A Smart Pest &amp; Crop Disease Detection and Weather Forecast Monitoring 
Overview

AgriHUB is an AI-powered agricultural monitoring system that leverages deep learning and IoT technologies to help farmers detect crop diseases, monitor soil conditions, and optimize resource usage. The system is designed to run on a Raspberry Pi 5, enabling real-time data processing with low power consumption.

#Features

AI-Based Pest & Disease Detection

Uses a YOLOv8 model trained on Rice, Wheat, and Sugarcane datasets.

Provides accurate real-time identification of pests and plant diseases.

IoT Sensor Integration

Water Level Sensor: Monitors water availability for irrigation.

Water Flow Rate Sensor: Ensures efficient water distribution.

Soil Moisture Sensor: Prevents over/under-watering.

Sensors are managed via Arduino Uno, feeding data to the Raspberry Pi.

Weather Forecasting

Integrates with WeatherAPI to provide local weather predictions.

Helps in planning irrigation and disease prevention strategies.

Optimized for Raspberry Pi

Runs efficiently without requiring TPU/NPU accelerators.

Model optimized via ONNX and TFLite for lightweight execution.

Supports offline usage, ensuring uninterrupted farm monitoring.

#Technical Stack

AI Model: YOLOv8 (trained on agricultural datasets)

Hardware: Raspberry Pi 5, Arduino Uno, IoT Sensors

Software: Python, OpenCV, Ultralytics YOLO, OpenWeatherMap API

#Benefits

Early Pest & Disease Detection: Reduces crop losses through timely intervention.

Smart Irrigation Management: Prevents resource wastage by optimizing water use.

Increased Crop Yield & Profitability: Helps farmers make data-driven decisions.

User-Friendly Interface: Provides actionable insights in a simple, intuitive format.

#Dependencies

Install the following dependencies -> FastAPI, dash, Ultralytics, PyTorch, OpenCV, Uvicorn

#Future Enhancements

Expansion to additional crops and pest species.

Edge TPU support for faster inference on low-power devices.

Integration with mobile apps for remote monitoring.

