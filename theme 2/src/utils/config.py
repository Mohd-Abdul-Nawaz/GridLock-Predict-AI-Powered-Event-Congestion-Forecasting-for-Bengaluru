"""
Configuration module for the Smart Traffic Analytics Platform.
Centralizes all configuration parameters for easy tuning.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Data files
PARKING_DATA_FILE = os.path.join(DATA_DIR, 'jan to may police violation_anonymized791b166.csv')
EVENT_DATA_FILE = os.path.join(DATA_DIR, 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

# Theme 1: Parking Analytics
PARKING_CONFIG = {
    'dbscan_eps': 0.001,          # ~100m for hotspot clustering
    'dbscan_min_samples': 5,       # Minimum violations to form a hotspot
    'congestion_radius_m': 200,    # Radius to check congestion impact
    'high_severity_threshold': 20,  # Violations count for high severity
    'medium_severity_threshold': 10,
    'peak_hours': [7, 8, 9, 17, 18, 19],  # Peak traffic hours
    'time_window_minutes': 30,     # Time window for temporal analysis
}

# Theme 2: Event Analytics
EVENT_CONFIG = {
    'model_types': ['xgboost', 'lightgbm', 'random_forest'],
    'ensemble_method': 'weighted',  # 'average' or 'weighted'
    'test_size': 0.2,
    'random_state': 42,
    'resource_tiers': {
        'critical': {'manpower': 20, 'barricades': 15, 'diversions': 3, 'patrol_vehicles': 4},
        'high': {'manpower': 12, 'barricades': 8, 'diversions': 2, 'patrol_vehicles': 2},
        'medium': {'manpower': 6, 'barricades': 4, 'diversions': 1, 'patrol_vehicles': 1},
        'low': {'manpower': 3, 'barricades': 2, 'diversions': 0, 'patrol_vehicles': 0},
    }
}

# Theme 3: Computer Vision
CV_CONFIG = {
    'yolo_model': 'yolov8n.pt',    # YOLOv8 nano for speed
    'confidence_threshold': 0.5,    # Detection confidence threshold
    'iou_threshold': 0.45,         # NMS IoU threshold
    'ocr_engine': 'easyocr',       # 'tesseract' or 'easyocr'
    'ocr_confidence': 0.2,         # OCR confidence threshold
    'upscaler': 'fsrcnn',         # 'fsrcnn' (offline) or 'letsenhance' (online API)
    'letsenhance_api_key': '',     # set via env LETSENHANCE_API_KEY if using online upscaler
    'letsenhance_endpoint': 'https://api.letsenhance.io/v1/photo/enhance',
    'violation_classes': {
        'no_helmet': {'description': 'Rider without helmet', 'fine': 500},
        'no_seatbelt': {'description': 'Driver without seatbelt', 'fine': 1000},
        'triple_riding': {'description': 'More than 2 riders on 2-wheeler', 'fine': 500},
        'wrong_side': {'description': 'Driving on wrong side', 'fine': 2000},
        'red_light': {'description': 'Red light violation', 'fine': 1000},
        'illegal_parking': {'description': 'Illegal parking', 'fine': 500},
        'stop_line': {'description': 'Stop line violation', 'fine': 500},
    },
    'max_image_size': (1280, 1280),  # Max image dimension for processing
    'batch_size': 8,                # Batch size for YOLO inference
}

# Application-wide settings
APP_CONFIG = {
    'app_name': 'Smart Traffic Analytics Platform',
    'version': '2.0.0',
    'team_name': 'Grid Lock Hackers',
    'hackathon': 'Flipkart Grid Lock 2.0 - Round 2',
    'debug': True,
    'port': 5000,
    'host': '0.0.0.0',
}

# Color schemes for visualizations
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'heatmap_colorscale': [
        [0, 'rgba(102, 126, 234, 0)'],
        [0.25, 'rgba(102, 126, 234, 0.3)'],
        [0.5, 'rgba(245, 158, 11, 0.5)'],
        [0.75, 'rgba(239, 68, 68, 0.7)'],
        [1, 'rgba(220, 38, 38, 0.9)'],
    ]
}