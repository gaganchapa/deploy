import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import gradio as gr
import folium

# --- Core Model ---
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(784, 1),
        )

    def forward(self, x):
        return self.model(x)

device = torch.device('cpu')
model = Model()
model_path = os.path.join(os.path.dirname(__file__), 'saved_modelcyclone')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

DATA_FILE = 'archive_data.json'

def load_archive_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_archive_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def predict_cyclone(image, latitude, longitude, capture_date, capture_time):
    if image is None:
        return "⚠️ Please upload a satellite image.", None, "No Data"
    
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (ValueError, TypeError):
        return "⚠️ Please provide valid numerical latitude (-90 to 90) and longitude (-180 to 180).", None, "No Data"

    # Preprocess image
    # Gradio provides RGB numpy array
    img_rgb = np.array(image)
    # Convert RGB to BGR for compatibility with original pipeline or convert directly
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    totensor = transforms.ToTensor()
    tensor_img = totensor(img_bgr)
    resize = transforms.Resize(size=(250, 250))
    tensor_img = resize(tensor_img).unsqueeze(0)

    with torch.no_grad():
        predicted_intensity = torch.round(model(tensor_img)).item()

    # Determine cyclone severity categorization
    if predicted_intensity < 34:
        category = "Depression / Deep Depression (< 34 knots)"
    elif predicted_intensity <= 47:
        category = "Cyclonic Storm (34 - 47 knots)"
    elif predicted_intensity <= 63:
        category = "Severe Cyclonic Storm (48 - 63 knots)"
    elif predicted_intensity <= 89:
        category = "Very Severe Cyclonic Storm (64 - 89 knots)"
    elif predicted_intensity <= 119:
        category = "Extremely Severe Cyclonic Storm (90 - 119 knots)"
    else:
        category = "Super Cyclonic Storm (>= 120 knots)"

    # Create Folium Map centered on coordinates
    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="OpenStreetMap")
    folium.Marker(
        [lat, lon],
        popup=f"Cyclone Center<br>Intensity: {predicted_intensity} knots<br>Date: {capture_date} {capture_time}",
        tooltip="Cyclone Eye",
        icon=folium.Icon(color="red", icon="cloud")
    ).add_to(m)
    folium.Circle(
        radius=50000,
        location=[lat, lon],
        color="crimson",
        fill=True,
        fill_opacity=0.3
    ).add_to(m)

    map_html = m.get_root().render()
    iframe_html = f'<iframe srcdoc="{map_html.replace(\'"\', \'&quot;\')}" style="width:100%; height:420px; border:none; border-radius:8px;"></iframe>'

    # Save to archive
    archive = load_archive_data()
    next_id = max((r['id'] for r in archive), default=-1) + 1
    archive.append({
        'id': next_id,
        'capture_date': str(capture_date),
        'capture_time': str(capture_time),
        'latitude': lat,
        'longitude': lon,
        'predicted_intensity': float(predicted_intensity),
        'timestamp': datetime.now().isoformat()
    })
    save_archive_data(archive)

    summary_text = f"### 🌀 Predicted Intensity: **{predicted_intensity:.1f} knots**\n**Category:** {category}\n**Coordinates:** {lat}°, {lon}°"
    return summary_text, iframe_html, f"{predicted_intensity} knots"

with gr.Blocks(title="CycloNet - Cyclone Intensity Estimation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌀 CycloNet: Tropical Cyclone Intensity Estimator
        Upload satellite infrared imagery along with coordinates to estimate cyclone intensity using the trained deep learning model.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Satellite Image (IR / RGB)")
            with gr.Row():
                input_lat = gr.Number(label="Latitude", value=14.5)
                input_lon = gr.Number(label="Longitude", value=85.0)
            with gr.Row():
                input_date = gr.Textbox(label="Capture Date (YYYY-MM-DD)", value="2026-11-09")
                input_time = gr.Textbox(label="Capture Time (HH:MM UTC)", value="12:00")
            predict_btn = gr.Button("Analyze Cyclone", variant="primary")

        with gr.Column(scale=1):
            output_summary = gr.Markdown(label="Results")
            output_label = gr.Label(label="Intensity")
            output_map = gr.HTML(label="Cyclone Location Map")

    predict_btn.click(
        fn=predict_cyclone,
        inputs=[input_image, input_lat, input_lon, input_date, input_time],
        outputs=[output_summary, output_map, output_label]
    )

if __name__ == "__main__":
    demo.launch()

