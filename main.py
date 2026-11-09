import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import jinja2
import io
import json
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from geopy.geocoders import Nominatim
import folium
from folium import plugins


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


model = Model()
model.load_state_dict(torch.load('saved_modelcyclone', map_location=torch.device('cpu')))
model.eval()
# model ends here


app = Flask(__name__)

'''Data Persistence Setup'''
DATA_FILE = 'archive_data.json'

def load_archive_data():
    """Load archive data from JSON file"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_archive_data(data):
    """Save archive data to JSON file"""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

_LOCATION_CACHE = {}

def get_location_info(lat, lon):
    """Get country/state/sea name from coordinates with caching"""
    cache_key = (round(float(lat), 3), round(float(lon), 3))
    if cache_key in _LOCATION_CACHE:
        return _LOCATION_CACHE[cache_key]

    try:
        geolocator = Nominatim(user_agent="cyclonet_app_v2")
        location = geolocator.reverse(f"{lat}, {lon}", language='en', timeout=2)
        if not location:
            _LOCATION_CACHE[cache_key] = f"Lat: {lat}, Lon: {lon}"
            return _LOCATION_CACHE[cache_key]

        address = location.raw.get('address', {})
        country = address.get('country', 'Unknown')
        state = address.get('state', '')
        
        if 'sea' in location.address.lower() or 'ocean' in location.address.lower():
            res = f"{location.address}"
        elif state:
            res = f"{state}, {country}"
        else:
            res = country
        _LOCATION_CACHE[cache_key] = res
        return res
    except:
        return f"Lat: {lat}, Lon: {lon}"

@app.route("/")
def index():
    return render_template("index.html", rain=1, x=0)


'''Image upload and display'''

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "trailblazers"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# upload form data to database
@app.route("/", methods=['POST'])
def predict():
    # get values from html form
    img_date = request.form.get("image_date")
    img_time = request.form.get("image_time")
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    imagefile = request.files.get('imagefile')

    if not all([img_date, img_time, latitude, longitude]) or not imagefile or not imagefile.filename:
        flash('Please complete the date, time, latitude, longitude, and image fields.')
        return redirect(request.url)

    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        flash('Latitude and longitude must be valid numbers.')
        return redirect(request.url)

    if not -90 <= latitude_value <= 90 or not -180 <= longitude_value <= 180:
        flash('Latitude must be between -90 and 90, and longitude between -180 and 180.')
        return redirect(request.url)

    print(img_date)
    print(img_time)
    print(latitude)
    print(longitude)

    if allowed_img(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)

        image = cv2.imread(image_path)
        image = np.array(image)
        totensor = transforms.ToTensor()
        image = totensor(image)
        resize = transforms.Resize(size=(250, 250))
        image = resize(image)
        image = image.unsqueeze(0)
        predicted_intensity = torch.round(model(image)).item()
        print(predicted_intensity)
        r=100
        if predicted_intensity >= 0 and predicted_intensity <= 30:
            r = 50
        if predicted_intensity >= 30 and predicted_intensity <= 50:
            r = 70
        if predicted_intensity >= 50 and predicted_intensity <= 70:
            r = 80
        if predicted_intensity >= 70 and predicted_intensity <= 100:
            r = 90

        print(r)

        print('upload_image filename: ', filename)
        print('image date: ', img_date)
        print('image time: ', img_time)
        print('cyclone latitude: ', latitude)
        print('cyclone longitude: ', longitude)

        # Save to archive data
        archive_data = load_archive_data()
        # Use max existing id + 1 to avoid duplicate IDs after deletions
        next_id = max((r['id'] for r in archive_data), default=-1) + 1
        record = {
            'id': next_id,
            'capture_date': img_date,
            'capture_time': img_time,
            'latitude': latitude_value,
            'longitude': longitude_value,
            'predicted_intensity': float(predicted_intensity),
            'image_filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        archive_data.append(record)
        save_archive_data(archive_data)

        return render_template('index.html', prediction=predicted_intensity, filename=filename, rain=r, x=1)
    else:
        flash('Allowed image types are jpg, jpeg, png')
        return redirect(request.url)


# display image after upload
@app.route('/<filename>')
def display_image(filename):
    # print('display_image flename: ' + filename)
    # return redirect(url_for('static', filename='uploads/' + filename), code=301)
    return render_template('image.html', filename=filename)


# archive page
@app.route('/archive')
def archive():
    archive_data = load_archive_data()
    # Add default/cached location info to each record without blocking on heavy network lookups
    for record in archive_data:
        if 'location' not in record or not record['location']:
            record['location'] = f"Lat: {record.get('latitude')}, Lon: {record.get('longitude')}"
    return render_template('archive.html', items=archive_data)

@app.route('/api/delete_selected', methods=['POST'])
def delete_selected():
    """Delete selected archive records and uploaded images no longer in use."""
    data = request.get_json(silent=True) or {}
    selected_ids = set(data.get('selected_ids', []))

    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400

    archive_data = load_archive_data()
    deleted_records = [record for record in archive_data if record.get('id') in selected_ids]
    if not deleted_records:
        return jsonify({'error': 'Invalid records'}), 404

    remaining_records = [record for record in archive_data if record.get('id') not in selected_ids]
    remaining_filenames = {record.get('image_filename') for record in remaining_records}
    for record in deleted_records:
        filename = record.get('image_filename')
        if filename and filename not in remaining_filenames:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(image_path):
                os.remove(image_path)

    save_archive_data(remaining_records)
    return jsonify({'success': True, 'message': f'Deleted {len(deleted_records)} record(s).'})

@app.route('/api/plot_map', methods=['POST'])
def plot_map():
    """Generate interactive map showing cyclone path"""
    data = request.json
    selected_ids = data.get('selected_ids', [])
    
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400
    
    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    
    if not selected_records:
        return jsonify({'error': 'Invalid records'}), 400
    
    # Sort by timestamp
    selected_records.sort(key=lambda x: x['timestamp'])
    
    # Create map centered on average coordinates
    lats = [r['latitude'] for r in selected_records]
    lons = [r['longitude'] for r in selected_records]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add path line
    path = [[r['latitude'], r['longitude']] for r in selected_records]
    folium.PolyLine(
        path,
        color='red',
        weight=2,
        opacity=0.7,
        popup='Cyclone Path'
    ).add_to(m)
    
    # Add markers for each record
    for i, record in enumerate(selected_records):
        folium.CircleMarker(
            location=[record['latitude'], record['longitude']],
            radius=5,
            popup=f"Date: {record['capture_date']}<br>Time: {record['capture_time']}<br>Intensity: {record['predicted_intensity']}",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.7
        ).add_to(m)
    
    # Save map
    map_path = 'static/map_plot.html'
    m.save(map_path)
    
    return jsonify({'success': True, 'map_url': '/static/map_plot.html'})

@app.route('/api/plot_intensity', methods=['POST'])
def plot_intensity():
    """Generate intensity vs time plot"""
    data = request.json or {}
    selected_ids = data.get('selected_ids', [])
    
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400
    
    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    
    if not selected_records:
        return jsonify({'error': 'Invalid records'}), 400
    
    # Sort by timestamp
    selected_records.sort(key=lambda x: x.get('timestamp') or '')
    
    # Extract data
    times = [(str(r.get('capture_date', '')) + ' ' + str(r.get('capture_time', ''))).strip() or f"Rec {r.get('id')}" for r in selected_records]
    intensities = [float(r.get('predicted_intensity', 0)) for r in selected_records]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(intensities)), intensities, marker='o', linestyle='-', color='red', linewidth=2)
    ax.set_xlabel('Record')
    ax.set_ylabel('Predicted Intensity')
    ax.set_title('Cyclone Intensity Over Time')
    ax.grid(True)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'static/intensity_plot.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return jsonify({'success': True, 'plot_url': '/static/intensity_plot.png'})

@app.route('/api/plot_location', methods=['POST'])
def plot_location():
    """Generate location plot showing which countries/states the cyclone is moving towards"""
    data = request.json
    selected_ids = data.get('selected_ids', [])
    
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400
    
    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    
    if not selected_records:
        return jsonify({'error': 'Invalid records'}), 400
    
    # Sort by timestamp
    selected_records.sort(key=lambda x: x['timestamp'])
    
    # Get location info for each record
    locations = []
    for record in selected_records:
        location = get_location_info(record['latitude'], record['longitude'])
        locations.append(location)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(locations)), [1]*len(locations), color='steelblue')
    ax.set_yticks(range(len(locations)))
    ax.set_yticklabels(locations)
    ax.set_xlabel('Records')
    ax.set_title('Cyclone Path - Location Information Over Time')
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'static/location_plot.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return jsonify({'success': True, 'plot_url': '/static/location_plot.png'})

@app.route('/api/save_selected', methods=['POST'])
def save_selected():
    """Save selected records to a CSV file"""
    data = request.json
    selected_ids = data.get('selected_ids', [])
    
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400
    
    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    
    # Create CSV content
    csv_path = 'static/selected_archive.csv'
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'capture_date', 'capture_time', 'latitude', 'longitude', 'predicted_intensity', 'image_filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected_records:
            writer.writerow({k: record[k] for k in fieldnames})
    
    return jsonify({'success': True, 'message': 'Data saved to CSV', 'file_url': '/static/selected_archive.csv'})


if __name__ == "__main__":
    app.run(debug=True)