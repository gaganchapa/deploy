import os
import io
import json
import csv
from gradio_client import Client, handle_file
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from geopy.geocoders import Nominatim
import folium
from folium import plugins

# Hugging Face Space ID or URL
HF_SPACE_ID = os.environ.get(
    "HF_SPACE_ID",
    "gagandwaz/cyclonet-api"
)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Initialize Gradio Client (caches connection)
_client = None

def get_hf_client():
    global _client
    if _client is None:
        _client = Client(HF_SPACE_ID, token=HF_TOKEN)
    return _client

def call_hf_inference(image_path: str) -> float:
    """Send image to Hugging Face ZeroGPU Gradio Space via official gradio_client."""
    client = get_hf_client()
    result = client.predict(
        image=handle_file(image_path),
        api_name="/predict"
    )
    return float(result)

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

@app.route("/", methods=['POST'])
def predict():
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

    if allowed_img(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)

        try:
            predicted_intensity = call_hf_inference(image_path)
        except Exception as e:
            print(f"HF inference error: {e}")
            flash('Inference service is warming up or unavailable. Please try again in a few seconds.')
            return redirect(request.url)

        r = 100
        if predicted_intensity >= 0 and predicted_intensity <= 30:
            r = 50
        elif predicted_intensity > 30 and predicted_intensity <= 50:
            r = 70
        elif predicted_intensity > 50 and predicted_intensity <= 70:
            r = 80
        elif predicted_intensity > 70 and predicted_intensity <= 100:
            r = 90

        archive_data = load_archive_data()
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

@app.route('/<filename>')
def display_image(filename):
    return render_template('image.html', filename=filename)

@app.route('/archive')
def archive():
    archive_data = load_archive_data()
    for record in archive_data:
        if 'location' not in record or not record['location']:
            record['location'] = f"Lat: {record.get('latitude')}, Lon: {record.get('longitude')}"
    return render_template('archive.html', items=archive_data)

@app.route('/api/archive_locations', methods=['GET'])
def get_archive_locations():
    archive_data = load_archive_data()
    results = []
    for record in archive_data:
        loc = get_location_info(record['latitude'], record['longitude'])
        results.append({
            'id': record['id'],
            'latitude': record['latitude'],
            'longitude': record['longitude'],
            'location': loc,
            'intensity': record['predicted_intensity'],
            'date': record.get('capture_date', ''),
            'time': record.get('capture_time', '')
        })
    return jsonify(results)

@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    archive_data = load_archive_data()
    record_to_delete = next((r for r in archive_data if r['id'] == record_id), None)
    if record_to_delete:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], record_to_delete['image_filename'])
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        archive_data = [r for r in archive_data if r['id'] != record_id]
        save_archive_data(archive_data)
        flash('Record deleted successfully')
    else:
        flash('Record not found')
    return redirect(url_for('archive'))

@app.route('/api/plot_map', methods=['POST'])
@app.route('/api/plot_cyclone_map', methods=['POST'])
def plot_cyclone_map():
    data = request.json
    selected_ids = data.get('selected_ids', [])
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400

    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    if not selected_records:
        return jsonify({'error': 'Invalid records'}), 400

    selected_records.sort(key=lambda x: x.get('timestamp') or '')
    avg_lat = sum(r['latitude'] for r in selected_records) / len(selected_records)
    avg_lon = sum(r['longitude'] for r in selected_records) / len(selected_records)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, tiles='OpenStreetMap')
    coordinates = [[r['latitude'], r['longitude']] for r in selected_records]
    if len(coordinates) > 1:
        folium.PolyLine(coordinates, color="red", weight=2.5, opacity=0.8).add_to(m)

    for i, r in enumerate(selected_records):
        folium.CircleMarker(
            location=[r['latitude'], r['longitude']],
            radius=6,
            popup=f"Date: {r.get('capture_date')} {r.get('capture_time')}<br>Intensity: {r.get('predicted_intensity')}",
            color="crimson",
            fill=True,
            fill_color="red"
        ).add_to(m)

    map_path = 'static/map_plot.html'
    m.save(map_path)
    return jsonify({'success': True, 'map_url': '/static/map_plot.html'})

@app.route('/api/plot_intensity', methods=['POST'])
def plot_intensity():
    data = request.json
    selected_ids = data.get('selected_ids', [])
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400

    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    if not selected_records:
        return jsonify({'error': 'Invalid records'}), 400

    selected_records.sort(key=lambda x: x.get('timestamp') or '')
    times = [(str(r.get('capture_date', '')) + ' ' + str(r.get('capture_time', ''))).strip() or f"Rec {r.get('id')}" for r in selected_records]
    intensities = [float(r.get('predicted_intensity', 0)) for r in selected_records]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(intensities)), intensities, marker='o', linestyle='-', color='red', linewidth=2)
    ax.set_xlabel('Record')
    ax.set_ylabel('Predicted Intensity')
    ax.set_title('Cyclone Intensity Over Time')
    ax.grid(True)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, ha='right')
    plt.tight_layout()

    plot_path = 'static/intensity_plot.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return jsonify({'success': True, 'plot_url': '/static/intensity_plot.png'})

@app.route('/api/plot_location', methods=['POST'])
def plot_location():
    data = request.json
    selected_ids = data.get('selected_ids', [])
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400

    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
    if not selected_records:
        return jsonify({'error': 'Invalid records'}), 400

    selected_records.sort(key=lambda x: x['timestamp'])
    locations = [get_location_info(r['latitude'], r['longitude']) for r in selected_records]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(locations)), [1]*len(locations), color='steelblue')
    ax.set_yticks(range(len(locations)))
    ax.set_yticklabels(locations)
    ax.set_xlabel('Records')
    ax.set_title('Cyclone Path - Location Information Over Time')
    ax.grid(True, axis='x')
    plt.tight_layout()

    plot_path = 'static/location_plot.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return jsonify({'success': True, 'plot_url': '/static/location_plot.png'})

@app.route('/api/save_selected', methods=['POST'])
def save_selected():
    data = request.json
    selected_ids = data.get('selected_ids', [])
    if not selected_ids:
        return jsonify({'error': 'No records selected'}), 400

    archive_data = load_archive_data()
    selected_records = [r for r in archive_data if r['id'] in selected_ids]
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
