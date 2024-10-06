from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json

# Import your prediction module
from model import load_and_train_model, predict_bird_species

app = Flask(__name__)
CORS(app)

# Load and train the model when the server starts
classifier, scaler, pca = load_and_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Predict the bird species using the model
    prediction = predict_bird_species(file, classifier, scaler, pca)

    return jsonify({'prediction': prediction})

# Load the bird data from res.json
with open('res.json') as f:
    bird_data = json.load(f)

# Endpoint to get bird details by name
@app.route('/bird/<name>', methods=['GET'])
def get_bird(name):
    for bird in bird_data['birds']:
        if bird['name'].lower() == name.lower():
            return jsonify(bird)
    return jsonify({'error': 'Bird not found'}), 404

# Serve images from the "res" directory
@app.route('/res/<filename>')
def serve_image(filename):
    file_path = os.path.join('res', filename)
    if os.path.exists(file_path):
        return send_from_directory('res', filename)
    else:
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    # Start the Flask server
    app.run(debug=True)

