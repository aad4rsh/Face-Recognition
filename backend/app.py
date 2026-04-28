from flask import Flask, request, jsonify
from flask_cors import CORS
from model_core import FaceRecognitionModel
from data_processor import get_lfw_data
import numpy as np

app = Flask(__name__)
CORS(app)

# Global model instance
model = FaceRecognitionModel(n_components=100)
X, y, target_names, H, W = get_lfw_data()
names = [target_names[i] for i in y]

# Train on startup (for simplicity in this demo)
model.train(X, names)

@app.route('/samples', methods=['GET'])
def get_samples():
    # Pick 10 random indices from dataset to show in UI
    indices = np.random.choice(len(X), 10, replace=False)
    samples = []
    for idx in indices:
        samples.append({
            "id": int(idx),
            "pixels": X[idx].tolist(),
            "name": names[idx]
        })
    return jsonify(samples)

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    pixels = np.array(data['pixels'])
    name, distance = model.recognize(pixels)
    return jsonify({"name": name, "distance": float(distance)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)