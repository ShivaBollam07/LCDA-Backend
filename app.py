import io
import os
from flask import Flask, jsonify,request
from dotenv import load_dotenv
from skimage.feature import graycoprops, graycomatrix
from PIL import Image

import numpy as np
import os
import pickle

load_dotenv(override=True)
app = Flask(__name__)
modelname = os.getenv("MODEL")

featuresmap = {
    "c": "contrast",
    "d": "dissimilarity",
    "h": "homogeneity",
    "e": "energy",
    "C": "correlation",
    "a": "ASM",
}

req_features = os.getenv("FEATURES")

properties = [featuresmap[i] for i in req_features]
model = pickle.load(open("./models/" + modelname, "rb"))


def get_image_feature_vec(image):
    g = graycomatrix(np.array(image), [1, 2], [0, np.pi / 2], normed=True)
    featurevec = []
    for property in properties:
        featurevec.append(graycoprops(g, property).flatten())

    return np.concatenate(featurevec, axis=0, dtype=np.float32)

def predict(image: np.ndarray):
    features = get_image_feature_vec(image)
    pred  =  model.predict([features])
    if pred[0] == 1:
        return "Infected"
    else:
        return "Not Infected"




@app.route("/predict", methods=["POST"])
def handle_predict():
    if "image" not in request.files:
        return jsonify({"error": "1No image part in the request"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "2No image selected"}), 400

    if file:
        try:
            img = Image.open(io.BytesIO(file.read())).convert("L")
            img_array = np.array(img)

            return jsonify({"success": True, "prediction": predict(img_array)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)


