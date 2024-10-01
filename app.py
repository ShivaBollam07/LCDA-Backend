import io
import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv

from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

load_dotenv(override=True)
modelName = os.getenv("MODEL")
req_features = os.getenv("FEATURES")


app = Flask(__name__)
modelPath = "./" + "models/" + modelName

print("Loading the model---------")
model = ViTForImageClassification.from_pretrained(modelPath)
feature_extractor = ViTImageProcessor.from_pretrained(modelPath)
model.eval()

print("Model is Ready")

id2label = {
    "0": "Pepper bell Bacterial spot",
    "1": "Pepper bell healthy",
    "10": "Tomato Two spotted spider mite",
    "11": "Tomato Target Spot",
    "12": "Tomato Yellow Leaf Curl Virus",
    "13": "Tomato mosaic virus",
    "14": "Tomato healthy",
    "2": "Potato Early blight",
    "3": "Potato Late blight",
    "4": "Potato healthy",
    "5": "Tomato Bacterial spot",
    "6": "Tomato Early blight",
    "7": "Tomato Late blight",
    "8": "Tomato Leaf Mold",
    "9": "Tomato Septoria leaf spot",
}



def predict(image):
    inputs = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return id2label[str(predicted_label)]


@app.route("/predict", methods=["POST"])
def handle_predict():
    print(request.files)
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if file:
        try:
            img = Image.open(io.BytesIO(file.read()))
            prediction = predict(img)
            return jsonify({"success": True, "prediction": prediction}), 200
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

        



if __name__ == "__main__":
    app.run(debug=False, port=os.getenv("PORT"))
