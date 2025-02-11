from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)

# Load the CNN model
model = load_model("model.h5")  # Replace "model.h5" with your actual model file

# Load Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def get_prediction(face):
    face = cv2.resize(face, (48, 48))
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = np.expand_dims(face_gray, axis=-1)
    prediction = model.predict(np.array([face_gray]))
    label = np.argmax(prediction)
    return emotions[label]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        frame = cv2.imread(file_path)
        faces = faceCascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        predictions = []
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
            pred = get_prediction(cropped_face)
            predictions.append(pred)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
        cv2.imwrite(result_path, frame)
        
        # Return a clean and user-friendly response
        response = {
            "status": "success",
            "message": "Emotion detection completed successfully!",
            "predictions": predictions,
            "image_url": result_path.replace("\\", "/")  # Fix path format for URL
        }
        
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
