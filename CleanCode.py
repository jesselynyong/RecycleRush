from flask import Flask, render_template, request, jsonify, session, url_for
import threading
import serial
import time
import requests
import cv2
import time
import pickle
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from skimage.feature import corner_harris
from skimage import io, transform
from transformers import BeitFeatureExtractor, BeitForImageClassification
from torch.utils.data import DataLoader
import torch
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

model = BeitForImageClassification.from_pretrained("C:/Users/Thomas Houlachi/Documents/Reyclce rush/App/best_model_beit3")
feature_extractor = BeitFeatureExtractor.from_pretrained("C:/Users/Thomas Houlachi/Documents/Reyclce rush/App/best_extractor_beit3")
classifier = joblib.load('C:/Users/Thomas Houlachi/Documents/Reyclce rush/App/logistic_regression_model.joblib')

@app.route('/')
def index():
    # Serve the HTML page
    responseTime = session.get('responseTime', '')
    return render_template('scanner.html', responseTime=responseTime)
@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/recy')
def recy():
    return render_template('info.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/scanner')
def scanner():
    return render_template('scanner.html')

@app.route('/more')
def more():
    return render_template('more.html')

#changer nom def, et route
@app.route('/control_led', methods=['POST'])
def control_led():
    image_data = request.form['image']
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Prétraitez l'image selon les besoins (redimensionnement, normalisation, etc.)
    inputs = feature_extractor(img, return_tensors="pt")
    model.eval()
    # Effectuer une inférence
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = classifier.predict(logits.numpy())
    # Effectuez la prédiction avec votre modèle
    # prediction = model.predict(img)
    led_number = str(prediction[0])
    print
    if prediction == 0:
        prediction = "metal"
    elif prediction == 1:
        prediction = "cardboard"
    elif prediction == 2:
        prediction = "trash"
    print(prediction, led_number)
    try:
        # Dynamically manage the serial connection swithin the handler
        with serial.Serial('COM4', 9600, timeout=None) as arduino:
            time.sleep(2) # Wait for the connection to establish
            # Send the LED number to the Arduino, including a newline character
            arduino.write((led_number + '\n').encode())
            #is it possible this section is too fast^
            
            responseTime = arduino.read(2)


            responseTime=int.from_bytes(responseTime, byteorder="little")
            responseTime = responseTime/1000
            # Optionally, wait for a response here with arduino.readline()
            return render_template('scanner.html', responseTime=responseTime, prediction = prediction)
            #return ('', 204)  # Return an empty response to signify success
    except serial.SerialException as e:
        print(f"Failed to communicate with Arduino: {e}")
        return "Failed to communicate with Arduino", 500
    ################################################################################
  #Timer

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
