# from pyparsing import null_debug_action
from flask import Flask, render_template, request

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
import json

import warnings
warnings.filterwarnings('ignore')

#import numpy as np
app = Flask(__name__)

with open("savedModel/crop_mapping.json") as cf:
    crop_classes = json.load(cf)
    cf.close()

scaler_loaded = joblib.load("savedModel/scaler.save")

crop_file = open('savedModel/Crop_Recommendation.pkl', 'rb')
crop_model = pickle.load(crop_file)

@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/sub", methods=["POST"])
def submit():
    if request.method == "POST":
        n = request.form["nitrogen"]
        p = request.form["phosphorus"]
        k = request.form["potassium"]
        t = request.form["temperature"]
        h = request.form["humidity"]
        ph = request.form["ph"]
        r = request.form["rainfall"]

        a = np.array([n, p, k, t, h, ph, r])
        a = a.reshape(1, 7)

        a = scaler_loaded.transform(a)
        a = np.array(a)

        prediction = crop_model.predict(a)
        # answer = prediction.argmax()

        result = []
        for p in prediction:
            p = list(p)
            result = [p.index(m) for m in sorted(p, reverse=True)][:5]
            print(result)

        answer = [crop_classes.get(str(a)) for a in result]
        img1 = "static/images/"+answer[0]+".jpg"
        img2 = "static/images/"+answer[1]+".jpg"
        img3 = "static/images/"+answer[2]+".jpg"
        img4 = "static/images/"+answer[3]+".jpg"
        img5 = "static/images/"+answer[4]+".jpg"
    return render_template("submit.html", crop1=answer[0], 
    crop2=answer[1], crop3=answer[2], crop4=answer[3], crop5=answer[4], 
    img1=img1, img2=img2, img3=img3, img4=img4, img5=img5)


if __name__ == "__main__":
    app.run(debug=True)
