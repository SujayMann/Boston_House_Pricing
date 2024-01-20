import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = float(regmodel.predict(final_input)[0])
    # round(output, 4)
    return render_template('home.html', prediction_text="The predicted house price is {:.4f} ($1000s)".format(output))

if __name__ == "__main__":
    try:
        app.run(debug=False, port=4000)
    except Exception as e:
        print(e)