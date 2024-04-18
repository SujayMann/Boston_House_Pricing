import pickle
from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
tree_model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    print(data, "hello")
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = float(tree_model.predict(final_input)[0])
    return render_template('home.html', prediction_text="The predicted house price is {:.3f} ($1000s)".format(output))

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(e)