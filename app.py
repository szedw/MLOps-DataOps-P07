from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd

from modules.insurance_model import InsuranceModel

app = Flask(__name__)

@app.route("/")
def home():
    return "API MODELING"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    result_predict = InsuranceModel().runModel(df, typed='single')

    return jsonify({
        "status":"predicted",
        "predict_result":result_predict
    })


if __name__ == "__main__":
    app.run()