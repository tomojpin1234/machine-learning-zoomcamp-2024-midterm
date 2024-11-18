import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Parameters

output_file_full = "models/model_full.bin"
output_file_lite = "models/model_lite.bin"

# Load the CSV once to avoid repeated loading
test_lite_file = "data/df_test_lite.csv"
test_full_file = "data/df_test_full.csv"

test_houses_lite = pd.read_csv(test_lite_file, encoding="ISO-8859-2")
test_houses_full = pd.read_csv(test_full_file, encoding="ISO-8859-2")


app = Flask(__name__)
CORS(app)


def get_features(model_type: str):
    categorical = ["city"]
    numerical = [
        "floor",
        "latitude",
        "longitude",
        "rooms",
        "sq",
    ]    
    if model_type == "full":
        categorical.append("address")
        numerical.append("year")
    return categorical, numerical


def load_model(output_file: str):
    with open(output_file, "rb") as f_in:
        scaler, ohe, model = pickle.load(f_in)
    return scaler, ohe, model


def predict_house_price(df, model_type: str):
    categorical, numerical = get_features(model_type)
    output_file = output_file_full if model_type == "full" else output_file_lite
    scaler, ohe, model = load_model(output_file)
    X_val_num = df[numerical].values
    X_val_num = scaler.transform(X_val_num)
    X_val_cat = ohe.transform(df[categorical].values)
    X_val = np.column_stack([X_val_num, X_val_cat])

    y_pred = model.predict(X_val)
    return y_pred


@app.route("/random-house", methods=["GET"])
def random_house():
    # Get model type from query parameters
    model_type = request.args.get(
        "model_type", "lite"
    )  # Default to "lite" if not specified

    # Choose the correct DataFrame based on model type
    if model_type == "full":
        houses = test_houses_full
    else:
        houses = test_houses_lite

    # Select a random row from the DataFrame
    random_row = houses.sample(n=1).iloc[0]
    random_house_data = random_row.to_dict()
    return jsonify(random_house_data)


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        # Parse JSON data
        house = request.get_json()
    else:
        # Parse form data
        house = request.form.to_dict()

    print("predict")
    print("house: ", house)
    model_type = house.get("model_type", "lite")

    df_house = pd.DataFrame([house])

    y_pred = predict_house_price(df_house, model_type)
    price_predicted = np.expm1(y_pred[0])

    print(f"Predicted price is {price_predicted}")
    result = {"house_price": price_predicted}
    return jsonify(result)


# Define the form route
@app.route("/", methods=["GET"])
def form():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
