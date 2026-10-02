"""
app.py -- Vehicle Price Prediction (ElasticNet)

Loads model.pkl (a dict bundle containing the fitted model, scaler,
feature_columns, and category mappings saved from the training
notebook) and serves predictions through a simple HTML form.

Run with:
    python app.py
Then open http://127.0.0.1:5000/ in a browser.
"""

import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the trained bundle once, at startup
# ---------------------------------------------------------------
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
num_doors_map = bundle["num_doors_map"]
num_cylinders_map = bundle["num_cylinders_map"]
categorical_cols = bundle["categorical_cols"]

# Dropdown options shown on the form -- taken directly from the
# training data's own categories, so the form can never submit a
# category the model wasn't trained on.
MAKE_OPTIONS = ['alfa-romero','audi','bmw','chevrolet','dodge','honda','isuzu',
                 'jaguar','mazda','mercedes-benz','mercury','mitsubishi','nissan',
                 'peugot','plymouth','porsche','renault','saab','subaru','toyota',
                 'volkswagen','volvo']
FUEL_TYPE_OPTIONS = ['gas', 'diesel']
ASPIRATION_OPTIONS = ['std', 'turbo']
BODY_STYLE_OPTIONS = ['convertible', 'hardtop', 'hatchback', 'sedan', 'wagon']
DRIVE_WHEELS_OPTIONS = ['rwd', 'fwd', '4wd']
ENGINE_LOCATION_OPTIONS = ['front', 'rear']
ENGINE_TYPE_OPTIONS = ['dohc', 'ohc', 'ohcf', 'ohcv', 'rotor']
FUEL_SYSTEM_OPTIONS = ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi']
NUM_DOORS_OPTIONS = ['two', 'four']
NUM_CYLINDERS_OPTIONS = ['two', 'three', 'four', 'five', 'six', 'eight', 'twelve']


def preprocess(form):
    """
    Rebuild, from a single raw form submission, the EXACT same feature
    representation the model was trained on in the notebook:
      1. map num-of-doors / num-of-cylinders text -> numbers
      2. one-hot encode the categorical columns
      3. reindex against the saved feature_columns (fills any column
         the model expects but this one row doesn't produce with 0 --
         this is the safe equivalent of OneHotEncoder(handle_unknown=
         'ignore'), needed here because training used pd.get_dummies,
         which has no memory of its own)
      4. scale with the SAME fitted scaler used in training
    """
    row = {
        "symboling": float(form["symboling"]),
        "normalized-losses": float(form["normalized-losses"]),
        "num-of-doors": form["num-of-doors"],
        "wheel-base": float(form["wheel-base"]),
        "length": float(form["length"]),
        "width": float(form["width"]),
        "height": float(form["height"]),
        "curb-weight": float(form["curb-weight"]),
        "num-of-cylinders": form["num-of-cylinders"],
        "engine-size": float(form["engine-size"]),
        "bore": float(form["bore"]),
        "stroke": float(form["stroke"]),
        "compression-ratio": float(form["compression-ratio"]),
        "horsepower": float(form["horsepower"]),
        "peak-rpm": float(form["peak-rpm"]),
        "highway-mpg": float(form["highway-mpg"]),
        "make": form["make"],
        "fuel-type": form["fuel-type"],
        "aspiration": form["aspiration"],
        "body-style": form["body-style"],
        "drive-wheels": form["drive-wheels"],
        "engine-location": form["engine-location"],
        "engine-type": form["engine-type"],
        "fuel-system": form["fuel-system"],
    }

    df = pd.DataFrame([row])

    # Same two text->number mappings used in training
    df["num-of-doors"] = df["num-of-doors"].replace(num_doors_map)
    df["num-of-cylinders"] = df["num-of-cylinders"].replace(num_cylinders_map)

    # Same one-hot encoding step used in training
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    # Reindex to the EXACT columns/order the scaler and model expect.
    # Any dummy column this single row didn't produce (because the
    # other categories aren't present in a 1-row frame) is added back
    # as 0 -- this is what prevents the KeyError we hit on the last
    # project's deployment.
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale with the SAME fitted scaler object used in training
    scaled = scaler.transform(df_encoded)
    return scaled


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        makes=MAKE_OPTIONS, fuel_types=FUEL_TYPE_OPTIONS, aspirations=ASPIRATION_OPTIONS,
        body_styles=BODY_STYLE_OPTIONS, drive_wheels=DRIVE_WHEELS_OPTIONS,
        engine_locations=ENGINE_LOCATION_OPTIONS, engine_types=ENGINE_TYPE_OPTIONS,
        fuel_systems=FUEL_SYSTEM_OPTIONS, num_doors=NUM_DOORS_OPTIONS,
        num_cylinders=NUM_CYLINDERS_OPTIONS, prediction=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = preprocess(request.form)
        price = model.predict(X)[0]
        prediction = f"${price:,.2f}"
    except Exception as e:
        prediction = f"Error: {e}"

    return render_template(
        "index.html",
        makes=MAKE_OPTIONS, fuel_types=FUEL_TYPE_OPTIONS, aspirations=ASPIRATION_OPTIONS,
        body_styles=BODY_STYLE_OPTIONS, drive_wheels=DRIVE_WHEELS_OPTIONS,
        engine_locations=ENGINE_LOCATION_OPTIONS, engine_types=ENGINE_TYPE_OPTIONS,
        fuel_systems=FUEL_SYSTEM_OPTIONS, num_doors=NUM_DOORS_OPTIONS,
        num_cylinders=NUM_CYLINDERS_OPTIONS, prediction=prediction,
        form_values=request.form
    )


if __name__ == "__main__":
    app.run(debug=True)
