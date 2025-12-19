from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "knn_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    prediction_class = None # For styling (risk vs no risk)

    if request.method == "POST":
        try:
            # Get values from form
            bp = float(request.form.get("bp"))
            chol = float(request.form.get("cholestrol"))

            # Create a DataFrame to avoid feature name warnings
            features = pd.DataFrame([[bp, chol]], columns=['BP', 'Cholestrol'])

            # Make prediction
            prediction = model.predict(features)[0]

            if prediction == 1:
                prediction_text = "⚠️ Heart Risk Detected"
                prediction_class = "danger"
            else:
                prediction_text = "✅ No Heart Risk Detected"
                prediction_class = "success"
                
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
            prediction_class = "secondary"

    return render_template("index.html", 
                           prediction_text=prediction_text, 
                           prediction_class=prediction_class)

if __name__ == "__main__":
    app.run(debug=True)
