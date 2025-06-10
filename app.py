from flask import Flask, request, render_template
import pandas as pd
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("solar_still_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load and preprocess radiation data
df = pd.read_csv("365_day_radiation_based_water_output.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = df['Date'].dt.dayofyear

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    radiation = None
    water = None
    date_str = None

    if request.method == "POST":
        try:
            date_str = request.form.get("date")
            water = float(request.form.get("water"))

            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            day_of_year = date_obj.timetuple().tm_yday

            # Get radiation for that day
            radiation = df[df["DayOfYear"] == day_of_year]["Radiation (W/m²)"].mean()

            if pd.isna(radiation):
                prediction = "No radiation data for this date."
            else:
                # Prepare data
                input_data = pd.DataFrame([[day_of_year, radiation, water * 1000]],  # Liters to ml
                                          columns=["DayOfYear", "Radiation (W/m²)", "Input Water (ml)"])
                input_scaled = scaler.transform(input_data)

                # Make prediction
                prediction = round(model.predict(input_scaled)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html",
                           prediction=prediction,
                           radiation=round(radiation, 2) if radiation else None,
                           date=date_str,
                           water=water)

# Use 0.0.0.0 for cloud deployment compatibility
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
