from flask import Flask, request, jsonify
import plotly.graph_objects as go  
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return "The server is running"

model = joblib.load("waste_model.pkl")

def _parse(data):
    try:
        students = float(data["students_enrolled"])
        attendance = float(data["attendance_percent"])
        menu_count = float(data["menu_count"])
        leftover = float(data["previous_day_leftover_kg"])
        nonveg = float(data["nonveg_items"])
        meal_type = 1 if data["meal_type"] == "dinner" else 0
        day = 0

        people = round(students * attendance / 100)
        main_veg = people * 0.1
        main_nonveg = people * 0.08 if nonveg else 0

        vector = np.array([[students, attendance, 0, menu_count, leftover,
                            nonveg, meal_type, day, main_nonveg, main_veg, people]])
        
        return vector, []
    
    except Exception as e:
        return None, [str(e)]
    
@app.route("/predict_graph", methods=["GET","POST"])
def predict_graph():
    if request.method == "GET":
        days = ["Monday","Tuesday","Wednesday","Thursday"]
        waste_values = [10,12,9,11]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x = days,
                y = waste_values,
                mode="lines+markers",
                name="Predicted Waste"
            )
        )

        fig.update_layout(
            title = "Food Waste Trend",
            xaxis_title = "Day",
            yaxis_title = "Predicted Waste (kg)"
        )

        return fig.to_html()
    
    data = request.get_json()
    if not data or "data" not in data:
        return jsonify({"error": "Invalid"}), 400

    days = []
    waste_values = []

    for entry in data["data"]:
        vector, errors = _parse(entry)

        if errors:
            continue

        waste = float(model.predict(vector)[0])

        days.append(entry["day"])
        waste_values.append(round(waste,4))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=days,
            y=waste_values,
            mode="lines+markers",
            name="Predicted Waste"
        )
    )

    fig.update_layout(
        title = "Food Waste Trend",
        xaxis_title = "Day",
        yaxis_title = "Predicted Waste (kg)"
    )

    return fig.to_html()

if __name__ == "__main__":
    app.run(debug=True)