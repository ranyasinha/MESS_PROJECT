from flask import Flask, request, jsonify
import plotly.graph_objects as go
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("waste_model.pkl")

def get_risk(waste):
    if waste <= 10:
        return "Low"
    elif waste <= 20:
        return "Medium"
    else:
        return "High"
    

def _parse(data):
    try:
        students = float(data["students_enrolled"])
        attendance = float(data["attendance_percent"])
        menu_count = float(data["menu_count"])
        leftover = float(data["previous_day_leftover_kg"])
        nonveg = float(data["nonveg_items"])

        meal_type = 1 if data["meal_type"] == "dinner" else 0

        day_map = {
            "Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6
        }

        day = day_map.get(data["day"],0)

        people = round(students * attendance / 100)
        main_veg = people * 0.1
        main_nonveg = people * 0.08 if nonveg else 0

        vector = np.array([[students, attendance, 0, menu_count, leftover, nonveg, meal_type, day, main_nonveg, main_veg, people]])

        return vector, []
    
    except Exception as e:
        return None, [str(e)]
    
@app.route("/")
def home():
    return """
    <h2> Food Waste Dashboard</h2>
    <a href="/predict_gauge">Opening Guage Chart</a>
    """

@app.route("/predict_gauge", methods=["GET", "POST"])
def predict_gauge():
    if request.method == "GET":
        waste = 12.5
    else:
        data = request.get_json()
        entry = data["data"][0]

        vector, errors = _parse(entry)

        if errors:
            return jsonify({"errors":errors})
        
        waste = float(model.predict(vector)[0])
        waste = round(waste, 2)

    risk = get_risk(waste)

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = waste,
        number = {'font' : {'size': 60}},
        title = {'text' : "Predicted Food Waste (kg)", 'font' : {'size' : 24}},
        gauge = {
            'axis' : {'range' : [0,30]},
            'bar' : {'color' : "black"},
            'steps' : [
                {'range' : [0,10], 'color' : "green"},
                {'range' : [10,20], 'color' : "yellow"},
                {'range' : [20,30], 'color' : "red"}
            ]
        }
    ))
        
    fig.add_annotation(
        text = f"<b>Risk Level: {risk}</b>",
        x = 0.5,
        y = 0.85,
        showarrow = False,
        font = dict(size = 22, color = "black"),

    )

    fig.add_trace(go.Scatter(
        x = [waste],
        y = [0],
        mode = "markers",
        marker = dict(size = 20, color = "rgba(0,0,0,0)"),
        showlegend = False
    ))

    fig.add_trace(go.Scatter(x = [None], y = [None], mode = 'markers', marker = dict(size = 12, color = "green"), name = 'Low (0-10 kg)'))

    fig.add_trace(go.Scatter(x = [None], y = [None], mode = 'markers', marker = dict(size=12, color = 'yellow'), name = 'Medium (10-20 kg)'))

    fig.add_trace(go.Scatter(x = [None], y = [None], mode = 'markers', marker = dict(size = 12, color = 'red'), name = 'High (20-30 kg)'))

    fig.update_layout(legend = dict(x = 1.05, y = 0.9), margin = dict(l = 50, r = 150, t = 100, b = 50))

    return fig.to_html()

if __name__ == "__main__":
    app.run(debug=True)