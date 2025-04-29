import random
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

MIN_MULTIPLIER = 10.0

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/predictor_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Models
class HistoricalData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    multiplier = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    predicted_time = db.Column(db.String(20), nullable=False)
    interval = db.Column(db.Integer, nullable=False)
    predicted_min = db.Column(db.Float, nullable=False)
    predicted_max = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'))
    is_accurate = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Helpers
def classify_band(minutes):
    if minutes <= 2:
        return "Fast"
    elif minutes <= 5:
        return "Medium"
    else:
        return "Slow"

def generate_interval_from_band(band):
    if band == "Fast":
        return random.randint(1, 2)
    elif band == "Medium":
        return random.randint(3, 5)
    else:
        return random.randint(6, 10)

def calculate_intervals(entries):
    intervals = []
    prev_time = None
    for entry in entries:
        if prev_time:
            delta = (entry.created_at - prev_time).total_seconds() / 60
            intervals.append(int(round(delta)))
        prev_time = entry.created_at
    return intervals

def find_similar_patterns(current_pattern, all_intervals, tolerance=2):
    matches = []
    for i in range(len(all_intervals) - len(current_pattern) - 1):
        if all(abs(all_intervals[i+j] - current_pattern[j]) <= tolerance for j in range(len(current_pattern))):
            matches.append(i)
    return matches

def find_similar_multiplier_patterns(current_multipliers, all_multipliers, window_size=5, tolerance=5.0):
    matches = []
    for i in range(len(all_multipliers) - window_size - 1):
        window = all_multipliers[i:i+window_size]
        if all(abs(window[j] - current_multipliers[j]) <= tolerance for j in range(window_size)):
            matches.append(i)
    return matches

def calculate_range(values, all_values):
    if len(values) < 2:
        return (MIN_MULTIPLIER, MIN_MULTIPLIER + 10.0)

    matches = find_similar_multiplier_patterns(values[-5:], all_values)
    
    if matches:
        next_multipliers = [all_values[i+5] for i in matches if i+5 < len(all_values)]
        if next_multipliers:
            predicted_avg = np.mean(next_multipliers)
            predicted_std = np.std(next_multipliers)
            min_val = max(MIN_MULTIPLIER, predicted_avg - predicted_std)
            max_val = max(min_val + 1, predicted_avg + predicted_std)
            return (min_val, max_val)
    
    model = LinearRegression().fit(np.arange(len(values)).reshape(-1, 1), values)
    next_val = model.predict([[len(values)]])[0]
    std = np.std(values)
    min_val = max(MIN_MULTIPLIER, next_val - std)
    max_val = max(min_val + 1, next_val + std)

    return (min_val, max_val)

def get_confidence_icon(conf):
    conf = int(conf)
    if conf >= 90: return f"🛡️ High Safety ({conf}%)"
    elif conf >= 70: return f"✅ Reliable ({conf}%)"
    elif conf >= 50: return f"⚠️ Caution ({conf}%)"
    elif conf >= 30: return f"🔥 Volatile ({conf}%)"
    return f"💎 Speculative ({conf}%)"

def calculate_confidence(current_pattern, all_intervals):
    matches = find_similar_patterns(current_pattern, all_intervals)
    
    if not matches:
        return 1  # No matches, lowest confidence
    
    deviations = []
    for idx in matches:
        historical_pattern = all_intervals[idx:idx+len(current_pattern)]
        deviation = np.mean([abs(h - c) for h, c in zip(historical_pattern, current_pattern)])
        deviations.append(deviation)
    
    if not deviations:
        return 1

    avg_deviation = np.mean(deviations)
    confidence = max(1, min(100, int(100 - avg_deviation * 10)))  # You can adjust 10 if needed
    return confidence


def build_transition_matrix(intervals):
    bands = [classify_band(i) for i in intervals]
    transitions = {
        "Fast": {"Fast": 0, "Medium": 0, "Slow": 0},
        "Medium": {"Fast": 0, "Medium": 0, "Slow": 0},
        "Slow": {"Fast": 0, "Medium": 0, "Slow": 0}
    }
    
    for i in range(len(bands) - 1):
        current_band = bands[i]
        next_band = bands[i+1]
        transitions[current_band][next_band] += 1

    transition_probs = {}
    for band, counts in transitions.items():
        total = sum(counts.values())
        if total > 0:
            transition_probs[band] = {k: v/total for k, v in counts.items()}
        else:
            transition_probs[band] = {k: 1/3 for k in counts.keys()}
    
    return transition_probs

def predict_next_band(current_band, transition_probs):
    bands = list(transition_probs[current_band].keys())
    probs = list(transition_probs[current_band].values())
    return random.choices(bands, weights=probs)[0]

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    entries = HistoricalData.query.order_by(HistoricalData.created_at).all()
    if len(entries) < 10:
        return jsonify({"error": "Need at least 10 entries"}), 400

    intervals = calculate_intervals(entries)
    multipliers = [e.multiplier for e in entries]

    current_pattern = intervals[-5:] if len(intervals) >= 5 else [5] * 5
    confidence = calculate_confidence(current_pattern, intervals)
    confidence_display = get_confidence_icon(confidence)

    transition_probs = build_transition_matrix(intervals)

    current_band = classify_band(intervals[-1]) if intervals else "Fast"
    predicted_intervals = []
    for _ in range(10):
        next_band = predict_next_band(current_band, transition_probs)
        interval = generate_interval_from_band(next_band)
        predicted_intervals.append(interval)
        current_band = next_band

    last_time = datetime.now()
    last_values = multipliers[-5:]
    all_values = multipliers

    results = []
    for interval in predicted_intervals:
        last_time += timedelta(minutes=interval)
        min_val, max_val = calculate_range(last_values, all_values)
        mid_val = (min_val + max_val) / 2

        prediction = Prediction(
            predicted_time=last_time.strftime('%H:%M'),
            interval=interval,
            predicted_min=min_val,
            predicted_max=max_val,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.flush()

        results.append({
            "id": prediction.id,
            "time": prediction.predicted_time,
            "interval": interval,
            "min": f"{min_val:.2f}x",
            "max": f"{max_val:.2f}x",
            "mid": f"{mid_val:.2f}x",
            "confidence": confidence_display
        })

        last_values.append(mid_val)

    db.session.commit()
    return jsonify(results)

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    try:
        feedback = Feedback(
            prediction_id=data['prediction_id'],
            is_accurate=data['is_accurate']
        )
        db.session.add(feedback)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def system_status():
    try:
        record_count = HistoricalData.query.count()
        online = record_count > 0
        return jsonify({"online": online, "records": record_count})
    except Exception:
        return jsonify({"online": False, "records": 0})

if __name__ == '__main__':
    app.run(debug=True)
