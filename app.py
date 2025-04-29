import random
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # replace this securely in production

# === DATABASE CONFIG ===
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://1223707:BRAGyh12345@localhost/1223707"
  # <-- UPDATE THIS
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)

# === MODELS ===
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))  # hashed in production

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === HELPER FUNCTIONS ===
def classify_band(minutes):
    if minutes <= 2: return "Fast"
    elif minutes <= 5: return "Medium"
    return "Slow"

def predict_next_band(current_band):
    options = {"Fast": ["Fast", "Medium"], "Medium": ["Fast", "Medium"], "Slow": ["Fast", "Medium", "Slow"]}
    weights = {"Fast": [0.6, 0.4], "Medium": [0.7, 0.3], "Slow": [0.5, 0.3, 0.2]}
    return random.choices(options[current_band], weights[current_band])[0]

def generate_interval_from_band(band):
    return random.randint(1, 2) if band == "Fast" else random.randint(3, 5) if band == "Medium" else random.randint(6, 10)

def calculate_intervals(entries):
    intervals, prev = [], None
    for e in entries:
        if prev: intervals.append(int(round((e.created_at - prev).total_seconds() / 60)))
        prev = e.created_at
    return intervals

def find_similar_patterns(current, all_, tolerance=2):
    return [i for i in range(len(all_) - len(current) - 1)
            if all(abs(all_[i+j] - current[j]) <= tolerance for j in range(len(current)))]

def find_similar_multiplier_patterns(current, all_, window=5, tolerance=5.0):
    return [i for i in range(len(all_) - window - 1)
            if all(abs(all_[i+j] - current[j]) <= tolerance for j in range(window))]

def calculate_range(values, all_values):
    if len(values) < 2: return (10.0, 20.0)
    matches = find_similar_multiplier_patterns(values[-5:], all_values)
    if matches:
        next_vals = [all_values[i+5] for i in matches if i+5 < len(all_values)]
        if next_vals:
            avg, std = np.mean(next_vals), np.std(next_vals)
            return max(10.0, avg - std), max(avg - std + 1, avg + std)
    model = LinearRegression().fit(np.arange(len(values)).reshape(-1, 1), values)
    pred = model.predict([[len(values)]])[0]
    std = np.std(values)
    return max(10.0, pred - std), max(pred - std + 1, pred + std)

def get_confidence_icon(conf):
    conf = int(conf)
    if conf >= 90: return f"🛡️ High Safety ({conf}%)"
    elif conf >= 70: return f"✅ Reliable ({conf}%)"
    elif conf >= 50: return f"⚠️ Caution ({conf}%)"
    elif conf >= 30: return f"🔥 Volatile ({conf}%)"
    return f"💎 Speculative ({conf}%)"

def calculate_confidence(pattern, intervals):
    matches = find_similar_patterns(pattern, intervals)
    return min(100, max(1, int(len(matches) * 100 / (len(intervals) or 1))))

# === ROUTES ===
@app.route('/')
@login_required
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.password == request.form['password']:
            login_user(user)
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    return render_template("login.html")

@app.route('/logout')
def logout():
    logout_user()
    return redirect('/login')

@app.route('/predict')
@login_required
def predict():
    entries = HistoricalData.query.order_by(HistoricalData.created_at).all()
    if len(entries) < 10:
        return jsonify({"error": "Need at least 10 entries"}), 400

    intervals = calculate_intervals(entries)
    multipliers = [e.multiplier for e in entries]
    current_pattern = intervals[-5:] if len(intervals) >= 5 else [5] * 5
    confidence = calculate_confidence(current_pattern, intervals)
    confidence_display = get_confidence_icon(confidence)
    predicted_intervals = [generate_interval_from_band(predict_next_band(classify_band(intervals[-1] if intervals else 2))) for _ in range(10)]

    last_time = datetime.now()
    last_values = multipliers[-5:]
    results = []

    for interval in predicted_intervals:
        last_time += timedelta(minutes=interval)
        min_val, max_val = calculate_range(last_values, multipliers)
        mid_val = (min_val + max_val) / 2
        prediction = Prediction(predicted_time=last_time.strftime('%H:%M'), interval=interval,
                                predicted_min=min_val, predicted_max=max_val, confidence=confidence)
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
@login_required
def handle_feedback():
    data = request.get_json()
    feedback = Feedback(prediction_id=data['prediction_id'], is_accurate=data['is_accurate'])
    db.session.add(feedback)
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/add', methods=['POST'])
@login_required
def add_entry():
    data = request.get_json()
    exists = HistoricalData.query.filter_by(date=data['date'], time=data['time'], multiplier=data['multiplier']).first()
    if exists:
        return jsonify({"status": "exists"}), 409
    entry = HistoricalData(date=data['date'], time=data['time'], multiplier=data['multiplier'])
    db.session.add(entry)
    db.session.commit()
    return jsonify({"status": "success"}), 201

@app.route('/high-risk')
@login_required
def high_risk_prediction():
    now = datetime.now()
    high_entries = HistoricalData.query.filter(HistoricalData.multiplier >= 500).all()
    time_counter = Counter(e.time[:5] for e in high_entries)
    common_times = time_counter.most_common()
    upcoming = []

    for t_str, count in common_times:
        h, m = map(int, t_str.split(":"))
        future = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if future > now:
            upcoming.append({"time": future.strftime("%H:%M"), "chance": min(100, int(count * 100 / len(high_entries)))})
        if len(upcoming) >= 5:
            break

    return jsonify(upcoming)

if __name__ == '__main__':
    app.run(debug=True)
