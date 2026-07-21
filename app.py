import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'linear-model.pkl')

# Attempt to load the trained linear model gracefully
model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        model_error = f"Error unpickling model: {str(e)}"
else:
    model_error = f"Model file not found at path: {MODEL_PATH}. Make sure 'linear-model.pkl' is committed to Git."

# Feature list based on model architecture
FEATURE_NAMES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "waterfront present",
    "number of views",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year",
    "Renovation Year",
    "lot_area_renov",
    "Number of schools nearby",
    "Distance from the airport"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --input-bg: #0f172a;
            --error-bg: rgba(239, 68, 68, 0.1);
            --error-border: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .alert-error {
            background: var(--error-bg);
            border: 1px solid var(--error-border);
            color: #fca5a5;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
            text-align: center;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: capitalize;
        }

        .input-group input {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease-in-out;
        }

        .input-group input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background-color: var(--accent);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0.9rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease-in-out;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:disabled {
            background-color: var(--border-color);
            cursor: not-allowed;
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--accent);
            border-radius: 12px;
            text-align: center;
        }

        .result-card h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .result-card .price {
            font-size: 2.25rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>House Price Prediction</h1>
            <p>Enter property metrics below to estimate value</p>
        </div>

        {% if model_error %}
        <div class="alert-error">
            <strong>Deployment Issue:</strong> {{ model_error }}
        </div>
        {% endif %}

        <form method="POST" action="/predict" class="grid-form">
            {% for feature in features %}
            <div class="input-group">
                <label for="{{ loop.index0 }}">{{ feature }}</label>
                <input 
                    type="number" 
                    step="any" 
                    id="{{ loop.index0 }}" 
                    name="feature_{{ loop.index0 }}" 
                    value="{{ form_data.get('feature_' ~ loop.index0, '') }}"
                    required 
                    placeholder="0.0">
            </div>
            {% endfor %}

            <button type="submit" class="btn-submit" {% if model_error %}disabled{% endif %}>
                Calculate Estimated Value
            </button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h2>Estimated Property Value</h2>
            <div class="price">${{ "{:,.2f}".format(prediction) }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURE_NAMES, 
        prediction=None, 
        form_data={}, 
        model_error=model_error
    )

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            features=FEATURE_NAMES, 
            prediction=None, 
            form_data=request.form, 
            model_error=model_error
        )

    try:
        form_data = request.form
        features_input = [float(form_data[f'feature_{i}']) for i in range(len(FEATURE_NAMES))]
        prediction = model.predict([features_input])[0]
        return render_template_string(
            HTML_TEMPLATE, 
            features=FEATURE_NAMES, 
            prediction=prediction, 
            form_data=form_data, 
            model_error=None
        )
    except Exception as e:
        return f"Error processing prediction: {str(e)}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
