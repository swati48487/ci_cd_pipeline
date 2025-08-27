import os
import numpy as np
import joblib
from flask import Flask, request, jsonify

#---config----
MODEL_PATH = os.getenv("MODEL_PATH", "kmean_mall.pkl")  # adjust filename if needed

#--App--
app = Flask(__name__)  # ✅ FIXED: proper Flask app initialization

# Load model once
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    """
    Accepts either:
    {"input": [[...feature vector...], [...]]}  # 2D list
    or
    {"input": [...feature vector...]}           # 1D list
    """
    try:
        payload = request.get_json(force=True)
        x = payload.get("input")
        if x is None:
            return jsonify(error='Missing "input"'), 400  # ✅ FIXED: quote error

        # Normalize to 2D array
        if isinstance(x, list) and len(x) > 0 and not isinstance(x[0], list):
            x = [x]

        X = np.array(x, dtype=float)
        preds = model.predict(X).tolist()

        # Optional: map cluster labels to customer segments
        cluster_map = {
            0: "Budget Shopper",
            1: "Mid-range Customer",
            2: "Luxury Spender"
        }
        mapped_preds = [cluster_map.get(p, f"Cluster {p}") for p in preds]

        return jsonify(prediction=mapped_preds), 200

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # Local dev only; Render will run with Gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))