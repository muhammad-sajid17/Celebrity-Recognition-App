from flask import Flask, request, jsonify
from flask_cors import CORS
from Deployment import predict_celebrity  # <-- updated function
import base64

app = Flask(__name__)
CORS(app)  # allow all origins, for testing only


@app.route("/")
def home():
    return jsonify({"message": "Celebrity Recognition API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expect JSON with base64 string
        data = request.get_json()

        if not data or "image_b64" not in data:
            return jsonify({"error": "No base64 image string provided"}), 400

        b64_str = data["image_b64"]

        # Call predictor
        result = predict_celebrity(b64_str)
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
