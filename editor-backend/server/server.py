from flask import Flask, jsonify, request
from server.readability_workflow import ReadabilityWorkflow

app = Flask(__name__)
workflow = ReadabilityWorkflow()


@app.route("/")
def index():
    return jsonify({"response": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    res = workflow.predict(req["prompt"])
    return jsonify(res), 200


if __name__ == "__main__":
    app.run(debug=True)
