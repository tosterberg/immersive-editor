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


@app.route("/save")
def save():
    workflow.save_annotations()
    return jsonify({"response": "done"}), 200


@app.route("/annotate", methods=["POST"])
def annotate():
    req = request.get_json()
    print(req)
    workflow.add_rated_response(req["annotations"])
    return save()


if __name__ == "__main__":
    app.run(debug=True)
