import os
import sys
import requests
import time
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from flask import Flask, jsonify, request, render_template
import readability_workflow


URL_BASE = "http://127.0.0.1:5000"
LARGE_MODEL = os.environ.get("LARGE_MODEL", False)


class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "default"


class RewriteForm(FlaskForm):
    sentence = TextAreaField("Original Sentence:")
    submit = SubmitField("Rewrite")


app = Flask(__name__)
app.config.from_object(Config)
api_headers = {"Content-Type": "application/json"}
workflow = readability_workflow.ReadabilityWorkflow(llm=LARGE_MODEL)


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", title="Home")


@app.route("/rewrite", methods=["GET", "POST"])
def rewrite():
    form = RewriteForm()
    res = None
    if request.method == "POST":
        prompt = request.form["sentence"]
        if len(prompt) > 0:
            start_time = time.time()
            res = requests.post(
                f"{URL_BASE}/predict", headers=api_headers, json={"prompt": prompt},
            ).json()
            end_time = time.time()
            print(
                f"Prediction response time: {end_time - start_time:.2f} seconds",
                file=sys.stdout,
            )
    return render_template("predict.html", title="Rewrite", form=form, res=res)


@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    res = workflow.predict(req["prompt"])
    return res["inferences"], 200


@app.route("/annotate", methods=["POST"])
def annotate():
    req = request.get_json()
    workflow.add_rated_response(req["annotations"])
    workflow.save_annotations()
    return jsonify({"response": "done"}), 200


if __name__ == "__main__":
    app.run(debug=True)
