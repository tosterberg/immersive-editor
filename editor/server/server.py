import os
import sys
import requests
import time
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, BooleanField
from flask import Flask, jsonify, request, render_template
from sentence_splitter import SentenceSplitter
import readability_workflow


URL_BASE = "http://127.0.0.1:5000"
# Setting the following to True will load a falcon-7b model, adding 2 minutes to server startup, and about 1 minute
# to prediction response time for sentence rewrites. Using a larger GPU with batch encoding for parallel
# prediction would significantly speed this up, but local hardware doesn't have the space to support it.
LARGE_MODEL = os.environ.get("USE_LARGE_MODEL", False)


class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "default"


class RewriteForm(FlaskForm):
    sentence = TextAreaField("Original Text:")
    submit = SubmitField("Run")


class AnnotateForm(FlaskForm):
    accept_1 = BooleanField("Approve")
    accept_2 = BooleanField("Approve")
    accept_3 = BooleanField("Approve")
    accept_4 = BooleanField("Approve")
    submit = SubmitField("Annotate")


app = Flask(__name__)
app.config.from_object(Config)
api_headers = {"Content-Type": "application/json"}
workflow = readability_workflow.ReadabilityWorkflow(llm=LARGE_MODEL)
work_state = dict()


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", title="Home")


@app.route("/rewrite", methods=["GET", "POST"])
def rewrite():
    form = RewriteForm()
    annotate = AnnotateForm()
    res = None
    if request.method == "POST":
        if "sentence" in request.form.keys():
            prompt = request.form["sentence"]
            if len(prompt) > 0:
                work_state["prompt"] = prompt
                res = requests.post(
                    f"{URL_BASE}/predict", headers=api_headers, json={"prompt": prompt},
                ).json()
                work_state["inferences"] = res["inferences"]
        elif "accept" in request.form.keys():
            for i in range(4):
                if str(i + 1) in request.form.keys():
                    work_state["inferences"][i]["chosen"] = True
                res = requests.post(f"{URL_BASE}/annotate")
    return render_template(
        "predict.html",
        title="Rewrite",
        form=form,
        annotate=annotate,
        work_state=work_state,
        res=res,
    )


@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    res = workflow.predict(req["prompt"])
    return res, 200


@app.route("/readability", methods=["GET", "POST"])
def readability():
    form = RewriteForm()
    res = None
    if request.method == "POST":
        splitter = SentenceSplitter(language="en")
        prompt = request.form["sentence"]
        sentences = splitter.split(text=prompt)
        read = workflow.get_readability(sentences)
        res = dict({"sentences": []})
        for i in range(len(read)):
            res["sentences"].append({"sentence": sentences[i], "readability": read[i]})
        res = res["sentences"]
    return render_template("readability.html", form=form, res=res)


@app.route("/annotate", methods=["POST"])
def annotate():
    req = request.get_json()
    workflow.add_rated_response(req["annotations"])
    workflow.save_annotations()
    return jsonify({"response": "done"}), 200


def get_state():
    global work_state
    return work_state


def clear_state():
    global work_state
    work_state = dict()
    return work_state


if __name__ == "__main__":
    app.run(debug=True)
