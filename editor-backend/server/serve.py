from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"response": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True)
