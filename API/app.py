from flask import Flask
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)
CORS(app)


def predict_text(seed, length):

    dummy_prediction = 'a'*length
    text = seed + dummy_prediction

    return text


@app.route("/")
def hello():
    return "Hola, I'm Bot Quijote de la Mancha."


@app.route("/predict/<string:seed>/<int:length>", methods=["GET"])
def predict(seed, length):
    predicted_text = predict_text(seed, length)
    return jsonify(predicted_text)


if __name__ == '__main__':
    app.run(debug=True)
