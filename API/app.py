from flask import Flask

app = Flask(__name__)


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
    return predicted_text


if __name__ == '__main__':
    app.run(debug=True)
