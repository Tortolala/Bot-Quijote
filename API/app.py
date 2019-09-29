from flask import Flask
from flask_cors import CORS
from flask import jsonify
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False


train_on_gpu = torch.cuda.is_available()
lr = 0.001
n_hidden = 512
n_layers = 4


class QuijoteModel(nn.Module):

    def __init__(self, tokens, n_hidden, n_layers, l=0.001):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars), n_hidden,
                            n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        out, hidden_state = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden_state


with open('quijotemodel.net', 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')

loaded = QuijoteModel(
    checkpoint['characters'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])


def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.0
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def predict_char(net, char, h=None, top_k=None):

    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if(train_on_gpu):
        inputs = inputs.cuda()

    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

    p = F.softmax(out, dim=1).data
    if(train_on_gpu):
        p = p.cpu()

    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    return net.int2char[char], h


def predict_with_model(net, size, prime='example', top_k=None):

    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval()

    chars = [ch for ch in prime]

    weight = next(net.parameters()).data
    h = (weight.new(n_layers, 1, n_hidden).zero_(),
         weight.new(n_layers, 1, n_hidden).zero_())

    for ii in range(size):
        char, h = predict_char(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


def predict_dummy(seed, length):

    dummy_prediction = 'a'*length
    text = seed + dummy_prediction

    return text


@app.route("/")
def hello():
    return "Hola, I'm Bot Quijote de la Mancha."


@app.route("/predict/<string:seed>/<int:length>", methods=["GET"])
def predict(seed, length):
    predicted_text = predict_with_model(loaded, length, prime=seed)
    return jsonify(predicted_text)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
