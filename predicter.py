# -*- coding: utf-8 -*-

import numpy
from gensim.models import word2vec
from sklearn.cross_validation import train_test_split

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, cuda, serializers, training, iterators, report
from chainer.training import extensions
from chainer.datasets import tuple_dataset

from trans_vector import TransVecotr
from simple_cnn import SimpleCNN

GPU = -1
N_BATCH = 100

MODEL_PATH = "./data/model/jawiki.model"
WAKATI_PATH = "./data/text/train_w.csv"
TRAINER_PATH = "./model/snapshot_iter_50"
N_OUTPUT = 50
FILTER_H = 3
FILTER_W = 76
MID_UNITS = 1250
N_UNITS = 200
N_LABEL = 2
LAB_DIC = {"yakiu":0, "genju":1}

xp = cuda.cupy if GPU >= 0 else numpy


def main():
    w2v_dict = TransVecotr(MODEL_PATH)
    dataset, height, width = w2v_dict(WAKATI_PATH)

    feat_data = dataset["vec"]
    label_data = xp.array([LAB_DIC[i] for i in dataset["lab"]], dtype=xp.int32)

    input_channel = 1
    x_train = xp.array(feat_data, dtype=xp.float32).reshape(len(feat_data), input_channel, height, width) 
    train = tuple_dataset.TupleDataset(x_train, label_data)
    train_iter = iterators.SerialIterator(train, N_BATCH)

    model = L.Classifier(SimpleCNN(input_channel, N_OUTPUT, FILTER_H, width, MID_UNITS, N_UNITS, N_LABEL))
    if GPU >= 0:
        model.to_gpu()
    optimizer = optimizers.AdaGrad()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=GPU)
    trainer = training.Trainer(updater, (1, 'epoch'), out="result")
    serializers.load_npz(TRAINER_PATH, trainer)

    while True:
        input_text = raw_input('input text :')
        if input_text == "exit":
            break
        pred_vec = w2v_dict.gen_pred_vec(input_text, height)
        pred_vec = xp.array([pred_vec], dtype=xp.float32)
        pred_data = xp.array([pred_vec], dtype=xp.float32)
        hyp_data = model.predictor(pred_data)
        res_dict = {v:k for k, v in LAB_DIC.items()}
        if res_dict[hyp_data.data.argmax()] == "yakiu":
            print "彡(ﾟ)(ﾟ) やきう民"
        else:
            print "(´・ω・`) 原住民"
        print

if __name__ == '__main__':
    main()
