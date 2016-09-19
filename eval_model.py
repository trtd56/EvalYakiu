# -*- coding: utf-8 -*-

import numpy
import six
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
N_EPOCH = 50
N_BATCH = 100

MODEL_PATH = "./data/model/jawiki.model"
WAKATI_PATH = "./data/text/train_w.csv"
N_OUTPUT = 50
FILTER_H = 3
FILTER_W = 76
MID_UNITS = 1250
N_UNITS = 200
N_LABEL = 2
SNAPSHOT_INTERVAL = 10
LAB_DIC = {"yakiu":0, "genju":1}


xp = cuda.cupy if GPU >= 0 else numpy


def main():
    w2v_dict = TransVecotr(MODEL_PATH)
    dataset, height, width = w2v_dict(WAKATI_PATH)

    feat_data = dataset["vec"]
    label_data = xp.array([LAB_DIC[i] for i in dataset["lab"]], dtype=xp.int32)
    x_train, x_test, y_train, y_test = train_test_split(feat_data, label_data, test_size=0.15)

    input_channel = 1
    x_train = xp.array(x_train, dtype=xp.float32).reshape(len(x_train), input_channel, height, width) 
    x_test  = xp.array(x_test, dtype=xp.float32).reshape(len(x_test), input_channel, height, width)

    train = tuple_dataset.TupleDataset(x_train, y_train)
    test = tuple_dataset.TupleDataset(x_test, y_test)

    train_iter = iterators.SerialIterator(train, N_BATCH)
    test_iter = iterators.SerialIterator(test, N_BATCH, repeat=False, shuffle=False)

    model = L.Classifier(SimpleCNN(input_channel, N_OUTPUT, FILTER_H, width, MID_UNITS, N_UNITS, N_LABEL))
    if GPU >= 0:
        model.to_gpu()

    optimizer = optimizers.AdaGrad()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=GPU)
    trainer = training.Trainer(updater, (N_EPOCH, 'epoch'), out="result")
    trainer.extend(extensions.Evaluator(test_iter, model, device=GPU))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
