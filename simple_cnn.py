# -*- coding: utf-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class SimpleCNN(Chain):

    def __init__(self, n_input, n_output, filter_h, filter_w, mid_units, n_units, n_label):
        super(SimpleCNN, self).__init__(
            conv1 = L.Convolution2D(n_input, n_output, (filter_h, filter_w)),
            l1    = L.Linear(mid_units, n_units),
            l2    = L.Linear(n_units,  n_label),
        )

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h2)
        return y
