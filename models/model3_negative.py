# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:35
# @Author  : godot
# @FileName: model3_positive.py
# @Project : pyOED
# @Software: PyCharm
from math import log, exp

import numpy as np


class Model3Negative():
    def __init__(self, parameters):
        self.model_name = "model3Negative"
        self.m = 3
        self.k = 1
        self.parameters = parameters
        self.a = self.parameters[0]
        self.b = self.parameters[1]
        self.c = self.parameters[2]
        self.d = self.parameters[3]

    def par_a(self, x):
        return x

    def par_b(self, x):
        a = self.a
        b = self.b
        d = self.d
        return (-a * d * log(x) * x) / b

    def par_d(self, x):
        a = self.a
        d = self.d
        return a * log(x) * x * log((-log(x)) ** (-d))

    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
         [∂η(x,Theta) / ∂θm]]
        """

        x = exp(-((x[0] / self.b) ** self.d))
        return np.array([[self.par_a(x),
                          self.par_b(x),
                          self.par_d(x)]]).T
