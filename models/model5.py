# -*- coding: utf-8 -*-
# @Time    : 10/6/22 04:26
# @Author  : godot
# @FileName: model5.py
# @Project : pyOED
# @Software: PyCharm


from math import exp
from math import log

import numpy as np


class Model5():
    def __init__(self, parameters):
        self.model_name = "model5"
        self.m = 4
        self.k = 1
        self.parameters = parameters
        self.a = self.parameters[0]
        self.b = self.parameters[1]
        self.c = self.parameters[2]
        self.d = self.parameters[3]

    def par_a(self, x):
        c = self.c

        return c - (c - 1) * x

    def par_b(self, x):
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return a * d * (-log(x)) * (1 - c) * x / b

    def par_c(self, x):
        a = self.a
        return a * (1 - x)

    def par_d(self, x):
        a = self.a
        c = self.c
        d = self.d
        return -a * (-log(x)) * (1 - c) * x * log((-log(x)) ** (1 / d))

    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        x = exp(-(x[0] / self.b) ** self.d)
        return np.array([[self.par_a(x),
                          self.par_b(x),
                          self.par_c(x),
                          self.par_d(x)]]).T
