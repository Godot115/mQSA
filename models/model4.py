# -*- coding: utf-8 -*-
# @Time    : 10/6/22 04:22
# @Author  : godot
# @FileName: model4.py
# @Project : pyOED
# @Software: PyCharm


import math

import numpy as np


class Model4():
    def __init__(self, parameters):
        self.model_name = "model4"
        self.m = 3
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
        return a * x * math.log(x) * (c - 1) / b

    def par_c(self, x):
        a = self.a
        return a - a * x

    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        x = math.exp(-x[0] / self.b)
        return np.array([[self.par_a(x),
                          self.par_b(x),
                          self.par_c(x)]]).T
