# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:14
# @Author  : godot
# @FileName: model2.py
# @Project : pyOED
# @Software: PyCharm
import math

import numpy as np


class Model2():

    def __init__(self, parameters):
        self.model_name = "model2"
        self.m = 2
        self.k = 1
        self.parameters = parameters
        self.a = self.parameters[0]
        self.b = self.parameters[1]

    def par_a(self, x):
        return x

    def par_b(self, x):
        a = self.a
        b = self.b
        return -a * b * math.log(x) * x / (b * b)

    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        x = math.exp(x[0] / self.parameters[1])
        return np.array([[self.par_a(x),
                          self.par_b(x)]]).T
