# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:14
# @Author  : godot
# @FileName: model2.py
# @Project : pyOED
# @Software: PyCharm
import math

import numpy as np


class LogisticModel():

    def __init__(self, parameters):
        self.model_name = "logistic"
        self.m = 6  # number of parameters
        self.k = 5  # number of independent variables
        self.parameters = parameters
        self.t0 = self.parameters[0]
        self.t1 = self.parameters[1]
        self.t2 = self.parameters[2]
        self.t3 = self.parameters[3]
        self.t4 = self.parameters[4]
        self.t5 = self.parameters[5]

    def par_t0(self, x, const):
        return const

    def par_t1(self, x, const):
        return x[0] * const

    def par_t2(self, x, const):
        return x[1] * const

    def par_t3(self, x, const):
        return x[2] * const

    def par_t4(self, x, const):
        return x[3] * const

    def par_t5(self, x, const):
        return x[4] * const

    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        const = math.exp(self.t0 + self.t1 * x[0] + self.t2 * x[1] + self.t3 * x[2] + self.t4 * x[3] + self.t5 * x[4])
        const = math.sqrt(const / (1 + const) ** 2)

        # print(np.array([[1]+[i for i in x]]).T)
        # return np.array([[1]+[i for i in x]]).T
        #
        return np.array([[self.par_t0(x, const),
                          self.par_t1(x, const),
                          self.par_t2(x, const),
                          self.par_t3(x, const),
                          self.par_t4(x, const),
                          self.par_t5(x, const)]]).T
