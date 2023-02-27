# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:14
# @Author  : godot
# @FileName: model2.py
# @Project : pyOED
# @Software: PyCharm
import math

import numpy as np


class PotatoModel():

    def __init__(self, parameters):
        self.model_name = "Potato"
        self.m = 8  # number of parameters
        self.k = 8  # number of independent variables
        self.parameters = parameters
        self.t0 = self.parameters[0]
        self.t1 = self.parameters[1]
        self.t2 = self.parameters[2]
        self.t3 = self.parameters[3]
        self.t4 = self.parameters[4]
        self.t5 = self.parameters[5]
        self.t6 = self.parameters[6]
        self.t7 = self.parameters[7]

    # Y = 0.063*INSO + 0.087*NITRO + 1.604*PLANT - 0.509* EMERG+ 0.414*DENST+3.349*SFERTP-0.224*SFERTK-0.418*SFERTM
    def par_t0(self, x):
        return x[0]

    def par_t1(self, x):
        return x[1]

    def par_t2(self, x):
        return x[2]

    def par_t3(self, x):
        return x[3]

    def par_t4(self, x):
        return x[4]

    def par_t5(self, x):
        return x[5]

    def par_t6(self, x):
        return x[6]

    def par_t7(self, x):
        return x[7]

    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """

        # print(np.array([[1]+[i for i in x]]).T)
        # return np.array([[1]+[i for i in x]]).T
        #
        return np.array([[self.par_t0(x),
                          self.par_t1(x),
                          self.par_t2(x),
                          self.par_t3(x),
                          self.par_t4(x),
                          self.par_t5(x),
                          self.par_t6(x),
                          self.par_t7(x)]]).T
