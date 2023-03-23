# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:14
# @Author  : godot
# @FileName: model2.py
# @Project : pyOED
# @Software: PyCharm
import math

import numpy as np


class AbsorptionModel():

    def __init__(self, parameters):
        self.model_name = "Absorption"
        self.m = 8  # number of parameters
        self.k = 3  # number of independent variables
        self.parameters = parameters
        self.t0 = self.parameters[0]
        self.t1 = self.parameters[1]
        self.t2 = self.parameters[2]
        self.t3 = self.parameters[3]
        self.t4 = self.parameters[4]
        self.t5 = self.parameters[5]
        self.t6 = self.parameters[6]
        self.t7 = self.parameters[7]

    # D[a_1 *t + a_2*C+a_3*M+a_4*Power[t,2]+a_5*Power[C,2]+a_6*Power[M,2]+a_7*t*C+a_8*Power[t,3],a_8]
    # Y = 0.063*INSO + 0.087*NITRO + 1.604*PLANT - 0.509* EMERG+ 0.414*DENST+3.349*SFERTP-0.224*SFERTK-0.418*SFERTM
    def par_t0(self, x):
        return x[0]

    def par_t1(self, x):
        return x[1]

    def par_t2(self, x):
        return x[2]

    def par_t3(self, x):
        return x[0] * x[0]

    def par_t4(self, x):
        return x[1] * x[1]

    def par_t5(self, x):
        return x[2] * x[2]

    def par_t6(self, x):
        return x[0] * x[1]

    def par_t7(self, x):
        return x[0] * x[0] * x[0]

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

        # self.t = x[0]
        # self.C = x[1]
        # self.M = x[2]
        return np.array([[self.par_t0(x),
                          self.par_t1(x),
                          self.par_t2(x),
                          self.par_t3(x),
                          self.par_t4(x),
                          self.par_t5(x),
                          self.par_t6(x),
                          self.par_t7(x)]]).T
