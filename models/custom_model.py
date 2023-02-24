# -*- coding: utf-8 -*-
# @Time    : 10/27/22 03:10
# @Author  : godot
# @FileName: custom_model.py
# @Project : pyOED
# @Software: PyCharm


import math
import time

import numpy as np
import sympy
import sympy
from sympy import *
from math import e, log

from models.model2 import Model2


class CustomModel():

    def __init__(self, expr, factor_notations, parameter_notations, parameters):
        self.model_name = "customModel"
        self.m = len(parameter_notations)
        self.k = len(factor_notations)
        self.parameters = parameters
        self.factor_variables = []
        self.parameter_variables = []
        for parameter_notation in parameter_notations:
            exec(f"{parameter_notation} = Symbol('{parameter_notation}')")
            self.parameter_variables.append(eval(parameter_notation))
        for factor_notation in factor_notations:
            exec(f"{factor_notation} = Symbol('{factor_notation}')")
            self.factor_variables.append(eval(factor_notation))
        exec(f"expr = {expr}")
        self.diff_exprs = []
        for parameter_variable in self.parameter_variables:
            diff_expr = diff(expr, parameter_variable)
            for idx in range(len(self.parameter_variables)):
                diff_expr = diff_expr.subs(self.parameter_variables[idx], self.parameters[idx])
            for idx in range(len(factor_notations)):
                diff_expr = diff_expr.subs(self.factor_variables[idx], f"factor{idx}")

            diff_str = diff_expr.__str__()
            for idx in range(len(factor_notations)):
                diff_str = diff_str.replace(f"factor{idx}", f"x[{idx}]")
            self.diff_exprs.append(diff_str)



    def par_deriv_vec(self, x):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        array = []
        for diff_expr in self.diff_exprs:
            array.append(eval(str(diff_expr)))
        return np.array([array]).T

