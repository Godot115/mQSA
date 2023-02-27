# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:02
# @Author  : godot
# @FileName: model_util.py
# @Project : pyOED
# @Software: PyCharm
import itertools
import math
import time

import numpy as np


class ModelUtil():

    def __init__(self, model, criterion, lob, hib, grid_size):
        self.model = model
        self.m = model.m
        self.inf_mat = np.zeros((self.m, self.m))
        self.lb = lob
        self.hb = hib
        self.levels = []
        self.grid_size = grid_size
        self.generate_candidate_set()
        self.criterion = criterion
        if criterion == "D-optimal":
            self.gradient_func = self.cal_variance
            self.cal_ve_alpha_s = self.cal_ve_alpha_s_d
            self.gradient_target = self.get_m
            self.get_criterion_val = self.get_det_inv_inf_mat
        elif criterion == 'A-optimal':
            self.gradient_func = self.cal_a
            self.cal_ve_alpha_s = self.cal_ve_alpha_s_a
            self.gradient_target = self.get_tr_inv_fim
            self.get_criterion_val = self.get_tr_inv_fim

    def generate_candidate_set(self):
        lb = self.lb
        hb = self.hb
        grid_size = self.grid_size
        for idx in range(len(lb)):
            self.levels.append(np.linspace(lb[idx], hb[idx], grid_size))
        self.candidate_set = [point for point in itertools.product(*self.levels)]

    def generate_star_set(self, x):
        levels = self.levels
        result = set()
        for factor in range(len(x)):
            line = levels[factor]
            for i in range(len(line)):
                result.add(tuple(x[:factor] + (line[i],) + x[factor + 1:]))
        # print(result)
        result = list(result)
        if tuple(x) in result:
            result.remove(tuple(x))
        return result

    def get_inf_mat(self):
        return self.inf_mat

    def get_det_inv_inf_mat(self):
        return np.linalg.det(np.linalg.inv(self.inf_mat))

    def get_det_fim(self):
        return np.linalg.det(self.inf_mat)

    def cal_inf_mat(self, design_points):
        """
        :param designPoints: design points
        :return: information matrix
        """
        self.inf_mat = np.zeros((self.m, self.m))
        for i in range(len(design_points)):
            self.inf_mat += self.model.par_deriv_vec(design_points[i][0]) * \
                            self.model.par_deriv_vec(design_points[i][0]).T * \
                            design_points[i][1]
        return np.array(self.inf_mat)

    def cal_inf_mat_w(self, design_points, weights):
        """
        :param designPoints: design points
        :return: information matrix
        """
        self.inf_mat = np.zeros((self.m, self.m))
        for i in range(len(design_points)):
            self.inf_mat += self.model.par_deriv_vec(design_points[i]) * \
                            self.model.par_deriv_vec(design_points[i]).T * \
                            weights[design_points[i]]
        return np.array(self.inf_mat)

    def cal_variance(self, x):
        inv_inf_mat = np.linalg.inv(self.inf_mat)
        left = np.matmul(self.model.par_deriv_vec(x).T, inv_inf_mat)
        result = np.matmul(left, self.model.par_deriv_vec(x))
        return result[0][0]

    def get_m(self):
        return self.m

    def cal_a(self, x):
        inv_inf_mat = np.linalg.inv(self.inf_mat)
        left = np.matmul(self.model.par_deriv_vec(x).T, np.matmul(inv_inf_mat, inv_inf_mat))
        result = np.matmul(left, self.model.par_deriv_vec(x))
        return result[0][0]

    def cal_ve_alpha_s_d(self, x_add, x_del, w_add, w_del):
        var_add = self.cal_variance(x_add)
        var_del = self.cal_variance(x_del)
        combined_var = self.cal_combined_variance(x_add, x_del)
        res = min(max((var_del - var_add) / (2 * (combined_var ** 2 - var_add * var_del)), -w_add), w_del)
        if np.isnan(res):
            print("nan")
            print(x_add, x_del, w_add, w_del)
            print()
        return res

    def cal_ve_alpha_s_a(self, x_add, x_del, w_add, w_del):
        threshold = 1e-14
        a_add = self.cal_a(x_add)
        a_del = self.cal_a(x_del)
        combined_a = self.cal_combined_a(x_add, x_del)
        d_add = self.cal_variance(x_add)
        d_del = self.cal_variance(x_del)
        combined_d = self.cal_combined_variance(x_add, x_del)
        A = a_add - a_del
        B = 2 * combined_a * combined_d - d_add * a_del - d_del * a_add
        C = d_add - d_del
        D = d_add * d_del - combined_d ** 2
        G = A * D + B * C
        if abs(G) <= threshold and abs(B) > threshold:
            r = -A / (2 * B)
            if r >= -w_add and r <= w_del:
                return r
        if abs(G) > 0:
            r = -(B + math.sqrt(B ** 2 - A * G)) / G
            if r >= -w_add and r <= w_del:
                return r
        if A > threshold:
            return w_del
        if A < -threshold:
            return -w_add
        return 0

    def get_tr_inv_fim(self):
        inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.trace(inv_inf_mat)

    def cal_combined_a(self, x_i, x_j):
        inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.matmul(np.matmul(self.model.par_deriv_vec(x_i).T, np.matmul(inv_inf_mat, inv_inf_mat)),
                         self.model.par_deriv_vec(x_j))[0][0]

    def cal_combined_variance(self, x_i, x_j):
        inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.matmul(np.matmul(self.model.par_deriv_vec(x_i).T, inv_inf_mat),
                         self.model.par_deriv_vec(x_j))[0][0]

    def cal_delta(self, x_i, x_j):
        return self.cal_variance(x_j) - \
               (self.cal_variance(x_i) * self.cal_variance(x_j) -
                self.cal_combined_variance(x_i, x_j) *
                self.cal_combined_variance(x_i, x_j)) - \
               self.cal_variance(x_i)

    def cal_observ_mass(self, x):
        return self.model.par_deriv_vec(x) * \
               self.model.par_deriv_vec(x).T

    def fim_add_mass(self, mass, step):
        self.inf_mat = (1 - step) * self.inf_mat + step * mass

    def obj_func(self, x):
        dimension = 1
        num_of_sup = 3
        points_len = num_of_sup * dimension
        points = []
        point = tuple()
        for p in x[:points_len]:
            point += (p,)
            if len(point) == dimension:
                points.append(point)
                point = tuple()
        weights = x[points_len:]
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        weights = [float(w) for w in weights]
        self.inf_mat = np.zeros((self.m, self.m))
        for i in range(len(points)):
            self.inf_mat += self.model.par_deriv_vec(points[i]) * \
                            self.model.par_deriv_vec(points[i]).T * weights[i]
        return np.linalg.det(self.inf_mat) * (1 - abs(1 - weights_sum))
