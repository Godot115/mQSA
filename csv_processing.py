# -*- coding: utf-8 -*-
# @Time    : 2/27/23 22:53
# @Author  : godot
# @FileName: csv_processing.py
# @Project : mQSA
# @Software: PyCharm

import pandas as pd

df = pd.read_csv("logistic_log.csv", index_col=False)
means = []
for theta in df.theta.unique():
    for algorithm in df.algorithm.unique():
        rows = df[(df.theta == theta) & (df.algorithm == algorithm)]
        # drop 'theta', 'seed', 'best_fitness' indice in rows.
        # rows.drop(['algorithm', 'theta', 'seed', 'best_fitness'], axis=1, inplace=True)
        row = rows.mean()
        row.drop(['seed'], inplace=True)
        print(row)
        row['theta'] = theta
        row['algorithm'] = algorithm
        means.append(row)

means = pd.DataFrame(means)
cols = means.columns.tolist()
cols = cols[-2:] + cols[:-2]
means = means[cols]
means.to_csv("logistic_means_theta_algorithm.csv", index=False)
