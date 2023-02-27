# -*- coding: utf-8 -*-
# @Time    : 2/27/23 22:53
# @Author  : godot
# @FileName: csv_processing.py
# @Project : mQSA
# @Software: PyCharm

import pandas as pd

df = pd.read_csv("poisson_log.csv")
means = []
for theta in df.theta.unique():
    for algorithm in df.algorithm.unique():
        row = df[(df.theta == theta) & (df.algorithm == algorithm)].mean()
        row.drop(["seed"], inplace=True)
        row["algorithm"] = algorithm
        row["theta"] = theta
        means.append(row)

means = pd.DataFrame(means)
cols = means.columns.tolist()
cols = cols[-2:] + cols[:-2]
means = means[cols]
means.to_csv("poisson_means_theta_algorithm.csv", index=False)


