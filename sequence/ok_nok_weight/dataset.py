# -*- coding:utf-8 _*-
"""
@version: 
author: just_silent
@time: 2022/02/25
@file: dataset.py
@function: 
@modify: 
"""

import openpyxl
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader():
    def __init__(self, path):
        self.train_input = []
        self.train_target = []
        self.test_input = []
        self.test_target = []
        self.batch_size = 8
        self.xlsx2tensor(path)

    def xlsx2tensor(self, path):
        data = pd.read_csv(path)
        df = data[['Type', 'Device Name', 'Fault number', 'Stud-ID:', 'Date / Time',
                           'Application', 'Tool type', 'System weld  counter', 'Tool weld  counter',
                           'Outlet weld counter WOP',
                           'Outlet weld counter', 'Optimization', 'Clean time', 'Mode',
                           'Weld mode', 'Pilot Weldcurrent Arc Voltage  Actual (Up)',
                           'Main Weldcurrent Voltage  Actual (Us)',
                           'Weldtime  Actual (It)', 'Weld Energy  Actual (Es)', 'Lift height Actual',
                           'Penetration  Actual (P)',
                           'Weldcurrent Actual (Is)', 'DroptimeActual(ms)', 'StickoutActual']]
        df.drop(columns=['Fault number', 'Stud-ID:', 'Date / Time', 'Application', 'Tool type',
                         'System weld  counter', 'Tool weld  counter', 'Outlet weld counter WOP', 'Outlet weld counter',
                         'Optimization', 'Mode', 'Weld mode', 'Device Name'], inplace=True)
        X = df.iloc[:, 1:12]
        y = df.iloc[:, 0]
        enc = LabelEncoder()
        X.iloc[:, 0] = enc.fit_transform(X.iloc[:, 0])
        X.iloc[:, 5] = enc.fit_transform(X.iloc[:, 5])
        X.iloc[:, 6] = enc.fit_transform(X.iloc[:, 6])
        y = enc.fit_transform(y)
        y = enc.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        for i in range(X_train.shape[0]):
            if self.batch_size*(i+1)< X_train.shape[0]:
                input = X_train.iloc[self.batch_size*i:self.batch_size*(i+1),:].values
                target = y_train[self.batch_size*i:self.batch_size*(i+1)]
                self.train_input.append(torch.tensor(input))
                self.train_target.append(torch.tensor(target))
        for i in range(X_test.shape[0]):
            if self.batch_size*(i+1)< X_test.shape[0]:
                input = X_test.iloc[self.batch_size*i:self.batch_size*(i+1),:].values
                target = y_test[self.batch_size*i:self.batch_size*(i+1)]
                self.test_input.append(torch.tensor(input))
                self.test_target.append(torch.tensor(target))
        return self