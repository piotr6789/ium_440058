from numpy.lib.function_base import average
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
import sys

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear(x)
        return self.sigmoid(out)

results = pd.read_csv('diabetes2.csv')

results.dropna()

data_train, data_valid, data_test = np.split(results.sample(frac=1), [int(.6*len(results)), int(.8*len(results))])
columns_to_train   =  ['Glucose', 'BloodPressure', 'Insulin', 'Age']

x_train = data_train[columns_to_train].astype(np.float32)
y_train = data_train['Outcome'].astype(np.float32)

x_test = data_test[columns_to_train].astype(np.float32)
y_test = data_test['Outcome'].astype(np.float32)

fTrain = torch.from_numpy(x_train.values)
tTrain = torch.from_numpy(y_train.values.reshape(460,1))

fTest= torch.from_numpy(x_test.values)
tTest = torch.from_numpy(y_test.values)

input_dim = 4
output_dim = 1

model = LogisticRegressionModel(input_dim, output_dim)

pred = model(fTest)
accuracy = accuracy_score(tTest, np.argmax(pred.detach().numpy(), axis = 1))


f1 = f1_score(tTest, np.argmax(pred.detach().numpy(), axis = 1), average = None)
rmse = mean_squared_error(tTest, pred.detach().numpy())

print(f'Accuracy: {accuracy}')
print(f'F1: {f1_score}')
print(f'RMSE: {rmse}')

with open("results.txt", 'w') as outfile:
    outfile.write("Accuracy: " + str(accuracy) + "\n")
    outfile.write("F1: " + str(f1_score) + "\n")
    outfile.write("RMSE: " + str(rmse) + "\n")