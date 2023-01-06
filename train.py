import numpy as np
import pandas
from PyLMD import LMD
from pyemd import emd as EMD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import LSTM
import os
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import svm
from xgboost import XGBRegressor

np.random.seed(0)

# load data
data_file_path = osp.join('data', 'walmart-sales-dataset-of-45stores.csv')
df = pandas.read_csv(data_file_path)
df.info()
df.head()
mx = df['Weekly_Sales'].max()
mn = df['Weekly_Sales'].min()
df['Weekly_Sales'] = df['Weekly_Sales'].apply(lambda x: (x - mn) / (mx - mn))

# LMD decomposition
num_pf = 0
lmd = LMD()
df_stores = df.groupby('Store')
PFs_stores = []
res_stores = []
time_num = df_stores.get_group(1).shape[0]
store_num = df_stores.ngroups
for idx,df_store in df_stores:
    y = df_store['Weekly_Sales'].values
    PFs, res = lmd.lmd(y)
    PFs_stores.append(PFs)
    res_stores.append(res)
    num_pf = max(num_pf, len(PFs))

data_stores = []

for i in range(len(PFs_stores)):
    if len(PFs_stores[i]) < num_pf:
        PFs_stores[i] = np.pad(PFs_stores[i], ((0,num_pf-len(PFs_stores[i])),(0,0)), 'constant', constant_values=0)
    data_stores.append(df_stores.get_group(i+1))
    for j in range(num_pf):
        data_stores[-1].insert(len(data_stores[-1].columns), 'PF'+str(j), PFs_stores[i][j])
    data_stores[-1].insert(len(data_stores[-1].columns), 'Res', res_stores[i])
    
# class Data:
#     x = None
#     y = None
#     def __init__(self,x,y) -> None:
#         self.x = x.astype(np.float64)
#         self.y = y.astype(np.float64)
    
#     def __str__(self) -> str:
#         return 'x: ' + str(self.x) + ' y: ' + str(self.y)
    
class MyDataset(Dataset):
    data = None
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        x,y = self.data[index]
        return torch.Tensor(x), torch.Tensor(y)
    
train_ratio = 0.8
train_sample = np.random.choice(len(data_stores), int(len(data_stores)*train_ratio), replace=False)
test_sample = np.array(list(set(range(len(data_stores))) - set(train_sample)))

q = 4 # window size = q+1, use q previous data to predict the next one

print('window size: ', q+1)

def get_window(data, i, q):
    return data[i-q:i+1]

def get_data(sample):
    data = []
    for i in sample:
        store = data_stores[i]
        values = store.values
        for j in range(q, len(store)):
            window = get_window(values, j, q)
            # data.append((window[:-1,3:].astype(np.float32), window[-1,2:3].astype(np.float32)))
            data.append((window[:-1,2:8].astype(np.float32), window[-1,2:3].astype(np.float32)))
    return np.array(data,dtype=object)

train_data = get_data(train_sample)
test_data = get_data(test_sample)
test_one_data = get_data(test_sample[0:1])
# test_one_data = get_data(train_sample[0:1])

### SVM ###

def get_data_svm(data):
    x,y = data[:,0], data[:,1]
    x = np.concatenate(x).reshape(len(y),-1)
    y = np.concatenate(y)
    return x,y

x,y = get_data_svm(train_data)
regr = svm.SVR()
regr.fit(x, y)

x,y = get_data_svm(test_data)
y_pred = regr.predict(x)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
print('svm MAPE: ', mape,'%')

### XGBoost ###
x,y = get_data_svm(train_data)
regr = XGBRegressor()
regr.fit(x, y)
x,y = get_data_svm(test_data)
y_pred = regr.predict(x)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
print('xgboost MAPE: ', mape,'%')


train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
test_one_dataset = MyDataset(test_one_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

feature_num = train_data[0][0].shape[1]
label_num = train_data[0][1].shape[0]

print(feature_num, label_num)

class Model(nn.Module):
    def __init__(self,feature_num,label_num=1,H_size=16) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=feature_num,
            hidden_size=H_size,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(H_size, 1)
    
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

model = Model(feature_num)

def train(model,train_loader,test_loader,epoch_num):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in tqdm(range(epoch_num)):
        sum_loss = 0
        for step, (x, y) in enumerate(train_loader):
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            sum_loss += loss.item()
            optimizer.step()
        if epoch % 10 == 9:
            train_loss = sum_loss / len(train_loader)
            sum_loss = 0
            for step, (x, y) in enumerate(test_loader):
                output = model(x)
                loss = criterion(output, y)
                sum_loss += loss.item()
            tqdm.write('epoch: ' + str(epoch) + 
                       ' test loss: ' + str(sum_loss / len(test_loader)) + 
                       ' train loss: ' + str(train_loss))
            
def test(model,test_dataset,plot=True):
    real = np.array([y.item() for x,y in test_dataset])
    predict = []
    for x,y in test_dataset:
        output = model(x.unsqueeze(0))
        predict.append(output.item())
    predict = np.array(predict)
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot([i for i in range(len(test_dataset))],real , label='real')
        plt.plot([i for i in range(len(test_dataset))], predict, label='predict')
        plt.legend()
        plt.show()
    mape = np.mean(np.abs((real - predict) / real))
    print('MAPE: ' + str(mape*100) + '%')

train(model,train_loader,test_loader,100)

test(model,test_dataset,plot=False)

test(model,test_one_dataset)