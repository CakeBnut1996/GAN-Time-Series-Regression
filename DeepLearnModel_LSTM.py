import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from itertools import chain
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import timedelta
import gc  # Garbage Collector
from sklearn.pipeline import Pipeline
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, MSELoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, \
    BatchNorm2d, Dropout
from torch.optim import Adam, SGD, Adagrad
import random


###### Reshape ###########
class TransformData():
    def __init__(self, n_dim=24, nts=28):
        self.n_dim = n_dim
        self.nts = nts

    def fit(self, x, y=None):
        return self

    def transform(self, X_train, y=None):
        X_train = X_train.reshape(-1, self.nts, self.n_dim)
        X_train = torch.from_numpy(X_train)
        gc.collect()
        return X_train

#### Rolling Horizon #######
def input_window(dt, prefix, drop=1):
    df = (dt.pivot_table(index=['TMC', 'dayID'], columns='hour2num', values=prefix)
          .reset_index()
          .rename_axis(None, axis=1))
    cols = df.columns[~df.columns.isin(['TMC', 'dayID'])]
    df.rename(columns=dict(zip(cols, cols.astype(str) + prefix)), inplace=True)

    if drop == 1:
        df = df.drop(columns=['TMC', 'dayID'])

    return df

def input_window_time(dt, prefix):
    df = (dt.pivot_table(index=['TMC', 'dayID'], columns='hour2num', values='TMC_n0' + prefix)
          .reset_index()
          .rename_axis(None, axis=1))
    cols = df.columns[~df.columns.isin(['TMC', 'dayID'])]
    df.rename(columns=dict(zip(cols, cols.astype(str) + prefix)), inplace=True)
    df = df.drop(columns=['TMC', 'dayID'])

    return df

#### LSTM Model ########
class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=20, output_size=48, num_layers=2, seq_len=144):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size  # size = dimension
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_layer_size * self.seq_len, output_size)

    def init_hidden(self, batch_size=1):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size)
        self.hidden = (hidden_state, cell_state)

    def forward(self, input_seq):
        batch_size, seq_length, _ = input_seq.size()
        out, self.hidden = self.lstm(input_seq, self.hidden)
        x = self.fc(out.contiguous().view(batch_size, -1))
        # print(x.shape)
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # return predictions[-1]
        return x


class LSTM_train():
    def __init__(self, batch_size=150, n_dim=[], nts=[]):
        self.batch_size = batch_size
        self.model = []
        self.n_dim = n_dim
        self.nts = nts

    def fit(self, X_train, y_train, n_epochs=35):
        # We train LSTM with 24 hidden units.
        # A lower number of units is used so that it is less likely that LSTM would perfectly memorize the sequence.
        model = LSTM(input_size=self.n_dim, hidden_layer_size=24, output_size=self.nts, num_layers=1, seq_len=self.nts)
        # defining the optimizer
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.01)  #
        # defining the loss function
        criterion = MSELoss()

        model.train()
        coef = []
        for epoch in range(n_epochs):
            for b in range(0, (len(X_train) - self.batch_size), self.batch_size):
                inpt = X_train[b:b + self.batch_size, :, :];
                target = y_train[b:b + self.batch_size]
                x_batch = torch.tensor(inpt, dtype=torch.float32);
                y_batch = torch.tensor(target, dtype=torch.float32)

                del inpt, target
                gc.collect()

                # clearing the Gradients of the model parameters
                optimizer.zero_grad()
                # print(x_batch.size(0))
                model.init_hidden(x_batch.size(0))

                # prediction for training and validation set
                output_train = model(x_batch)
                output_train = output_train.reshape(self.batch_size, self.nts, 1)
                loss_train = criterion(output_train, y_batch)  # + l1_regularization + l2_regularization

                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()

                del x_batch, y_batch
                gc.collect()
            # w = list(model.parameters());
            # coef.append(w)
            if epoch % 2 == 0:
                # printing the validation loss
                print('Epoch ', epoch + 1, '\t', 'train loss: ', loss_train)

            self.model = model

        return self.model

    def predict(self, X_val):  # , y_val
        pred = torch.Tensor()
        for b in range(0, (len(X_val)), self.batch_size):
            inpt = X_val[b:b + self.batch_size, :, :];
            # target = y_val[b:b + self.batch_size]
            x_batch = torch.tensor(inpt, dtype=torch.float32);
            # y_batch = torch.tensor(target, dtype=torch.float32)
            self.model.init_hidden(x_batch.size(0))
            pred = torch.cat([pred, self.model(x_batch)], dim=0)
            print('Batch ', b, '\t', 'Finished')
        pred = pred.data.numpy()
        pred = pred.reshape(-1, self.nts);
        return pred


##################### MAIN #########################
### Step 1: Read file
dt = pd.read_csv('data/speed_input.csv')

### Step 2: Preprocessing
print('Number of links: ' + str(len(np.unique(dt.TMC))))
filter_out = ['TMC_n0_l15_ld0', 'TMC_n0_l30_ld0', 'TMC_n0_l45_ld0', 'TMC_n0_l60_ld0',
              'TMC_n0_l75_ld0', 'TMC_n0_l90_ld0', 'TMC_n0_l105_ld0']
if set(filter_out).issubset(dt.columns):
    dt = dt.drop(columns=filter_out);
    gc.collect()
filter_out = ['TMC_n1', 'TMC_n2', 'TMC_n3', 'TMC_n4']
if set(filter_out).issubset(dt.columns):
    dt = dt[dt.columns[~dt.columns.str.startswith(tuple(filter_out))]]

dt['hour2num'] = dt['minute'] / 15 # used to separate every two hours
dt.loc[dt['hour'] % 2 == 0, 'hour2num'] = dt.loc[dt['hour'] % 2 == 0, 'hour2num'] + 4
dt.info()

#### Step 3: Prepare inputs
print('Preparing time-related independent variables')
nts = 8
temp_features = ['month', 'dayofweek', 'timeID']
print('Rolling horizon for last week speeds...')
df = input_window(dt, 'TMC_n0_ld5', drop=0)
tmp = input_window_time(dt, '_l45_ld5');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l30_ld5');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l15_ld5');
df = pd.concat([df, tmp], axis=1)

n_feature = nts * 4

print('Rolling horizon for historical speeds related variables...')
tmp = input_window_time(dt, '_l' + str(8 * 15) + '_ld0');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l' + str(9 * 15) + '_ld0');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l' + str(10 * 15) + '_ld0');
df = pd.concat([df, tmp], axis=1)
n_feature = n_feature + nts * 3

### Step 4: Prepare outputs
tmp = input_window(dt, 'congestion');
df = pd.concat([df, tmp], axis=1)

### Step 5: Filtering out NAs
df = df.sort_values(by=['TMC', 'dayID'])
print('Original length: ' + str(len(df)))
df = df.dropna().reset_index(drop=True)
print('Length after dropping NAN: ' + str(len(df)))

df_all = df.copy();
uniq_days = list(np.unique(df_all.dayID));
del tmp, dt, df;
gc.collect()

### Step 6: Prepare training X_train and testing set X_val
ind_train = df_all[df_all['dayID'].isin(random.sample(uniq_days, int(0.8 * len(uniq_days))))].index.tolist()
df = df_all.drop(columns=['TMC', 'dayID']);

print(df.head())
df.info()

X_train = df.iloc[ind_train, 0:n_feature];
# print('Training columns: '+str(X_train.columns))
y_train = df.iloc[ind_train, n_feature:]
tmp = df.drop(ind_train, axis=0)
X_val = tmp.iloc[:, 0:n_feature];
y_val = tmp.iloc[:, n_feature:];
del tmp

### Step 7: Reshape the data to # of batches, # of time steps, # dimensions
n_dim = int(n_feature / nts)
X_train = np.array(X_train)
X_train = X_train.reshape(-1, nts, n_dim, order='F')
X_train = X_train.reshape(-1, n_dim)
X_val = np.array(X_val)
X_val = X_val.reshape(-1, nts, n_dim, order='F')
X_val = X_val.reshape(-1, n_dim)

y_train = np.array(y_train)
y_train = y_train.reshape(-1, nts, 1)
y_train = torch.from_numpy(y_train)

print("Dimension of the training set: " + str((X_train.shape[1])))
del df, df_all
gc.collect()

### Step 8: Training
bs = 150  # 450
steps = [('scaler', StandardScaler()),
         ('transform', TransformData(n_dim=n_dim, nts=nts)),
         ('lstm_train', LSTM_train(batch_size=bs, n_dim=n_dim, nts=nts))]
pipeline = Pipeline(steps)  # define the pipeline object.
model_num = pipeline.fit(X_train, y_train)
pickle.dump(model_num, open("LSTM" + ".dat", "wb"))

### Step 9: Testing
loaded_model = pickle.load(open("LSTM" + ".dat", "rb"))
pred = loaded_model.predict(np.array(X_val))
actual = np.array(y_val)
actual = actual.reshape(-1, nts);

########### Evaluation ##########
print('LSTM testing error: ' + str(np.mean(np.abs(pred - actual) / actual)))

