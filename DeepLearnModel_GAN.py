import torch
from torch.optim import Adam
import random
import numpy as np
import pandas as pd
import gc  # Garbage Collector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

######## Reshape #########
class TransformData():
    def __init__(self, n_dim=24, nts=28):
        self.n_dim = n_dim
        self.nts = nts

    def fit(self, x, y=None):
        return self

    def transform(self, X_train, y=None):
        X_train = X_train.reshape(-1, 8, self.n_dim)
        X_train = torch.from_numpy(X_train)
        return X_train

######## GAN Model & Training ######
class LSTMGenerator(torch.nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_dim, out_dim), torch.nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs

class LSTMDiscriminator(torch.nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element.
    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs

class GAN_train():
    def __init__(self, n_dim=[], nts=[], bs=[]):
        self.in_dim = n_dim
        self.model = []
        self.nz = n_dim
        self.seq_len = nts
        self.batch_size = bs

    def fit(self, X_train, y_train, n_epochs=25):
        netD = LSTMDiscriminator(in_dim=self.in_dim + 1, out_dim=1, hidden_dim=2)
        netG = LSTMGenerator(in_dim=self.in_dim, out_dim=self.in_dim + 1, hidden_dim=2)
        # print("|Discriminator Architecture|\n", netD)
        # print("|Generator Architecture|\n", netG)
        real_label = 0.9
        fake_label = 0.1

        # setup optimizer
        optimizerD = Adam(netD.parameters(), lr=0.01, betas=[0.5, 0.999])
        optimizerG = Adam(netG.parameters(), lr=0.01, betas=[0.5, 0.999])
        criterion = torch.nn.BCELoss()

        # Want errorD first decrease then increase; and errorG first increase then decrease
        for epoch in range(n_epochs):
            i = 0
            for b in range(0, (len(X_train) - self.batch_size), self.batch_size):
                inpt = X_train[b:b + self.batch_size, :, :];
                target = y_train[b:b + self.batch_size, :, :]
                x_batch = torch.tensor(inpt, dtype=torch.float32);
                y_batch = torch.tensor(target, dtype=torch.float32)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # Train with real data
                netD.zero_grad()
                real = y_batch
                # batch_size, seq_len = real.size(0), real.size(1)
                label = torch.full((self.batch_size, self.seq_len, 1), real_label, dtype=torch.float)

                output = netD(real)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Train with fake data
                noise = x_batch + torch.randn(self.batch_size, self.seq_len, self.nz) / (i + 1)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake  # By summing up these two discriminator losses we obtain the total mini-batch loss for the Discriminator. In practice, we will calculate the gradients separately, and then update them together.
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake)
                errG = criterion(output,
                                 label)  # PyTorch and most other Machine Learning frameworks usually minimize functions instead
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                i = i + 1

                ###########################
                # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
                ###########################
                # Report metrics
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f '
                  % (epoch, n_epochs,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        self.model = netG

        return self.model

    def predict(self, X_val):  # , y_val
        pred = torch.Tensor()
        for b in range(0, (len(X_val)), self.batch_size):
            inpt = X_val[b:b + self.batch_size, :, :];
            # target = y_val[b:b + self.batch_size]
            x_batch = torch.tensor(inpt, dtype=torch.float32);
            # y_batch = torch.tensor(target, dtype=torch.float32)
            pred = torch.cat([pred, self.model(x_batch)], dim=0)
            print('Batch ', b, '\t', 'Finished')

        pred = pred.data.numpy()
        pred = pred.reshape(-1, self.in_dim + 1);
        return pred

######## Rolling horizons #######
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

##################### MAIN #########################
### Step 1: Read Input
dt = pd.read_csv('data/speed_input.csv')

### Step 2: Preprocessing
print('Number of links: '+str(len(np.unique(dt.TMC))))
filter_out = ['TMC_n0_l15_ld0', 'TMC_n0_l30_ld0', 'TMC_n0_l45_ld0', 'TMC_n0_l60_ld0',
              'TMC_n0_l75_ld0', 'TMC_n0_l90_ld0', 'TMC_n0_l105_ld0']
if set(filter_out).issubset(dt.columns):
    dt = dt.drop(columns = filter_out); gc.collect()
filter_out = ['TMC_n1','TMC_n2','TMC_n3','TMC_n4']
if set(filter_out).issubset(dt.columns):
    dt = dt[dt.columns[~dt.columns.str.startswith(tuple(filter_out))]]
print(dt.columns)
dt['hour2num'] = dt['minute'] / 15 # used to separate every two hours
dt.loc[dt['hour'] % 2 == 0, 'hour2num'] = dt.loc[dt['hour'] % 2 == 0, 'hour2num'] + 4

#### Step 3: Prepare inputs
nts = 8 # number of time steps: The output contain 8 15-min intervals

print('Rolling horizon for historical speeds...')
df = input_window(dt, 'TMC_n0_ld5', drop=0)
tmp = input_window_time(dt, '_l45_ld5');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l30_ld5');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l15_ld5');
df = pd.concat([df, tmp], axis=1)

tmp = input_window_time(dt, '_ld1');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l45_ld1');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l30_ld1');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l15_ld1');
df = pd.concat([df, tmp], axis=1)

tmp = input_window_time(dt, '_l' + str(8 * 15) + '_ld0');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l' + str(9 * 15) + '_ld0');
df = pd.concat([df, tmp], axis=1)
tmp = input_window_time(dt, '_l' + str(10 * 15) + '_ld0');
df = pd.concat([df, tmp], axis=1)

n_feature = nts * 11  # 11 historical speed related variables

print('Rolling horizon for time-related variables...')
temp_features = ['month', 'dayofweek', 'timeID']
for cols in temp_features:
    tmp = input_window(dt, cols);
    df = pd.concat([df, tmp], axis=1)
    n_feature = n_feature + nts

### Step 4: Prepare outputs
tmp = input_window(dt, 'congestion');
df = pd.concat([df, tmp], axis=1)


### Step 5: Filtering out NAs
df = df.sort_values(by=['dayID'])
print('Original length: ' + str(len(df)))
df = df.dropna().reset_index(drop=True)
print('Length after dropping NAN: ' + str(len(df)))
df_all = df.copy();
uniq_days = list(np.unique(df_all.dayID));
del df;
gc.collect()

### Step 6: Prepare training X_train and testing set X_val
ind_train = df_all[df_all['dayID'].isin(random.sample(uniq_days, int(0.8 * len(uniq_days))))].index.tolist()
df = df_all.drop(columns=['TMC', 'dayID']);

X_train = df.iloc[ind_train, 0:n_feature];
# print('Training columns: '+str(X_train.columns))
y_train = df.iloc[ind_train, :]  # n_feature:
tmp = df.drop(ind_train, axis=0)
X_val = tmp.iloc[:, 0:n_feature]
y_val = tmp.iloc[:, :]  # n_feature:
del tmp

### Step 7: Reshape the data to # of batches, # of time steps, # dimensions
n_dim = int(n_feature / nts)  # Number of columns in each batch
X_train = np.array(X_train)
X_train = X_train.reshape(-1, nts, n_dim, order='F')
X_train = X_train.reshape(-1, n_dim)
X_val = np.array(X_val)
X_val = X_val.reshape(-1, nts, n_dim, order='F')
X_val = X_val.reshape(-1, n_dim)

y_train = np.array(y_train)
y_train = y_train.reshape(-1, nts, n_dim+1, order='F')
y_train = torch.from_numpy(y_train)
y_val = np.array(y_val)
y_val = y_val.reshape(-1, nts, n_dim+1, order='F')

### Step 8: Training
steps = [('scaler', StandardScaler()),
         ('transform', TransformData(n_dim=n_dim, nts=nts)),
         ('gan_train', GAN_train(n_dim=n_dim, nts=nts, bs=50))]
pipeline = Pipeline(steps)  # define the pipeline object.
model_num = pipeline.fit(X_train, y_train)
pickle.dump(model_num, open("models/GAN" + ".dat", "wb"))

### Step 9: Testing
loaded_model = pickle.load(open("models/GAN" + ".dat", "rb"))
pred = loaded_model.predict(np.array(X_val))
pred_y = pred[:, pred.shape[1]-1]
actual = y_val.reshape(-1, n_dim + 1)
actual = actual[:, actual.shape[1]-1]

print('Testing error (MAPE): ' + str(np.mean(np.abs(pred_y - actual) / actual)))











