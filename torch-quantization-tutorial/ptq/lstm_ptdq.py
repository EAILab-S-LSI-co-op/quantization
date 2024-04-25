#This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch 
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Import the library
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import time
from tqdm import tqdm 
import random 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv('/home/bosung/workspace/qtrnn/time_series_dataset/testset.csv')
data2 = df.copy()
# Any results you write to the current directory are saved as output.


# 랜덤성 제어
SEED = 7993

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


df= df.rename(index=str, columns={' _tempm': 'temperature'})
df= df[62304:]
data2 = df.copy()

data2 = data2[['datetime_utc','temperature' ]]
data2['year'] = data2.datetime_utc.str.slice(stop =4)
data2.head()

data2 = data2.rename(index=str, columns={' _tempm': 'temperature'})
data2['datetime_utc'] = pd.to_datetime(data2['datetime_utc'])


data2 = data2[data2.temperature < 50]
data2.set_index('datetime_utc', inplace= True)
data2.dropna(inplace=True)
data2.head()

data2.drop('year', axis =1, inplace = True)
data_array = data2.values
data_array = data_array.astype('float32')

scaler= MinMaxScaler(feature_range=(0,1))
sc = scaler.fit_transform(data2)

timestep = 48                                                                                                                                                                                                                                                                                                    
X= []
Y=[]


for i in range(len(sc)- (timestep)):
    X.append(sc[i:i+timestep])
    Y.append(sc[i+timestep])


X=np.asanyarray(X)
Y=np.asanyarray(Y)


k = 35000
Xtrain = X[:k,:,:]
Xtest = X[k:,:,:]    
Ytrain = Y[:k]    
Ytest= Y[k:]
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0],  Xtrain.shape[1],1))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1],1))





class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.d_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.d_device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.d_device)
        
        out, _ = self.lstm(x, (h0, c0))
        last_state = out[:, -1, :]
        out = self.fc(last_state)
        out = self.activation(out)
        
        return out

# Define the hyperparameters
input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1

# Create an instance of the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Print the model architecture
print(model)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    # Convert the input data to torch tensors and move to GPU
    inputs = torch.from_numpy(Xtrain).float().to(device)
    targets = torch.from_numpy(Ytrain).float().to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_postfix_str("Loss: {:.4f}".format(loss.item()))


def print_model_size(mdl):
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

# now, predict.
model.eval()
Xtest_tensor = torch.from_numpy(Xtest).float().to(device)
Ytest_pred = model(Xtest_tensor).cpu().detach().numpy()
print(Ytest_pred)
def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    return os.path.getsize("temp.p")/1e6
print("size of model before quantization: ", get_size_of_model(model))
Ytest_pred = scaler.inverse_transform(Ytest_pred)
Ytest = scaler.inverse_transform(Ytest)
# mape
mape = np.mean(np.abs((Ytest - Ytest_pred) / Ytest)) * 100

plt.figure(figsize=(14, 7))
plt.plot(Ytest, label='True temperature', color='blue')
plt.plot(Ytest_pred, label='Predicted temperature', color='red')
plt.title('Temperature prediction')
plt.legend()
plt.savefig("./res_img/lstm_before_qt.png")
print("MAPE: ",mape )


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

# test_model = LSTMModel(10, 10, 10, 10) # 이건 예시인데, 잘 되더라.

# model.qconfig = torch.ao.quantization.default_qconfig
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)
exit()
model_32_prepared = torch.quantization.prepare(model)
model_32_prepared.d_device = torch.device("cpu")

input_fp32 = torch.from_numpy(Xtest).float()
model_int8 = torch.quantization.convert(model_32_prepared)


model_int8.to("cpu")
model_int8.eval()
print("size of model after quantization: ", get_size_of_model(model_int8))

# eval
Xtest_tensor = Xtest_tensor.to('cpu')
# Xtest_tensor = torch.Tensor(Xtest_tensor)#.to('cpu')
model_int8.d_device = torch.device("cpu")
Ytest_pred_static_quantized = model_int8(Xtest_tensor)
Ytest_pred_static_quantized = scaler.inverse_transform(Ytest_pred_static_quantized.detach().numpy())


# mape
mape_static_quantized = np.mean(np.abs((Ytest - Ytest_pred_static_quantized) / Ytest)) * 100

plt.figure(figsize=(14, 7))
plt.plot(Ytest, label='True temperature', color='blue')
plt.plot(Ytest_pred_static_quantized, label='Static Quantized predicted temperature', color='red')
plt.title('Temperature static quantized prediction')
plt.legend()
plt.savefig("./res_img/lstm_after_qt.png")
print(mape_static_quantized)
print(print_model_size(model_int8))

# compare output type
print("Output type comparison")
print(Ytest_pred_static_quantized)
