import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def weighted_accuracy(pred, true):
    assert(len(pred) == len(true))
    num_labels = len(true)
    num_pos = sum(true)
    num_neg = num_labels - num_pos
    frac_pos = num_pos/num_labels
    weight_pos = 1/frac_pos
    weight_neg = 1/(1-frac_pos)
    num_pos_correct = 0
    num_neg_correct = 0
    for pred_i, true_i in zip(pred, true):
        num_pos_correct += (pred_i == true_i and true_i == 1)
        num_neg_correct += (pred_i == true_i and true_i == 0)
    weighted_accuracy = ((weight_pos * num_pos_correct)
                         + (weight_neg * num_neg_correct))/((weight_pos * num_pos) + (weight_neg * num_neg))
    return weighted_accuracy

# Loading training features and computing labels.
df = pd.read_csv("./data/train_2016.csv", sep=',',header=None, encoding='unicode_escape')
data = df.to_numpy()
data = data[1:, 1:]
ordA = ord("A")
for county in data:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[3] = county[3].replace(",", "")
data = data.astype(np.float32)
tmp = np.copy(data[:,0])
data[:,0] = data[:,2]
data[:,2] = tmp
yTr = data[:,1]-data[:, 0]
assert np.sum(np.where(yTr<=0, 0, 1))==225
xTr = data[:, 2:]
# xTr = data[:, 3:]

df = pd.read_csv("./data/train_2012.csv", sep=',',header=None, encoding='unicode_escape')
data = df.to_numpy()
data = data[1:, 1:]
ordA = ord("A")
for county in data:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[3] = county[3].replace(",", "")
data = data.astype(np.float32)
tmp = np.copy(data[:,0])
data[:,0] = data[:,2]
data[:,2] = tmp
yTr2 = data[:,1]-data[:, 0]
#yTr = yTr.astype(np.long)
assert np.sum(np.where(yTr2<=0, 0, 1))==322
xTr2 = data[:, 3:]

weight2016 = 3
xTr = np.hstack((xTr, xTr2))
yTr = np.sign(weight2016*yTr + yTr2)
yTr = np.where(yTr==-1, 0, yTr)
yTr = yTr.reshape((len(yTr), 1))

# Loading test features.
testdf = pd.read_csv("./data/test_2016_no_label.csv", sep=',',header=None, encoding='unicode_escape')
xTe = testdf.to_numpy()
FIPS = xTe[1:, 0]
xTe = xTe[1:, 1:]
ordA = ord("A")
for county in xTe:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[1] = county[1].replace(",", "")
xTe = xTe.astype(np.float32)
# xTe = xTe[:,1:]

testdf = pd.read_csv("./data/test_2012_no_label.csv", sep=',',header=None, encoding='unicode_escape')
xTe2 = testdf.to_numpy()
FIPS = xTe2[1:, 0]
xTe2 = xTe2[1:, 1:]
ordA = ord("A")
for county in xTe2:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[1] = county[1].replace(",", "")
xTe2 = xTe2.astype(np.float32)

xTe = np.hstack((xTe,xTe2[:,1:]))

def preprocess(xTr, xTe):
    """
    Preproces the data to make the training features have zero-mean and
    standard-deviation 1
    OUPUT:
        xTr - nxd training data
        xTe - mxd testing data
    OUPUT:
        xTr - pre-processed training data
        xTe - pre-processed testing data
        s,m - standard deviation and mean of xTr
            - any other data should be pre-processed by x-> (x-m)/s
    (The size of xTr and xTe should remain unchanged)
    """
    ntr, d = xTr.shape
    nte, _ = xTe.shape
    m = np.mean(xTr, axis=0)
    s = np.std(xTr, axis=0)
    xTr = (xTr-m)/s
    xTe = (xTe-m)/s
    return xTr, xTe

class NPDataset(Dataset):
    def __init__(self, x, y=None):
        # Where the initial logic happens like reading a csv, doing data augmentation, etc.
        assert y is None or len(x)==len(y)
        self.xdata = x; self.ydata = y

    def __len__(self):
        # Returns count of samples (an integer) you have.
        return len(self.xdata)

    def __getitem__(self, idx):
        # Given an index, returns the correponding datapoint.
        # This function is called from dataloader like this:
        # img, label = CSVDataset.__getitem__(99)  # For 99th item
        if self.ydata is None:
            return self.xdata[idx]
        return self.xdata[idx], self.ydata[idx]

class Net(nn.Module):
    def __init__(self, d):
        """
        You need to initialize most NN layers here.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        """
        Define in what order the input x is forwarded through all the NN layers to become a final output.
        """
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x =  F.relu(self.fc3(x))
        x =  self.fc4(x)
        return x

    def predict(self,x):
        #Apply softmax to output.
        pred = F.softmax(self.forward(x), dim=1)
        ans = []
        #Pick the class with maximum weight
        return torch.max(pred, 1)[1]


def trainNetwork(xtrain, ytrain, xtest):
    n, d = xtrain.shape
    net = Net(d)

    num_labels = len(ytrain)
    num_pos = sum(ytrain)
    frac_pos = num_pos/num_labels
    weight_pos = 1/frac_pos
    weight_zer = 1/(1-frac_pos)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([weight_zer, weight_pos]))

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    deepDataSet = NPDataset(xtrain, ytrain)
    dataloader = DataLoader(dataset = deepDataSet, batch_size = 50, shuffle=True)

    e_losses = []
    num_epochs = 15
    for e in range(num_epochs):
        e_losses += oneEpoch(net, optimizer, criterion, dataloader)
    # plt.plot(e_losses)
    # plt.show()
    return net

def oneEpoch(net, optimizer, criterion, dataloader):
    net.train()
    losses = []
    for bid, (input, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target.squeeze().long())
        loss.backward()
        optimizer.step()
        losses.append(loss.data.numpy())
    return losses

def deepnnClassifier(xt, yt, xv):
    xt, xv = preprocess(xt, xv)
    net = trainNetwork(xt, yt, xv)
    net.eval()
    preds = net.predict(torch.from_numpy(xv).type(torch.FloatTensor)).detach().numpy()
    return preds.astype(np.int)

# Validation, Training and Model Selection
def kFoldCross(xTr, yTr, k, classifier):
    """
    kFoldCross for Estimating Prediction Error and Calculating Training Error.
    """
    totTrain = 0; totVal = 0;
    for i in range(0, k):
        splitIndices = (i*len(xTr)//k, (i+1)*len(xTr)//k)
        if 0 in splitIndices:
            xt = xTr[splitIndices[1]:]
            yt = yTr[splitIndices[1]:]
        elif k in splitIndices:
            xt = xTr[:splitIndices[0]]
            yt = yTr[:splitIndices[0]]
        else:
            xt = np.concatenate((xTr[:splitIndices[0]], xTr[splitIndices[1]:]))
            yt = np.concatenate((yTr[:splitIndices[0]], yTr[splitIndices[1]:]))
        xv = xTr[splitIndices[0]:splitIndices[1]]
        yv = yTr[splitIndices[0]:splitIndices[1]]
        trainAcc = weighted_accuracy(classifier(xt, yt, xt), yt)
        valAcc =  weighted_accuracy(classifier(xt, yt, xv), yv)
        totTrain+= trainAcc
        totVal+= valAcc
    return totTrain/k, totVal/k

trainAcc, valAcc = kFoldCross(xTr, yTr, 5, deepnnClassifier)
print("Average Training Accuracy: "+str(trainAcc))
print("Average Validation Accuracy: "+str(valAcc))

preds = deepnnClassifier(xTr, yTr, xTe)
preddf = pd.DataFrame(data=preds, index=FIPS, columns=["Result"])
preddf.index.name = "FIPS"
preddf.to_csv("creativepreds.csv")
