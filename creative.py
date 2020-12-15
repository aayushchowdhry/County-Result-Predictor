import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

######################################### Data Loading ###################################################

# Loading training features from 2016.
df = pd.read_csv("./data/train_2016.csv", sep=',',header=None, encoding='unicode_escape')
data = df.to_numpy()
data = data[1:, 1:]
ordA = ord("A")
for county in data:
    # This converts the state of the county to a unique number between 1 and 26*26.
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[3] = county[3].replace(",", "")
data = data.astype(np.float32)
tmp = np.copy(data[:,0])
data[:,0] = data[:,2]
data[:,2] = tmp
xTr = data[:, 2:]

# yTr is currently difference in votes in 2016 between GOP and DEM.
yTr = data[:,1]-data[:, 0]
assert np.sum(np.where(yTr<=0, 0, 1))==225 # Sanity check

# Loading training features from 2012.
df = pd.read_csv("./data/train_2012.csv", sep=',',header=None, encoding='unicode_escape')
data = df.to_numpy()
data = data[1:, 1:]
ordA = ord("A")
for county in data:
    # This converts the state of the county to a unique number between 1 and 26*26.
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[3] = county[3].replace(",", "")
data = data.astype(np.float32)
tmp = np.copy(data[:,0])
data[:,0] = data[:,2]
data[:,2] = tmp
xTr2 = data[:, 3:] # Do not want state to be repeated hence col index is 3 not 2.

# yTr is currently difference in votes in 2012 between GOP and DEM.
yTr2 = data[:,1]-data[:, 0]
assert np.sum(np.where(yTr2<=0, 0, 1))==322 # Sanity Check

# Generate final training set.
xTr = np.hstack((xTr, xTr2)) # Stack 2016 features and 2012 features horizontally to get final xTr.
# Generate labels. Weight 2016 votes more as they are the lastest ones and we are predicting election for 2016 counties.
weight2016 = 4 # Manually set after validation.
yTr = np.sign(weight2016*yTr + yTr2)
yTr = np.where(yTr==-1, 0, yTr)
yTr = yTr.reshape((len(yTr), 1))


# Loading test features from 2016 and 2012 similar to training features.
testdf = pd.read_csv("./data/test_2016_no_label.csv", sep=',',header=None, encoding='unicode_escape')
xTe = testdf.to_numpy()
FIPS = xTe[1:, 0]
xTe = xTe[1:, 1:]
ordA = ord("A")
for county in xTe:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[1] = county[1].replace(",", "")
xTe = xTe.astype(np.float32)

testdf = pd.read_csv("./data/test_2012_no_label.csv", sep=',',header=None, encoding='unicode_escape')
xTe2 = testdf.to_numpy()
xTe2 = xTe2[1:, 1:]
ordA = ord("A")
for county in xTe2:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[1] = county[1].replace(",", "")
xTe2 = xTe2.astype(np.float32)
xTe = np.hstack((xTe,xTe2[:,1:])) # xTe2 is sliced as state isn't repeated in training too.

def preprocess(xTr, xTe):
    """
    Preproces the data by normalizing to make the training features have zero-mean and
    standard-deviation 1.

    Parameters:
        xTr: nxd training data.
        xTe: mxd testing data.
    Returns:
        nxd matrix of pre-processed (normalized) training data.
        mxd matrix of pre-processed (normalized) testing data.
    """
    ntr, d = xTr.shape
    nte, _ = xTe.shape
    m = np.mean(xTr, axis=0)
    s = np.std(xTr, axis=0)
    xTr = (xTr-m)/s
    xTe = (xTe-m)/s
    return xTr, xTe


###################################### Deep Learning Classifier (Neural Net) ##########################################

# Create custom dataset for training and test data.
class NPDataset(Dataset):
    def __init__(self, x, y=None):
        """
        Constructor for Dataset.

        Parameters:
            x: nxd features.
            optional y: n training labels if x is training data set.
        """
        # Where the initial logic happens like reading a csv, doing data augmentation, etc.
        assert y is None or len(x)==len(y)
        self.xdata = x; self.ydata = y

    def __len__(self):
        """
        Overwrite python's len function to return the count of samples (an integer).
        """
        return len(self.xdata)

    def __getitem__(self, idx):
        """
        Function to fetch data at index.

        Parameters:
            idx: index to fetch data from.
        Returns:
            feature vector and coressponding label at idx if dataset is training set.
            feature vector at idx if dataset is test set (y is None)
        """
        if self.ydata is None:
            return self.xdata[idx]
        return self.xdata[idx], self.ydata[idx]

# Define network (3 hidden layers of 20, 20 and 10 neurons respectively. ReLU activation.)
class Net(nn.Module):
    def __init__(self, d):
        """
        Initialize most NN layers here.

        Parameters:
            d: dimension of the feature vectors.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2) # output layer is 2 for cross entropy loss

    def forward(self, x):
        """
        Defines in what order the input x is forwarded through all the NN layers to become a final output.

        Parameters:
            x: nxd input vectors.
        Returns:
            nxd matrix generated after passing x through layers of neural network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x =  F.relu(self.fc3(x))
        x =  self.fc4(x)
        return x

    def predict(self,x):
        """
        Generate binary predictions.

        Parameters:
            x: nxd input vectors.
        Returns:
            n binary predictions for coressponding features.
        """
        pred = F.softmax(self.forward(x), dim=1) #Apply softmax to output.
        return torch.max(pred, 1)[1] #Pick the class with maximum weight

# Train neural net.
def trainNetwork(xtrain, ytrain):
    """
    Train neural network.

    Parameters:
        xtrain: nxd training features.
        ytrain: n training labels.
    Returns:
        trained neural network.
    """
    # Initialize network
    n, d = xtrain.shape
    net = Net(d)

    # Define loss function. Using weighted cross entropy loss due to imbalanced data.
    # Weighted based on number of positive and negative training labels.
    num_labels = len(ytrain)
    num_pos = sum(ytrain)
    frac_pos = num_pos/num_labels
    weight_pos = 1/frac_pos
    weight_zer = 1/(1-frac_pos)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([weight_zer, weight_pos]))

    # Define optimizer.
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Initialize datasets and wrap in data loader.
    deepDataSet = NPDataset(xtrain, ytrain)
    dataloader = DataLoader(dataset = deepDataSet, batch_size = 50, shuffle=True)

    # best combination of batch size, learning rate and number of epochs was found through manual
    # by printing out training and validation accuracy and actively avoiding overfitting.
    e_losses = []
    num_epochs = 15
    for e in range(num_epochs):
        e_losses += oneEpoch(net, optimizer, criterion, dataloader) # Train one epoch and keep track of training loss
    # plt.plot(e_losses)
    # plt.show()
    return net # return the model after training

def oneEpoch(net, optimizer, criterion, dataloader):
    """
    Trains one epoch of given neural network.

    Parameters:
        net: the neural network to train.
        optimizer: the optimizer to use.
        criterion: the loss function to apply.
        dataloader: wrapper of the training data.
    Returns:
        list of losser at each iteration for mini-batches in the epoch.
    """
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

# Generate prediction (final classifier)
def deepnnClassifier(xt, yt, xv):
    """
    Neural Network (Deep Learner).

    Parameters:
        xt: nxd training features.
        yt: n training labels.
        xv: mxd test features (features one wants predictions of)
    Returns:
        binary predictions for xv after learning from given data.
    """
    xt, xv = preprocess(xt, xv)
    net = trainNetwork(xt, yt)
    net.eval()
    preds = net.predict(torch.from_numpy(xv).type(torch.FloatTensor)).detach().numpy()
    return preds.astype(np.int)


########################################## Validation #####################################

def weighted_accuracy(pred, true):
    """
    Compute weighted accuracy of predictions

    Parameters:
        pred: n predictions.
        true: n true labels.
    Returns:
        weighted accuracy of predictions based on number of positive and negative true labels.
    """
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


def kFoldCross(xTr, yTr, k, classifier):
    """
    kFoldCross for Estimating Prediction Error and Calculating Training Error.

    Parameters:
        xTr: nxd training features.
        yTr: n training labels.
        k: number of pieces to break xTr and yTr into. 1 piece is used as validation set.
        classifier: function that takes training features, training labels, and test features as input
                    to give predictions. (the classier one is validating with kFoldCross)
    Returns:
        Average training loss.
        Average validation loss.
    """
    totTrain = 0; totVal = 0;
    for i in range(0, k):
        # data is split into (k-1) pieces to train and 1 piece to validate on
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

        totTrain+= weighted_accuracy(classifier(xt, yt, xt), yt) # compute training accuracy
        totVal+= weighted_accuracy(classifier(xt, yt, xv), yv) # compute validation accuracy

    return totTrain/k, totVal/k # return average training and test error

# Running validation after manually tuning hyperparameters such as learning rate, 2016 weight, number of epochs,
# batch size, loss criterion, optimization method etc.
trainAcc, valAcc = kFoldCross(xTr, yTr, 5, deepnnClassifier)
print("Average Training Accuracy: "+str(trainAcc))
print("Average Validation Accuracy: "+str(valAcc))

###################################### Saving Predictions ########################################
preds = deepnnClassifier(xTr, yTr, xTe)
preddf = pd.DataFrame(data=preds, index=FIPS, columns=["Result"])
preddf.index.name = "FIPS"
preddf.to_csv("creativepreds.csv")
