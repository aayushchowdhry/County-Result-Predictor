import pandas as pd
import numpy as np
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Loading training features.
df = pd.read_csv("train_2016.csv", sep=',',header=None, encoding='unicode_escape')
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

# Generating Labels
yTr = np.sign(data[:,1]-data[:, 0])
yTr = np.where(yTr==-1, 0, yTr)
assert np.sum(yTr)==225 # sanity check

# Loading test features.
testdf = pd.read_csv("test_2016_no_label.csv", sep=',',header=None, encoding='unicode_escape')
xTe = testdf.to_numpy()
FIPS = xTe[1:, 0] # Storing to later write preds file.
xTe = xTe[1:, 1:]
ordA = ord("A")
for county in xTe:
    # This converts the state of the county to a unique number between 1 and 26*26.
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[1] = county[1].replace(",", "")
xTe = xTe.astype(np.float32)

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

# First training algorithm: weighted kNNClassifier
def kNNClassifier(xtrain, ytrain, xtest, k):
    """
    Weighted kNN Classifer

    Parameters:
        xtrain: nxd training features.
        ytrain: n training labels.
        xtest: mxd test features (features one wants predictions of)
        k: number of neighbours to run kNN with.
    Returns:
        binary predictions for xtest after learning from given data.
    """
    xtrain, xtest = preprocess(xtrain, xtest)
    neigh = KNeighborsClassifier(n_neighbors=k, weights="distance")
    neigh.fit(xtrain, ytrain)
    return neigh.predict(xtest)

# Second training algorithm: soft-margin SVM
def SVMClassifier(xtrain, ytrain, xtest, C, kernel):
    """
    Soft Margin SVM

    Parameters:
        xtrain: nxd training features.
        ytrain: n training labels.
        xtest: mxd test features (features one wants predictions of)
        C: Weightage given to slack variables.
        Kernel: The kernel with which to run the SVM.
    Returns:
        binary predictions for xtest after learning from given data.
    """
    xtrain, xtest = preprocess(xtrain, xtest)
    sv = SVC(C=C, kernel=kernel)
    sv.fit(xtrain, ytrain)
    return sv.predict(xtest)

# Validation, Training and Model Selection
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

# k = 5
# bestAcc = -float("inf")
# # Generating Predictions
# for i in range(1, 11): # parameter selection for number of neighbours from 1-11
#     print("Running "+str(k)+"-FoldCross for "+str(i)+"-NNClassifier.")
#     trainAcc, valAcc = kFoldCross(xTr, yTr, 5, lambda xt, yt, xv: kNNClassifier(xt, yt, xv, i))
#     print("Average Training Accuracy: "+str(trainAcc))
#     print("Average Validation Accuracy: "+str(valAcc))
#     print()
#     if valAcc>bestAcc:
#         bestAcc = valAcc
#         bestK = i

# Manually found SVM has better performance using printed validation values and McNemar's test.

k = 5 # for kfoldcross
bestAcc = -float("inf")
Cs = [50.30, 50.31, 50.32, 50.33, 50.34] # good range of C values found through manual telescopic search
for C in Cs:
    print("Running SVM with C "+str(C)+" and rbf kernel.")
    trainAcc, valAcc = kFoldCross(xTr, yTr, 5, lambda xt, yt, xv: SVMClassifier(xt, yt, xv, C, "rbf"))
    print("Average Training Accuracy: "+str(trainAcc))
    print("Average Validation Accuracy: "+str(valAcc))
    print()
    if valAcc>bestAcc:
        bestAcc = valAcc
        bestC = C

# Writing CSV File
preds = SVMClassifier(xTr, yTr, xTe, bestC, "rbf").astype(np.int)
preddf = pd.DataFrame(data=preds, index=FIPS, columns=["Result"])
preddf.index.name = "FIPS"
preddf.to_csv("preds.csv")
