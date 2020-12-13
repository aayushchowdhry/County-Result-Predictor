import pandas as pd
import numpy as np
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
df = pd.read_csv("train_2016.csv", sep=',',header=None, encoding='unicode_escape')
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
yTr = np.sign(data[:,1]-data[:, 0])
yTr = np.where(yTr==-1, 0, yTr)
assert np.sum(yTr)==225
xTr = data[:, 2:]
# xTr = data[:, 3:]

# Loading test features.
testdf = pd.read_csv("test_2016_no_label.csv", sep=',',header=None, encoding='unicode_escape')
xTe = testdf.to_numpy()
FIPS = xTe[1:, 0]
xTe = xTe[1:, 1:]
ordA = ord("A")
for county in xTe:
    county[0] = (ord(county[0][-2])-ordA)*26+(ord(county[0][-1])-ordA)
    county[1] = county[1].replace(",", "")
xTe = xTe.astype(np.float32)
# xTe = xTe[:,1:]

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

# First training algorithm: kNNClassifier
def kNNClassifier(xtrain, ytrain, xtest, k):
    xtrain, xtest = preprocess(xtrain, xtest)
    neigh = KNeighborsClassifier(n_neighbors=k, weights="distance")
    neigh.fit(xtrain, ytrain)
    return neigh.predict(xtest)

# Second training algorithm: SVM
def SVMClassifier(xtrain, ytrain, xtest, C, kernel):
    xtrain, xtest = preprocess(xtrain, xtest)
    sv = SVC(C=C, kernel=kernel)
    sv.fit(xtrain, ytrain)
    return sv.predict(xtest)

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
        totTrain+= weighted_accuracy(classifier(xt, yt, xt), yt)
        totVal+= weighted_accuracy(classifier(xt, yt, xv), yv)
    return totTrain/k, totVal/k


# k = 10
# bestAcc = -float("inf")
# # Generating Predictions
# for i in range(1, 11):
#     print("Running "+str(k)+"-FoldCross for "+str(i)+"-NNClassifier.")
#     trainAcc, valAcc = kFoldCross(xTr, yTr, 5, lambda xt, yt, xv: kNNClassifier(xt, yt, xv, i))
#     print("Average Training Accuracy: "+str(trainAcc))
#     print("Average Validation Accuracy: "+str(valAcc))
#     print()
#     if valAcc>bestAcc:
#         bestAcc = valAcc
#         bestK = i

k = 5
bestAcc = -float("inf")
Cs = [50.30, 50.31, 50.32, 50.33, 50.34]
# Generating Predictions
for C in Cs:
    print("Running SVM with C "+str(C)+" and rbf kernel.")
    trainAcc, valAcc = kFoldCross(xTr, yTr, 5, lambda xt, yt, xv: SVMClassifier(xt, yt, xv, C, "rbf"))
    print("Average Training Accuracy: "+str(trainAcc))
    print("Average Validation Accuracy: "+str(valAcc))
    print()
    if valAcc>bestAcc:
        bestAcc = valAcc
        bestC = C

# print("Running NN.")
# trainAcc, valAcc = kFoldCross(xTr, yTr, 5, lambda xt, yt, xv: deepnnClassifier(xt, yt, xv))
# print("Average Training Accuracy: "+str(trainAcc))
# print("Average Validation Accuracy: "+str(valAcc))
# print()


# Writing CSV File
print("Best C: "+str(bestC))
preds = SVMClassifier(xTr, yTr, xTe, bestC, "rbf").astype(np.int)
preddf = pd.DataFrame(data=preds, index=FIPS, columns=["Result"])
preddf.index.name = "FIPS"
preddf.to_csv("preds.csv")


# TODO
# 2.4 Explanation in Words:
# You need to answer the following questions in the markdown cell after this cell:

# 2.4.1 How did you preprocess the dataset and features?

# 2.4.2 Which two learning methods from class did you choose and why did you made the choices?

# 2.4.3 How did you do the model selection?

# 2.4.4 Does the test performance reach a given baseline 68% performanc? (Please include a screenshot of Kaggle Submission)

################################################################################

# Type Markdown and LaTeX:
# Part 3: Creative Solution
# 3.1 Open-ended Code:
# You may follow the steps in part 2 again but making innovative changes like creating new features, using new training algorithms, etc. Make sure you explain everything clearly in part 3.2. Note that reaching the 75% creative baseline is only a small portion of this part. Any creative ideas will receive most points as long as they are reasonable and clearly explained.

# # Make sure you comment your code clearly and you may refer to these comments in the part 3.2
# # TODO
# 3.2 Explanation in Words:
# You need to answer the following questions in a markdown cell after this cell:

# 3.2.1 How much did you manage to improve performance on the test set compared to part 2? Did you reach the 75% accuracy for the test in Kaggle? (Please include a screenshot of Kaggle Submission)

# 3.2.2 Please explain in detail how you achieved this and what you did specifically and why you tried this.

# Type Markdown and LaTeX: 𝛼2
# Part 4: Kaggle Submission
# You need to generate a prediction CSV using the following cell from your trained model and submit the direct output of your code to Kaggle. The CSV shall contain TWO column named exactly "FIPS" and "Result" and 1555 total rows excluding the column names, "FIPS" column shall contain FIPS of counties with same order as in the test_2016_no_label.csv while "Result" column shall contain the 0 or 1 prdicaitons for corresponding columns. A sample predication file can be downloaded from Kaggle.

# TODO
# You may use pandas to generate a dataframe with FIPS and your predictions first
# and then use to_csv to generate a CSV file.
# Part 5: Resources and Literature Used
# Type Markdown and LaTeX: 𝛼
