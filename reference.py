"""
Three step process for machine learning, ignoring a bunch of important things
we talked about in class:

1. Get data
    1.1 Get a Dataset---this is provided by PyTorch
    1.2 Get a DataLoader---this is provided by PyTorch
2. Get a classifier, and other paraphenalia needed for training
    2.1 Create a classifier architecture, and instantiate it
    2.2 Get a criterion (loss function)
    2.3 Get an optimizer (thing that does parameter updates to the classifier)
3. Classify the data in some reasonably intelligent fashion. This will involve:
    3.1 A one_epoch() function that trains the classifier for one epoch---a
        complete iteration over the dataset from (1.1)
    3.2 A loop that calls one_epoch() the number of times we want to run an
        epoch.
4. ...
5. $$$
"""
# Have you ever *not* needed NumPy?
import numpy as np

# We need to import a bunch of things from PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# tqdm gives us nice progress bars, which provide a huge quality-of-life boost,
# so it's worth mentioning here.
from tqdm import tqdm

# 'something_from_torch = something_from_torch.to(device)' puts the thing onto
# the device. If you have CUDA available, this will massively speed
# computations. Note that all inputs to a computation must be on the same
# device, and that things are on the CPU by default.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setting a random seed means we can reproduce our results easily. In my
# understanding, this is generally good practice.
np.random.seed(1701)
torch.manual_seed(1701)

# Here I define a bunch of constants that'll be needed later on. I've chosen the
# first two arbitrarily; the last three are set to values that are probably
# decent across a wide range of tasks.
input_dim = 42          # This is the dimensionality of the inputs X
output_dim = 314159     # This is the number of classes in Y
num_workers = 4         # We want to parallelize loading data during training.
                        #   Otherwise, it's possible for moving data to model to
                        #   become a bottleneck in performance!
batch_size = 16         # This is the number of examples (x,y) that are run
                        #   through the neural net simultaneously. Think of it as
                        #   a mid-ground between SGD and GD.
num_epochs = 100        # The number of complete iterations over the dataset
                        #   made during training

save_file = "best_model.pt" # It's helpful to save the best-performing model
                            #   throughout training...

# 1.1---get the Dataset. A Dataset should extend PyTorch's base class, and needs
# to do two things:
# a) implement the __getitem__() method. This takes in an index to the data and
#       returns the x- and y-values of the ith datapoint, often as tensors.
# b) implement the __len__() method. This is just the amount of data in the
#       dataset.
class ToyDataset(Dataset):
    """An example dataset."""

    def __init__(self):
        pass

    def __getitem__(self, i):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

train_dataset = ToyDataset()

# 1.2---get a DataLoader. This wraps a Dataset and allows you to iterate through
# it. It's also where you can implement *a lot* of performance optimizations.
# 'shuffle=True' is set by default, but it's worth emphasizing here.
#
# As long as values for 'batch_size' are reasonable (4 <= x << amount of data),
# changing it will mostly change the speed and not performance of the model.
# 'num_workers' should be just under your number of CPUs.
train_loader = DataLoader(train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers)

# 2.1---here we define the actual neural net! It should be a class that inherits
# from torch.nn.Module, and has a forward() method that implements the function
#
#       f : X -> Y
#
# Because the neural net is *literally* a function, I'll call its outputs on an
# input 'x' 'fx'.
class NeuralNet(nn.Module):

    # As you know, a neural net is composed of layers. The first layer takes in
    # an example x, does a computation on it, and passes the result fx to the
    # second layer, and this process continues.
    #
    # Note that the architecture below illustrates how to build a neural net,
    # but shouldn't work too well otherwise.
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.f1 = nn.Linear(input_dim , 2)  # The first layer should map from
                                            #   the dimensionality of the input
                                            #   data to another dimensionality.
        self.r1 = nn.ReLU()                 # Activation functions operate
                                            #   element-wise, so they don't need
                                            #   a dimensionality specified, and
                                            #   they don't change the
                                            #   dimensionality of their inputs.
                                            #   Generally every linear layer is
                                            #   followed by an activation
                                            #   function.
        self.fc = nn.Linear(2, output_dim)  # The last layer should map from the
                                            #   last hidden dimensionality to
                                            #   the desired number of classes.
                                            #   At least in computer vision,
                                            #   it's generally called 'fc'.

    # Don't worry about adding an activation function at the end. It's done
    # automatically in the loss function! Of course, this means the outputs of
    # the model need to be softmaxed before they can be used as probabilities.

    # This is the function that implements f : X -> Y. Note that we can expand
    # it as fc(r1(f1(x)))))!
    def forward(self, x):
        fx = self.f1(x)
        fx = self.r1(fx)
        return self.fc(fx)

# Instantiate the model! We need to move it to whatever device it'll be running
# on before we instantiate the optimizer in 2.3.
model = NeuralNet(input_dim, output_dim).to(device)

# 2.2---The criterion is the loss function. Normally, you use CrossEntropy for a
# classification task. (Binary classification is a special case where you can
# get away with doing something else like 0-1 loss.)
criterion = nn.CrossEntropyLoss()

# 2.3---The optimizer takes care of the parameter updates via its step() method.
# To make it work, we need to pass in the 'parameters' of the model. Generally,
# you'll choose between Adam and SGD.
optimizer = optim.Adam(model.parameters())

# 3.1---I find it's useful to define a function implementing a single epoch, and
# then call it repeatedly from within a loop to train the neural net.
def one_epoch(model, train_loader, optimizer, criterion):
    """Returns [model] after being trained for one epoch on [data_loader] using
    [optimizer] and [criterion].
    """
    # [data_loader] returns batches of paired x- and y-values when we iterate
    # over it. Because of PyTorch magic, we don't have to worry about the batch
    # size!
    #
    # What is tqdm doing here? When we wrap an iterable in tqdm() during
    # iteration, we get a super nice progress bar!
    for x,y in tqdm(train_loader, desc="Running batches...", leave=False):
        # x is a tensor of dimension (batch_size x n_features), y is a tensor of
        # dimension (batch_size,). If they're not on the same device as [model],
        # they need to be moved there. (Note that it might be smarter to
        # accomplish this through the DataLoader.)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()   # We only want to accumulate the gradient for
                                #   the current batch. Therefore, we zero out
                                #   the gradient of the model.
        fx = model(x)           # Compute predictions for a batch 'x'
        loss = criterion(fx, y) # Compute the loss function's value on the
                                #   outputs of the model on 'x' and the true
                                #   labels 'y'
        loss.backward()         # Compute the gradients of the model's weights
                                #   with respect to the loss.
        optimizer.step()        # Update parameters by taking an intelligent
                                #   optimizer-determined step against the
                                #   gradient.

    # When the epoch is over, return the model.
    return model

# We can keep track of when after an epoch finishes our model has a better
# validation accuracy than anything else using this. We'll also need a function
# to compute validation accuracy!
best_val_acc = float("-inf")

def validate(model, val_loader):
    """Returns the accuracy of [model] on [val_loader]."""

    def batch_acc(x, y):
        """Returns the number of correct predictions of [model] on [x] given
        [y].
        """
        # Note that PyTorch can do most things that NumPy can. model(x) returns
        # a batch of predictions as a (batch_size x n_classes) tensor; y is a
        # (n_examples,) tensor of in which the i^th index contains the class of
        # the i^th example, eg. 5.
        #
        # Additionally, [model] is on [device], so you'll need to move [x] to
        # the device prior to computation. To move a tensor [t] to the CPU you
        # can call 't.cpu()'---this might be useful for comparing with [y].
        #
        # Can you fit this all on one line?
        raise NotImplementedError()

    # The length of a Dataset is the number of examples in it; the length of a
    # DataLoader is the number of batches. Therefore, to get the accuracy, it's
    # critical to divide by the length of the first and not the second!
    #
    # We can use NumPy here because it's generally (in my experience) faster for
    # things on the CPU than PyTorch.
    return np.sum([batch_acc(x, y) for x,y in val_loader]) / len(val_dataset)

# 3.2---Train the model on the entirety of the data once for every desired
# epoch. tqdm gives a nice progress bar.
for e in tqdm(range(num_epochs), desc="Running epochs..."):
    model = one_epoch(model, train_loader, optimizer, criterion)

    # Suppose we wanted to find the validation accuracy after epoch! All we'd
    # need would be a validation DataLoader, and a function to get accuracy
    # from! Putting this under 'with torch.no_grad():' turns off computing
    # gradients, because we don't need them. Moreover, wrapping it in
    # 'model.eval()' and 'model.train()' is important if we do fancy things like
    # dropout.
    with torch.no_grad():
        model.eval()
        val_acc = validate(model, val_loader)
        model.train()

    # When in the middle of a tqdm-ed loop, we need to use 'tqdm.write()'
    # instead of 'print()'.
    tqdm.write(f"Epoch {e:3} | validation accuracy: {val_acc}")

    # If the validation accuracy is the highest it's ever been, why not save the
    # the model? This is a natural way to figure out the correct number of
    # epochs!
    if val_acc > best_val_acc:
        tqdm.write(f"------Best epoch yet with accuracy {val_acc}. Saved!")
        best_val_acc = val_acc

        # Saving a PyTorch model is super easy! 'model.state_dict()' converts
        # all of its weights to a dictionary format. The general format here is
        # 'torch.save(dictionary, file)'. Note that we can put basically
        # anything we want in this dictionary too!
        torch.save(model.state_dict(), save_file)

# Now that training is done, let's load the best model so we can do something
# with it. First, we need to instantiate a new model of the same type as the
# old one. Then we can call 'load_state_dict()' on the loaded state_dict.
model = NeuralNet(input_dim, output_dim)
best_val_acc_model_weights = torch.load(save_file)
model.load_state_dict(best_val_acc_model_weights)

################################################################################
### WANT TO SEE THE CODE RUN? PASTE THE FOLLOWING CODE DIRECTLY ABOVE 1.2, THEN
### IMPLEMENT 'batch_acc()'.
### It's way beyond the scope of the course though still far too simple for the
### task it takes on; I use it to validate what I've written elsewhere. Also
### note that it downloads the MNIST (handwritten images) dataset to your
### computer, so beware...though you really can just delete it later.
###
### Aside: you should be able to get at least 95% accuracy on the MNIST dataset
### without trying. This gets up to about 70% accuracy because I deliberately
### wrote the worst neural net I could think of that'd illustrate what I needed
### to show.
################################################################################
# import torchvision
# from torchvision import transforms
# from torch.utils.data import Subset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda t: torch.flatten(t)),
# ])
# dataset = torchvision.datasets.MNIST("./MNIST/",
#     train=True, transform=transform, download=True)
# train_dataset = Subset(dataset, range(0, 10000))
# val_dataset = Subset(dataset, range(10000, 11000))
# val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
# input_dim, output_dim, num_epochs = 784, 10, 100
