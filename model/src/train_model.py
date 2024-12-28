import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from model.src.model import Model

# create instance of Model
model = Model()

# read dataset
dataset = pd.read_csv("model/data/iris.csv")

# replace the name of the flowers with a number to classify them later
dataset["variety"] = dataset["variety"].replace("Setosa", 0)
dataset["variety"] = dataset["variety"].replace("Versicolor", 1)
dataset["variety"] = dataset["variety"].replace("Virginica", 2)

# split dataset for supervised learning
X = dataset.drop("variety", axis = 1)
y = dataset["variety"]

# convert to numpy arrays
X = X.values
y = y.values

# split the Data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 20)

# convert test and train data to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# setup criterion to messure the loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# deffine number of epochs
epochs = 150
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    
    print(f"Epoch: {i} and loss: {loss}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("Training Done")

# plot the loss in relation to epochs
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
plt.show()


