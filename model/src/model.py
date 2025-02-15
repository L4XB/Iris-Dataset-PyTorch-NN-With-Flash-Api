import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(20)

class Model(nn.Module):
    # input layer with four features => sepal.lenght, sepal.with, petal.lenght, petal.width
    # hidden layer one with 7 neurons
    # hidden layer two with 8 neurons
    # output layer with 3 classifications => "Setosa", "Versicolor", "Virginica" 
    
    def __init__(self, in_features = 4, hl1 = 7, hl2 = 8, classification = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.out = nn.Linear(hl2, classification)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x
        

