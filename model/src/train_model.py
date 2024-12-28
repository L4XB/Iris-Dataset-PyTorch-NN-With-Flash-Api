import pandas as pd
import matplotlib.pyplot as plt

from model.src.model import Model

# create instance of Model
model = Model()

# read dataset
dataset = pd.read_csv("model/data/iris.csv")

print(dataset.head())
