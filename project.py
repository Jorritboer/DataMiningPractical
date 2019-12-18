import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv") 

plt.boxplot(data.trip_duration) # shows very very big outliars
# print(data.id)

plt.show()