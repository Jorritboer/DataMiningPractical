import numpy as np
import pandas as pd 

data = pd.read_csv('train.csv')

data = data.drop(range(int(len(data)/100), len(data)))

data.to_csv(r'./reduced_train.csv')
