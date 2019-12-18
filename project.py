import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = np.array(pd.read_csv('train.csv'))

# outliar removal
treshold = 6 # maximum amount of hours considered a reasonable trip length
filtered_data = data[ data[:,10] < (treshold * 60 * 60) ]

# visualize outliar removal
plt.subplot(211)
plt.title('Trip lengths before outliar removal')
plt.xlabel('Trip length in seconds')
plt.boxplot(data[:,10], vert=False) # shows very very big outliars
plt.subplot(212)
plt.title('Trip lengths after trips above ' + str(treshold) + ' are removed')
plt.xlabel('Trip length in seconds')
plt.boxplot(filtered_data[:,10], vert=False) # outliars removed

plt.show()