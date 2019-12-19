import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import datetime

data = np.array(pd.read_csv('train.csv'))

# outliar removal
treshold = 3 # maximum amount of hours considered a reasonable trip length
filtered_data = data[ data[:,10] < (treshold * 60 * 60) ]

# visualize outliar removal
# plt.subplot(211)
# plt.title('Trip lengths before outliar removal')
# plt.xlabel('Trip length in seconds')
# plt.boxplot(data[:,10], vert=False) # shows very very big outliars
# plt.subplot(212)
# plt.title('Trip lengths after trips above ' + str(treshold) + ' are removed')
# plt.xlabel('Trip length in seconds')
# plt.boxplot(filtered_data[:,10], vert=False) # outliars removed

# plt.show()

# PCA
filtered_data = filtered_data.transpose()
trip_times = filtered_data[10] # extract trip times because we don't want those in PCA
filtered_data = np.delete(filtered_data, 10, 0)
filtered_data = np.delete(filtered_data, 9, 0)
filtered_data = np.delete(filtered_data, 3, 0)
filtered_data = np.delete(filtered_data, 2, 0)
filtered_data = np.delete(filtered_data, 0, 0)

# for i in range(0,len(filtered_data[0])):
  # convert time
  # filtered_data[1][i] = time.mktime( datetime.datetime.strptime( filtered_data[1][i], '%Y-%m-%d %H:%M:%S').timetuple() )

filtered_data = filtered_data.astype(float)
means = filtered_data.mean(axis=1)
centered_filtered_data = filtered_data.transpose() - means

print(type(centered_filtered_data))
U,sv,Vt = np.linalg.svd(centered_filtered_data)
V = Vt.transpose()
print(np.shape(V))

A = np.dot(filtered_data.transpose(), V[0,:])
B = np.dot(filtered_data.transpose(), V[1,:])
plt.scatter(A, B)
plt.show()