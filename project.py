import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import datetime

def removeOutliars(data, treshold):
  # outliar removal
  # treshold: maximum amount of hours considered a reasonable trip length
  return data[ data[:,10] < (treshold * 60 * 60) ]

def visualizeOutliars(initial_data, filtered_data, treshold):
  plt.subplot(211)
  plt.title('Trip lengths before outliar removal')
  plt.xlabel('Trip length in seconds')
  plt.boxplot(initial_data[:,10], vert=False) # shows very very big outliars
  plt.subplot(212)
  plt.title('Trip lengths after trips above ' + str(treshold) + ' are removed')
  plt.xlabel('Trip length in seconds')
  plt.boxplot(filtered_data[:,10], vert=False) # outliars removed

  plt.show()

def format_data(data):
  # remove the non useful data for pca
  # converts time to useful measure
  # extracts trip duration
  # returns: [vendor id, pickup time formatted, passenger count, pickup pos, dropoff pos], trip duration
  trip_times = data[:,10] # extract trip times because we don't want those in PCA
  data = np.delete(data, 10, 1)
  data = np.delete(data, 9, 1)
  data = np.delete(data, 3, 1)
  data = np.delete(data, 2, 1)# remove pickuptime (remove this line when we have added time formatting)
  data = np.delete(data, 0, 1)

  # for i in range(0,len(data[0])):
  #   # convert time
  #   data[1][i] = time.mktime( datetime.datetime.strptime( data[1][i], '%Y-%m-%d %H:%M:%S').timetuple() )
  data = data.astype(float) # necessary for PCA

  return data, trip_times

def PCA(data):
  # Returns the principal components and singular values
  means = data.mean(axis=0)
  centered_data = data - means
  U,sv,Vt = np.linalg.svd(centered_data)
  V = Vt.transpose()
  return V,sv

def visualizePCA(principal_components, data):
  A = np.dot(data.transpose(), principal_components[0,:])
  B = np.dot(data.transpose(), principal_components[1,:])
  plt.scatter(A, B)
  plt.show()

def visualizeVariance(sv):
 squaredSV = sum(list(map(lambda x: x**2, sv)))
 for i in range(len(sv)):
    plt.bar(i, ((sv[i]**2)/squaredSV))
 plt.show()

def projectDataOntoPC(data, pcs, n):
  # project data onto the first n pcs
  a = np.dot(data,pcs[:,0:n])
  return a

data = np.array(pd.read_csv('reduced_train.csv'))
treshold = 3
filtered_data = removeOutliars(data, treshold)
#visualizeOutliars(data,filtered_data,treshold)
formatted_data, trip_times = format_data(filtered_data)

# principal_components, sv = PCA(formatted_data)
# projected_data = projectDataOntoPC(formatted_data, principal_components, 2)

# visualizePCA(principal_components, formatted_data)
# visualizeVariance(sv)