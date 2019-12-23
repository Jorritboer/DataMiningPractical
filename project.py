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
  data = data.transpose()
  trip_times = data[10] # extract trip times because we don't want those in PCA
  data = np.delete(data, 10, 0)
  data = np.delete(data, 9, 0)
  data = np.delete(data, 3, 0)
  data = np.delete(data, 2, 0)# remove pickuptime (remove this line when we have added time formatting)
  data = np.delete(data, 0, 0)

  # for i in range(0,len(data[0])):
  #   # convert time
  #   data[1][i] = time.mktime( datetime.datetime.strptime( data[1][i], '%Y-%m-%d %H:%M:%S').timetuple() )
  data = data.astype(float) # necessary for PCA

  return data, trip_times

def PCA(data):
  # Returns the principal components
  means = data.mean(axis=1)
  centered_data = data.transpose() - means

  U,sv,Vt = np.linalg.svd(centered_data)
  V = Vt.transpose()
  return V

def visualizePCA(principal_components, data):
  A = np.dot(data.transpose(), principal_components[0,:])
  B = np.dot(data.transpose(), principal_components[1,:])
  plt.scatter(A, B)
  plt.show()

data = np.array(pd.read_csv('reduced_train.csv'))
treshold = 3
filtered_data = removeOutliars(data, treshold)
# visualizeOutliars(data,filtered_data,treshod)
formatted_data, trip_times = format_data(filtered_data)

principal_components = PCA(formatted_data)
visualizePCA(principal_components, formatted_data)