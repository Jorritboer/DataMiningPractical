import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
  # returns: [vendor id, passenger count, pickup pos, dropoff pos, day of the year, day of the week, time], trip duration
  trip_times = data[:,10] # extract trip times because we don't want those in PCA
  pickup_times = data[:,2] # extract pickup times because we want to format it 
  data = np.delete(data, 10, 1)
  data = np.delete(data, 9, 1)
  data = np.delete(data, 3, 1)
  data = np.delete(data, 2, 1)
  data = np.delete(data, 0, 1)

  times = []
  for i,t in enumerate(pickup_times):
    # convert time
    time = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timetuple()
    times.append([time.tm_yday, time.tm_wday, 60 * time.tm_hour + time.tm_min])
    #we added columns: days away from january 1st, day of the week from sunday, minutes from 0:00   
  data = np.append(data,times, axis=1)
  data = data.astype(float) # necessary for PCA
  data = preprocessing.scale(data)
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

def linearRegression(X, y):
  return LinearRegression().fit(X,y)

def plotDifference(pcaData, normalData, tripTimes):
  X_train, X_test, y_train, y_test = train_test_split(pcaData, trip_times, test_size=0.1)
  model = linearRegression(X_train, y_train)

  yResult = model.predict(X_test)
  plt.hist(yResult - y_test, bins=100, label='With PCA')

  X_train, X_test, y_train, y_test = train_test_split(normalData, trip_times, test_size=0.1)
  model = linearRegression(X_train, y_train)

  yResult = model.predict(X_test)
  plt.hist(yResult - y_test, bins=100, histtype='step', label='Without PCA')
  plt.legend()
  plt.show()


data = np.array(pd.read_csv('reduced_train.csv'))
treshold = 3
filtered_data = removeOutliars(data, treshold)
# visualizeOutliars(data,filtered_data,treshold)
formatted_data, trip_times = format_data(filtered_data)

principal_components, sv = PCA(formatted_data)
projected_data = projectDataOntoPC(formatted_data, principal_components, 9)

plotDifference(projected_data, formatted_data, trip_times)

# visualizePCA(principal_components, formatted_data)
visualizeVariance(sv)