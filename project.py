#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from geopy import distance
from sklearn.model_selection import cross_val_score

#%%
def removeOutliars(data, treshold):
  # outliar removal
  # treshold: maximum amount of hours considered a reasonable trip length
  return data[ data[:,10] < (treshold * 60 * 60) ]

#%%
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

#%%
def format_data(data):
  # remove the non useful data for pca
  # converts time to useful measure
  # extracts trip duration
  # returns: [vendor id, passenger count, pickup pos, dropoff pos, day of the year, day of the week, time, distance], trip duration
  trip_times = data[:,10] # extract trip times because we don't want those in PCA
  pickup_times = data[:,2] # extract pickup times because we want to format it 
  locations = data[:,5:9] # extract locations because we want to format those
  data = np.delete(data, 10, 1) # trip duration 
  data = np.delete(data, 9, 1) # store_and_fwd_flag
  data = np.delete(data, 3, 1) # pickup datetime
  data = np.delete(data, 2, 1) # dropoff datetime
  data = np.delete(data, 0, 1) # id

  times = []
  for i,t in enumerate(pickup_times):
    # convert time
    time = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timetuple()
    times.append([time.tm_yday, time.tm_wday, 60 * time.tm_hour + time.tm_min])
    #we added columns: days away from january 1st, day of the week from sunday, minutes from 0:00   
  data = np.append(data,times, axis=1)

  distances = [] # calculates euclidian distance in meters
  for (pick_long, pick_lat, drop_long, drop_lat) in locations:
    distances.append( [distance.distance((pick_lat, pick_long), (drop_lat,drop_long)).m] )
  data = np.append(data,distances, axis=1)

  data = data.astype(float) # necessary for PCA
  data = preprocessing.scale(data)
  return data, trip_times

#%%
def PCA(data):
  # Returns the principal components and singular values
  means = data.mean(axis=0)
  centered_data = data - means
  U,sv,Vt = np.linalg.svd(centered_data)
  V = Vt.transpose()
  return V,sv

#%%
def visualizeVarianceBar(sv):
  squaredSV = sum(list(map(lambda x: x**2, sv)))
  for i in range(len(sv)):
     plt.bar(i, ((sv[i]**2)/squaredSV))
  plt.ylabel('Variance')
  plt.xlabel('Principal Components')
  plt.show()

#%%
def visualizeVarianceLine(sv):
  squaredSV = sum(list(map(lambda x: x**2, sv)))
  variances = []
  for i in range(len(sv)):
     variances.append(((sv[i]**2)/squaredSV))
  plt.plot(variances)
  plt.ylabel('Variance')
  plt.xlabel('Principal Components')
  plt.show()

#%%
def projectDataOntoPC(data, pcs, n):
  # project data onto the first n pcs
  a = np.dot(data,pcs[:,0:n])
  return a

#%%
def linearRegression(X, y):
  return LinearRegression().fit(X,y)

#%%
def plotDifferenceScatter(pcaData, normalData, tripTimes):
  X_train, X_test, y_train, y_test = train_test_split(pcaData, trip_times, test_size=0.1)
  model = linearRegression(X_train, y_train)

  print('Score for PCA model', model.score(X_test,y_test))
  yResult = model.predict(X_test)
  plt.figure(figsize=(10,5))
  # plt.hist(yResult - y_test, bins=20, label='With PCA', range=(-2000,2000))
  plt.subplot(121)
  plt.scatter(y_test, yResult )
  plt.plot([0,10000],[0,10000], color='green')
  plt.title('With PCA')
  plt.xlabel('Actual time')
  plt.ylabel('Predicted time')
  plt.xlim(0,4500)
  plt.ylim(0,4500)

  X_train, X_test, y_train, y_test = train_test_split(normalData, trip_times, test_size=0.1)
  model = linearRegression(X_train, y_train)

  print('Score for normal model', model.score(X_test,y_test))
  yResult = model.predict(X_test)
  # plt.hist(yResult - y_test, bins=20, histtype='step', label='Without PCA', range=(-2000,2000))
  plt.subplot(122)
  plt.scatter(y_test, yResult, color='orange' )
  plt.plot([0,10000],[0,10000], color='green')
  plt.title('Without PCA')
  plt.xlabel('Actual time')
  plt.ylabel('Predicted time')
  plt.xlim(0,4500)
  plt.ylim(0,4500)
  plt.show()


def plotDifferenceBar(pcaData, normalData, tripTimes):
  X_train, X_test, y_train, y_test = train_test_split(pcaData, trip_times, test_size=0.1)
  model = linearRegression(X_train, y_train)

  print('Score for PCA model', model.score(X_test,y_test))
  yResult = model.predict(X_test)
  plt.hist(yResult - y_test, bins=20, label='With PCA', range=(-2000,2000))
  plt.xlabel('Difference between predicted and actual trip time')
  plt.ylabel('Amount of trips')

  X_train, X_test, y_train, y_test = train_test_split(normalData, trip_times, test_size=0.1)
  model = linearRegression(X_train, y_train)

  print('Score for normal model', model.score(X_test,y_test))
  yResult = model.predict(X_test)
  plt.hist(yResult - y_test, bins=20, histtype='step', label='Without PCA', range=(-2000,2000))
  plt.legend()
  plt.ylabel('Amount of trips')
  plt.xlabel('Difference between predicted and actual trip time')
  plt.show()

#%%
def cross_validation_scores(pcaData, normalData, tripTimes):
  model = LinearRegression()
  errPca = cross_val_score(model, pcaData, trip_times, cv=10)
  errNoPca = cross_val_score(model, normalData, trip_times, cv=10)
  print('For the pca data the errors are: ', errPca, ' with an averag of ', (sum(errPca)/len(errPca)))
  print('For the not pca data the errors are: ', errNoPca, ' with an averag of ', (sum(errNoPca)/len(errNoPca)))

#%%
data = np.array(pd.read_csv('reduced_train.csv'))
print(np.shape(data))
treshold = 3
filtered_data = removeOutliars(data, treshold)
# visualizeOutliars(data,filtered_data,treshold)
formatted_data, trip_times = format_data(filtered_data)

#%%
principal_components, sv = PCA(formatted_data)
projected_data = projectDataOntoPC(formatted_data, principal_components, 6)

#%%
plotDifferenceScatter(projected_data, formatted_data, trip_times)

# visualizeVarianceLine(sv)

# %%
