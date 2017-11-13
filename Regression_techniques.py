import numpy as np
import pandas as pd
import unicodedata
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

df=pd.read_excel('Tunga_Bhadra River Basin Rainfall Data (1).xls',index=False)

#Only picked rows where atleast one non-zero value --> 14862 rows remain
#mse=5.01281658923 with removal
#without this error was 3.91294121404

#df=df.loc[(df.sum(axis=1) != 0.0)]

#Only pick rows where all elements are non-zero
#Error is 27.8114308714 in that case
df=df[(df != 0).all(1)]

lat_long = df.columns

coordinates = []
for i in range(1,len(lat_long)):
	coordinates.append(unicodedata.normalize('NFKD', lat_long[i]).encode('ascii','ignore'))

neighbours = []
lat = 12.625
longi = 76.625
for i in range(0,len(coordinates)):
    a= float(coordinates[i].split('N')[0])
    b= float((coordinates[i].split('N'))[1].split('E')[0])
    if a-lat == 0 and b-longi == 0:
        Y=coordinates[i]
    elif abs(a - lat) <= 0.250 and abs(b - longi) <=0.250:
        neighbours.append(coordinates[i])

X=pd.DataFrame()
for i in range(0,len(neighbours)):
    X[neighbours[i]]=df[neighbours[i]]

Xtrain = X.head(14000)
Ytrain= df[Y].head(14000)

#print Xtrain.shape
#print Ytrain.shape

reg = linear_model.LinearRegression()
reg.fit(Xtrain,Ytrain)

Xtest = X.tail(6000)
Y_actual = df[Y].tail(6000)
Y_predicted=reg.predict(Xtest)
print mean_squared_error(Y_actual, Y_predicted)

plt.scatter(Y_actual, Y_predicted,color='b')
#plt.plot(Xtrain, reg.predict(Xtest),color='k')

plt.show()
Y_actual = Y_actual.reset_index()
Y_actual.drop('index',axis=1,inplace=True)
Y_actual.plot(color='r')

pd.Series(Y_predicted).plot(color='b')
plt.show()
print reg.predict(Xtest)
