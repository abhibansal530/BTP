import sys
import pandas as pd 
import numpy as np 
import unicodedata
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from numpy import linalg as LA
from sklearn import decomposition
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#Read data from Excel sheet and do pre-processing
Df = pd.read_excel('./Tunga_Bhadra River Basin Rainfall Data (1).xls')
lat_long = Df.columns
coordinates = []
LAT, LONG = set(), set()
NUMSITES = None
GRIDSIZE = (None, None)

def parseCoordinates(x):
    """
    Parse given coordinates and
    return lat, long
    """
    lat = x.split('N')[0].strip()
    longi = x.split('N')[1].strip().split('E')[0].strip()
    return float(lat), float(longi)

def getPositionAsString(x, y):
    """
    Given lat, long
    convert back to original string
    """
    return str(x) + 'N' + str(y) + 'E'

def preProcess():
    """
    Do all required one time preprocessing
    """
    global coordinates
    global LAT
    global LONG
    global NUMSITES
    global GRIDSIZE
    for i in range(1,len(lat_long)):
        coordinates.append(unicodedata.normalize('NFKD', lat_long[i]).encode('ascii','ignore'))
    for c in coordinates:
        x, y = parseCoordinates(c)
        LAT.add(x)
        LONG.add(y)
    LAT = list(LAT)
    LONG = list(LONG)
    NUMSITES = len(coordinates)
    GRIDSIZE = (len(LAT), len(LONG))

def LinearRegressionHelper(siteToPredict, Neighbours, numTrain, numTest):
    X = pd.DataFrame()
    for n in Neighbours:
        X[n] = Df[n]
    Xt = X.head(numTrain)
    Yt = Df[siteToPredict].head(numTrain)
    Xt = np.array(Xt)
    Yt = np.array(Yt)
    #Remove 0 data values from training
    NonZeros = [i for i in range(0, len(Yt)) if Yt[i] != 0]
    Xtrain, Ytrain = Xt[NonZeros], Yt[NonZeros]
    #Train linear regression
    Reg = linear_model.LinearRegression()
    Reg.fit(Xtrain, Ytrain)
    #Test accuracy
    Xtest = X.tail(numTest)
    Y_actual = Df[siteToPredict].tail(numTest)
    Y_predicted = Reg.predict(Xtest)
    return Y_actual, Y_predicted

def simpleLinearRegression(latitude, longitude, neightbourThresh, numTrain, numTest):
    """
    Perform simple Linear Regression for
    given site considering neighbours within
    given threshold. Return actual, predicted
    """
    #Get neighbours
    Neighbours = []
    siteToPredict = None;
    for c in coordinates:
        x, y = parseCoordinates(c)
        if x == latitude and y == longitude:
            siteToPredict = c
            continue;
        if abs(x - latitude) <= neightbourThresh and abs(y - longitude) <= neightbourThresh:
            Neighbours.append(c)
    #Do linear regression
    return LinearRegressionHelper(siteToPredict, Neighbours, numTrain, numTest)

def plotScatter(X, Y, xlabel, ylabel):
    """
    Plot scatter plot b/w X and Y
    """
    plt.scatter(X, Y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def getAdjacencyAndDegreeMatrix(thresh):
    """
    Return an adjacency and degree matrix
    using correlation values above thresh
    """
    #Correlation Matrix
    Correlation = np.zeros((NUMSITES, NUMSITES))
    Corr = Df.loc[:, Df.columns[1]:].corr()
    for i in range(1, len(lat_long)):
        for j in range(1, len(lat_long)):
            Correlation[i - 1][j - 1] = Corr[lat_long[i]][lat_long[j]]
    #Adjacency Matrix
    A = [[0 if x <= thresh else x for x in row] for row in Correlation]
    for i in range(0, len(A)):
        A[i][i] = 0
    #Degree Matrix
    D = np.zeros((NUMSITES, NUMSITES))
    for i in range(0, len(D)):
        D[i][i] = sum(A[i])
    return A, D

def getLaplacian(thresh):
    """
    Return Laplacian of the graph considering
    correaltion above thres as edge weight
    """
    A, D = getAdjacencyAndDegreeMatrix(thresh)
    return D - A

def numCC(adj):
    """
    Given adjacency matrix of graph
    return no. of connected componenets
    """
    return csgraph.connected_components(A, directed=False)[0]

def eigenDecomp(M):
    """
    Return eigen decomposition of matrix M
    """
    return LA.eig(M)

def visulaizeEigen(v, num):
    """
    Given set of eigen values v
    visulaize num'th of them
    """
    Intensity = np.zeros(GRIDSIZE)
    for c, val in zip(coordinates, v[:,num]):
        lat ,lon = parseCoordinates(c)
        l, r = LAT.index(lat), LONG.index(lon)
        Intensity[l][r] = val
    #Remove negative values by scaling
    Min = np.min(Intensity)
    for i in range(0, GRIDSIZE[0]):
        for j in range(0, GRIDSIZE[1]):
            if (Intensity[i][j] != 0):
                Intensity[i][j] = (Intensity[i][j] + abs(Min))*100 + 5.0
    #Plot intensity
    plt.imshow(Intensity)
    plt.colorbar()
    plt.show()
    #Plot eigen values
    values = list(v[:,num])
    values.sort()
    plt.plot(values)
    plt.show()

def visulaizeEigen3D(v1, v2, v3):
    """
    Use same visualization with 3 eigen vectors
    """
    X = (v1 - np.min(v1))/(np.max(v1) - np.min(v1))
    Y = (v2 - np.min(v2))/(np.max(v2) - np.min(v2))
    Z = (v3 - np.min(v3))/(np.max(v3) - np.min(v3))
    threeD = plt.figure().gca(projection='3d')
    threeD.scatter(X, Y, Z)
    plt.show()

def Clustering(v1, v2, v3, numClusters, visualize=True):
    """
    Use given eigen vectors to form
    clusters. Plot them and return labels
    """
    X = np.zeros((NUMSITES, 3))
    X = zip(v1, v2, v3)
    X = np.array(X)
    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(X)
    Labels = kmeans.labels_
    if visualize:
        threeD = plt.figure().gca(projection='3d')
        threeD.scatter(X[:,0], X[:,1], X[:,2], c=Labels.astype(np.float), edgecolor='k')
        plt.show()
    return Labels

def LinearRegressionOnCluster(labels, latitude, longitude, numTrain, numTest):
    """
    Instead of spacial neighbours
    use cluster for linear regression
    """
    #Get label of required site
    assert(len(coordinates) == len(labels))
    lab, Pos = None, None
    siteToPredict = None
    for i in range(0, len(coordinates)):
        x, y = parseCoordinates(coordinates[i])
        if x == latitude and y == longitude:
            Pos = i
            siteToPredict = coordinates[i]
            break
    lab = labels[Pos]
    if Pos == None or lab == None or siteToPredict == None:
        print "Unable to get label"
        sys.exit(-1)
    #Get neighbours
    Neighbours = []
    for i in range(0, len(labels)):
        if i != Pos and labels[i] == lab:
            Neighbours.append(coordinates[i])
    #Do linear regression
    return LinearRegressionHelper(siteToPredict, Neighbours, numTrain, numTest)

def Compare(labels, thresh):
    """
    Compare clustering based regression
    and spatial neighbours regression
    """
    spatial = []
    clustering = []
    test, train = 4000, 16000
    for site in coordinates:
        x, y = parseCoordinates(site)
        #Clustering
        actual, predicted = LinearRegressionOnCluster(labels, x, y, train, test)
        clustering.append(r2_score(actual, predicted))
        #Spatial
        actual, predicted = simpleLinearRegression(x, y, thresh, train, test)
        spatial.append(r2_score(actual, predicted))
    print "Mean of spatial regression ", np.mean(spatial)
    print "Mean of cluster regression ", np.mean(clustering)
    #Plot r2 scores
    plt.scatter(range(1, NUMSITES+1), spatial, marker='o', label='spatial')
    plt.scatter(range(1, NUMSITES+1), clustering, marker='x', label='clustering')
    plt.legend()
    plt.show()



preProcess()
print GRIDSIZE
