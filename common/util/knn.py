#########################################  
# kNN: k Nearest Neighbors  

# Input:      newInput: vector to compare to existing dataset (1xN)  
#             dataSet:  size m data set of known vectors (NxM)  
#             labels:   data set labels (1xM vector)  
#             k:        number of neighbors to use for comparison   

# Output:     the most popular class label  
#########################################  

from numpy import *


# create a dataset which contains 4 samples with 2 classes  
def create_dataset():
    # create a matrix: each row as a sample  
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, labels


# classify using kNN  
def knn_classify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    max_count = 0
    max_index = None
    for idx_key, value in classCount.items():
        if value > max_count:
            max_count = value
            max_index = idx_key

    return max_index
    # return sortedDistIndices
