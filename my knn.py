# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:46:28 2019
@author: 201448494
"""

from sklearn import datasets
import matplotlib.pyplot as plt 
import numpy as  np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import time

digits = datasets.load_digits()
x = digits.data
y = digits.target
x,y = shuffle(x,y)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42, stratify=y)
InputNum=int
InputNo=int
#the KNNClassfier method for my knn algorithm
class KNNClassfier(object):
    def fit(self,X, Y):     
        self.x = X
        self.y = Y
    def predict(self,X_test):
        output = np.zeros((X_test.shape[0],1))
        for i in range(X_test.shape[0]):
            dis = [] 
            for j in range(self.x.shape[0]):
                if self.distance == 'euc': #Euclidean Distance
                    dis.append(np.linalg.norm(X_test[i]-self.x[j,:]))
            labels = []
            index=sorted(range(len(dis)), key=dis.__getitem__)
            for j in range(self.k):
                labels.append(self.y[index[j]])
            counts = []
            for label in labels:
                counts.append(labels.count(label))
            output[i] = labels[np.argmax(counts)]
        return output
    def score(self,x,y): #calculate the score(accuracy)
        pred = self.predict(x)
        err = 0.0
        for i in range(x.shape[0]):
            if pred[i]!=y[i]:
                err = err+1
        return 1-float(err/x.shape[0])
    def __init__(self, k=5, distance='euc'):
        self.k = k
        self.distance = distance
        self.x = None
        self.y = None
#F3: train the digits dataset with my knn algorithm
def trainMyKNN():
    print('Please wait no more than 40 seconds for my KNN to train the model')
    myknn_start_time = time.time()
    clf = KNNClassfier(k=3)
    clf.fit(X_train,y_train)
    print('\nOutput the training accuracy and testing accuracy of sklearn knn model \nWhen k=3:')
    print('myknn training accuracy:',clf.score(X_train,y_train))
    print('myknn testing accuracy:',clf.score(X_test, y_test))  
    errorNum=(1-clf.score(X_test, y_test))*360
    print('The error number: ', int(errorNum))
    myknn_end_time = time.time()
    print('myknn uses time:',myknn_end_time-myknn_start_time)
    # open the file
    file = open("MyKnnModel.pickle", "wb")
    # write the model into my file
    pickle.dump(KNeighborsClassifier, file)
    # close the file
    file.close()
    UI()

#F2: train dataset with the sklearn knn algorithm
def trainSklearnKNN():
    knn = KNeighborsClassifier(n_neighbors=3)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    # Print the accuracy
    print('When k=3')
    print('The training accuracy:', knn.score(X_train,y_train))
    print('The testing accuracy:', knn.score(X_test, y_test))
    errorNum=(1-knn.score(X_test, y_test))*360
    print('The error number: ', int(errorNum))
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    # Loop over different values of k
    for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
        knn.fit(X_train, y_train)    
    #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)  
    print('You may need to close the picture window to continue.')
    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    file = open("SKlearnKnnModel.pickle", "wb")
    pickle.dump(KNeighborsClassifier, file)
    file.close()
    UI()    
    
#F1: show the information of the digits dataset   
def ShowInfo():
    digits.keys()
    n_samples,n_features=digits.data.shape
    X_train_samples, X_train_features=X_train.data.shape
    print('Load and display the dataset information:\n')
    print('The number of total data entries:', n_samples)
    print('The number of training data entries: ', X_train_samples)
    X_test_samples=n_samples-X_train_samples
    print('The number of testing data entries: ', X_test_samples)
    print('The number of data classes:', digits.target_names.size)
    print('')
    count=0   
    for i in range(0,10):
        for x in digits.target :
            if x==digits.target[i]:
                count=count+1
        print('The number of data entries for class', i, ':',count)
        count=0
    X=digits.data
    maximum_list = []
    minimum_list = []
    i=1
    for c in range(64):
        maximun = X[0][c]
        minimum = X[0][c]
        for r in range(1, 1797):
            if X[r][c] > maximun:
                maximun = X[r][c]
            if X[r][c] < minimum:
                minimum = X[r][c]
        maximum_list.append(maximun)
        minimum_list.append(minimum)
    print('\nThe maximum value for feature 1 to 64: ')
    for n in maximum_list:
        print(n, end=' ')
    print('\nThe minimum value for feature 1 to 64: ')
    for n in minimum_list:
        print(n, end=' ')   
    UI()
        
#F4: load SKlearn knn model from the saved file and print the train and test accuracy
def LoadSKLearnModel():
    print('\n\nPlease wait for around 5 seconds to load the sklearn knn model')
    file = open("SKlearnKnnModel.pickle", "rb")
    # read the model from the file
    knn = pickle.load(file)
    #close the file
    file.close()
    knn = KNeighborsClassifier(n_neighbors=3)
    #Fit the classifier to the training data
    knn.fit(X_train,y_train)
    print('\nOutput the training accuracy and testing accuracy of sklearn knn model \nWhen k=3:')
    print('The training accuracy is:', knn.score(X_train,y_train))
    print('The testing accuracy is:', knn.score(X_test, y_test))
    errorNum=(1-knn.score(X_test, y_test))*360
    print('The error number: ', int(errorNum))
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
    print('You may need to close the picture window to continue.')
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()

#F4: load my knn model from the saved file and print the train and test accuracy
def LoadMyKNNModel():
    print('\n\nPlease wait for around 30 seconds to load my knn model')
    print('(The graph showing the accuracy of this model when k ranges from 1 to 8 was put in the document because of the long consumed time.)')
    myknn_start_time = time.time()
    file = open("MyKnnModel.pickle", "rb")
    clf = pickle.load(file)
    file.close()
    clf = KNNClassfier(k=5)
    clf.fit(X_train,y_train)
    print('\n\nOutput the training and testing accuracy of my knn model \nWhen k=3:')
    print('myknn training score:',clf.score(X_train,y_train))
    print('myknn testing score:',clf.score(X_test, y_test))    
    errorNum=(1-clf.score(X_test, y_test))*360
    print('The error number: ', int(errorNum))
    myknn_end_time = time.time()
    print('load myknn uses time:',myknn_end_time-myknn_start_time)      
#F5: query my knn model
def QueryMyknnModel():
    print('\n\nQuery my knn model:')
    file = open("MyKnnModel.pickle", "rb")
    clf = pickle.load(file)
    file.close()
    clf = KNNClassfier(k=5)
    clf.fit(X_train,y_train) 
    print('Please input the index of test data entity from 1 to 360:')
    while  True :
        UserInputIndex = input()
        try:
            if isinstance(eval(UserInputIndex) ,(int))==True and 361>eval(UserInputIndex) >0:
                InputNo=int(UserInputIndex)
                break
            else:
                print("Invalid input with a wrong number. Please input again with the number 1-360.")
        except:
            print("Invalid input with digits or other input. Please input again with the number 1-360.")
    i=InputNo+1436
    print('You may need to close the picture window to continue.')
    X= digits.data    
    plt.gray()
    plt.matshow(digits.images[i])
    plt.show()
    print('The target of digit is: ', digits.target[i])
    print('The digit is predicted as: ', clf.predict(X[i:i+1]))  
    if digits.target[i]==clf.predict(X[i:i+1]):
        print('The prediction succeed.')    
#f5: query sklearn knn model
def QuerySklearnknnModel():
    print('\n\nQuery the sklearn knn model:')
    file = open("SKlearnKnnModel.pickle", "rb")
    # read the model from the file
    knn = pickle.load(file)
    #close the file
    file.close()
    knn = KNeighborsClassifier(n_neighbors=3)
    #Fit the classifier to the training data
    knn.fit(X_train,y_train)
    print('Please input the index of test data entity from 1 to 360:')
    while  True :
        UserInputIndex = input()
        try:
            if isinstance(eval(UserInputIndex) ,(int))==True and 361>eval(UserInputIndex) >0:
                InputNo=int(UserInputIndex)
                break
            else:
                print("Invalid input with a wrong number. Please input again with the number 1-360.")
        except:
            print("Invalid input with digits or other input. Please input again with the number 1-360.")
    i=InputNo+1436
    print('You may need to close the picture window to continue.')
    X= digits.data    
    plt.gray()
    plt.matshow(digits.images[i])
    plt.show()
    print('The target of digit is: ', digits.target[i])
    print('The number is predicted as: ', knn.predict(X[i:i+1])) 
    if digits.target[i]==knn.predict(X[i:i+1]):
        print('The prediction succeed.')
    
#userinterface
def UI():            
    print('\n\nPlease choose one option from the following (Enter number from 1 to 6):\n\
      1. F1: Show information of optical hand written digits dataset\n\
      2. F2: Train the digits dataset and save the model by KNeighborsClassifier of scikit-learn\n\
      3. F3: Train the digits dataset and save the model by my KNN algorithm.\n\
      4. F4: Output the train and test error for both models.\n\
      5. F5: Query both models.\n\
      6. End the program.')
    while  True :   #input validation
        UserInput = input()
        try:
            if isinstance(eval(UserInput) ,(int))==True and 7>eval(UserInput) >0:
                InputNum=int(UserInput)
                break
            else:
                print("Invalid input with a wrong number. Please input again.")
        except:
            print("Invalid input with digits or other input. Please input again.")
    if InputNum==1:
        ShowInfo()
    elif InputNum==2:
        trainSklearnKNN()        
    elif InputNum==3:
        trainMyKNN()
    elif InputNum==4:
        LoadSKLearnModel() 
        LoadMyKNNModel()        
        UI()
    elif InputNum==5:
        QueryMyknnModel()
        QuerySklearnknnModel()
        UI()
    else:
        print('The end.')
UI()
