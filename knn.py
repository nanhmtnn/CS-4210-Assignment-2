#-------------------------------------------------------------------------
# AUTHOR: Thu Nguyen
# FILENAME: knn.py
# SPECIFICATION: Find LOO_CV error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

# Init error counter
error_counter = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    X = []
    Y = []

    for j, instance in enumerate(db):
        # For other instances except for the testing one
        if j != i:
            instance_features = list(map(float, instance[:-1]))
            if instance[-1] == 'spam':
                instance_class = 1
            else:
                instance_class = 0
            X.append(instance_features)
            Y.append(instance_class)

        #Store the test sample of this iteration in the vector testSample
        #--> add your Python code her

        # For testing instance
        testSample = []
        for val in db[i][:-1]:
            testSample.append(float(val))

        if db[i][-1] == 'spam':
            true_label = 1
        else:
            true_label = 0

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    if class_predicted != float(true_label):
        error_counter += 1

#Print the error rate
#--> add your Python code here

error_rate = error_counter / len(db)
print(f"LOO_CV error rate: {error_rate:.2f}")








