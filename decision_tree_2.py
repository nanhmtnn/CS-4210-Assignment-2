#-------------------------------------------------------------------------
# AUTHOR: Thu Nguyen
# FILENAME: decision_tree_2.py
# SPECIFICATION: Find the average accuracy of this model during the 10 runs
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    age = {'Young': 1, 'Presbyopic': 2, 'Prepresbyopic': 3}
    spectacle = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism = {'Yes': 1, 'No': 2}
    tear = {'Reduced': 1, 'Normal': 2}

    for row in dbTraining:
        X.append([age[row[0]], spectacle[row[1]], astigmatism[row[2]], tear[row[3]]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    labels = {'Yes': 1, 'No': 2}

    for row in dbTraining:
        Y.append(labels[row[4]])

    # Accuracy list to store the accuracy of each run
    accuracy_list = []

    #Loop your training and test tasks 10 times here
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here

        dbTest = []

        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

        # Init variables
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        
        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here

            instance_features = [age[data[0]], spectacle[data[1]], astigmatism[data[2]], tear[data[3]]]
            class_predicted = clf.predict([instance_features])[0]

            if data[4] == 'Yes':
                true_label = 1
            else:
                true_label = 2

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here

            if class_predicted == true_label:
                if true_label == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if true_label == 1:
                    false_negative += 1
                else:
                    false_positive += 1
            
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        accuracy_list.append(accuracy)


    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Final accuracy when training on {ds}: {avg_accuracy:.2f}")



