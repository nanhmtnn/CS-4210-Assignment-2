#-------------------------------------------------------------------------
# AUTHOR: Thu Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: Make probabilistic predictions
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []

#Reading the training data in a csv file
#--> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append(row)

X = []
Y = []

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity = {"High": 1, "Normal": 2}
wind = {"Weak": 1, "Strong": 2}

for row in db:
    X.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]])


#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
label = {"Yes": 1, "No": 2}
for row in db:
    Y.append(label[row[-1]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []

with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
        if i > 0: #skipping the header
            test_data.append(row)

#Printing the header os the solution
#--> add your Python code here
print(f"{'Day':<5}  {'Outlook':<8}  {'Temperature':<11}  {'Humidity':<8}  {'Wind':<6}  {'PlayTennis':<10}  {'Confidence'}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in test_data:
    test_instance = [[outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]]]
    prediction = clf.predict_proba(test_instance)[0]

    if prediction[0] > prediction[1]:
        confidence = prediction[0]
    else:   
        confidence = prediction[1]

    if confidence > 0.75:
        is_play = "Yes" if prediction[0] > prediction[1] else "No"
        print(f"{row[0]:<5}  {row[1]:<8}  {row[2]:<11}  {row[3]:<8}  {row[4]:<6}  {is_play:<10}  {confidence:.2f}")


    





