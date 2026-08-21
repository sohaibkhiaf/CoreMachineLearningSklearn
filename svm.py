from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

############################## Svm ##############################
iris = datasets.load_iris()

X=  iris.data
y= iris.target

print(X.shape)
print(y.shape)

classes = ["Iris Setosa", "Iris Versicolor", "Iris Verginica"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("predictions= ", predictions)
print("actual= ", y_test)
print("accuracy= ", acc)

for i in range (len(predictions)):
    print(classes[predictions[i]])

