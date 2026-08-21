import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale 
import pandas as pd

################## kmeans ################

bc = load_breast_cancer()
print(bc)

x= scale(bc.data)
print(x)

y = bc.target 

X_Train, X_test, y_train, y_test = train_test_split(x, y, test_size=6)
model= KMeans(n_clusters=2, random_state=0)

model.fit(X_Train)

predictions = model.predict(X_test)

labels= model.labels_

print("labels= ", labels)
print("predictions= ", predictions)
print("accuracy= ", accuracy_score(y_test, predictions))
print("actual= ", y_test)


print(pd.crosstab(y_train, labels))

