import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

# print(data.head())

X= data[["buying", "maint", "safety" ]].values
y= data["class"]

# print(X)
# print(y)


# convert features to numbers
# X:
le= LabelEncoder()
for i in range(len(X[0])):
    X[:, i]= le.fit_transform(X[:, i])

# print(X)

# y:

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y = y.map(label_mapping)

y= np.array(y)
# print(y)


# create the model
knn = neighbors.KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)

print("predictions: ", predictions)
print("accuracy: ", accuracy)

# test

test_index = 67
print("actual values: ", y_test[test_index])
print("prediction:", knn.predict(X_test)[test_index])



