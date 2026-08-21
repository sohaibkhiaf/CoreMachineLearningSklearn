from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


####################### logistic regression #########################

# load
data = load_breast_cancer()

X= data.data
y= data.target

print("X:")
print(X)
print(X.shape)
print("y:")
print(y)
print(y.shape)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# create and train the model
model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

# predict
prediction = model.predict(X_test)

# accuracy
acc = accuracy_score(y_test, prediction)

print("prediction= ", prediction)
print("actual = ", y_test)
print("accuracy= ", acc)



