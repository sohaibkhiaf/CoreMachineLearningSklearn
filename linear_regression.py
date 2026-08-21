from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


####################### linear regression #######################


housing = datasets.fetch_california_housing()

print(housing.feature_names)

X = housing.data
y = housing.target

print("X")
print(X)
print(X.shape)
print("y")
print(y)
print(y.shape)

l_reg = linear_model.LinearRegression()

plt.scatter(X[:, 2], y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = l_reg.fit(X_train, y_train)

predictions = l_reg.predict(X_test)

print("predictions= ", predictions)
print("R2 value= ", l_reg.score(X_test, y_test))
print("coef= ", l_reg.coef_)
print("intercept= ", l_reg.intercept_)

