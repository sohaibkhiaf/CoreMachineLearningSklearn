from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


####################################### decision tree ##########################
# load data
df = pd.read_csv('500hits.csv', encoding='latin-1')

# data cleaning
df = df.drop(columns= ['PLAYER', 'CS'])
print(df.to_string())

# data| labels
X= df.iloc[:, 0:13]
y= df.iloc[:, 13]

print("X: ")
print(X)
print("y: ")
print(y)

# train| test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# create model (entropy: use only the useful features in the decision tree,
# ccp_alpha: keep tree simple to prevent from overfitting )
dtc = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.04)
print(dtc.get_params())

# train
dtc.fit(X_train, y_train)

# test+ evaluate
y_pred = dtc.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(dtc.feature_importances_)
print(X.columns)

features = pd.DataFrame(dtc.feature_importances_, X.columns)
print(features.to_string())

