from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

###################### random forest classification #########################

# load data
df = pd.read_csv('500hits.csv', encoding='latin-1')

print(df.head())

# cleaning
df =df.drop(columns=["PLAYER", "CS"])

X= df.iloc[:, 0:13]
y= df.iloc[:, 13]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 17)

# create model
rf = RandomForestClassifier(
    n_estimators=1000,      # number of decision trees in the forest
    criterion='entropy',    # criterion used to measure the quality of a split
    min_samples_split=10,   # minimum number of samples required to split a node
    max_depth=14,           # maximum depth allowed for each decision tree
    random_state=42         # seed for reproducible random results
)

# train
rf.fit(X_train, y_train)

# prediction
y_pred = rf.predict(X_test)


# evaluation
print("rf.score= ", rf.score(X_test, y_test))

print("classification_report= ", classification_report(y_test, y_pred))

features = pd.DataFrame(rf.feature_importances_, index= X.columns)
print(features.head(15))


