import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

############ gradient boosting classifier ###################



# load data
wine = datasets.load_wine(as_frame=True)

# feature| target
X = wine.data
y = wine.target

print(X)
print(y)

# train | test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

# 3-fold cross validation
# gbc: passed to inform cross_val_score about the kind of model, not necessarily trained
scores = cross_val_score(gbc, X_train, y_train, cv=3, n_jobs=-1)
print(scores)
print(scores.mean())

# search for best params using GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 9]
}
# gbc: passed to inform cross_val_score about the kind of model, not necessarily trained
gsc = GridSearchCV(gbc, param_grid, cv=3, n_jobs=-1)
gsc.fit(X_train, y_train)

print(gsc.best_params_)
print(gsc.best_score_)


