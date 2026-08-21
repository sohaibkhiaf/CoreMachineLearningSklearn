from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

################### principal component analysis for dim reduction ################

df = pd.read_csv('2022mlbteams.csv')

df.drop('Tm', axis=1, inplace=True)

print(df.to_string())

# all features except playoffs
X = df.iloc[:, :-1]
# labels (playoffs)
y= df.iloc[:, -1]

print("y: ")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 21
)

# scalar object (for normalization)
ss = StandardScaler()

X_train = ss.fit_transform(X_train)

print(df.columns)

X_train = pd.DataFrame(X_train, columns=['#Bat', 'BatAge', 'R/G', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR',
       'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'TB',
       'GDP', 'HBP', 'SH', 'SF', 'IBB'])

print(X_train.head(10))
print(X_train.describe().round(2))

###################### PCA 1 ######################
# because n_components is not specified, PCA calculates all possible principal components
pca1 = PCA()
# fit(): PCA studies X_train and finds the directions/components with the most variance.
# transform(): PCA converts your original data into the new principal-component representation.
X_pca1 = pca1.fit_transform(X_train)
# This prints the percentage/proportion of total variance explained by each principal component.
print(pca1.explained_variance_ratio_)

plt.bar(range(1,len(pca1.explained_variance_ )+1),pca1.explained_variance_ )
plt.ylabel('Explained variance')
plt.xlabel('Components')
plt.plot(range(1,len(pca1.explained_variance_ )+1),
         np.cumsum(pca1.explained_variance_),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='upper left')
plt.show()

plt.plot(pca1.explained_variance_ratio_)
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
plt.show()

##################### PCA 2 ######################
# This code uses PCA to automatically reduce the number of features
# while keeping 95% of the information (variance).
pca2 = PCA(0.95)
X_pca2 = pca2.fit_transform(X_train)
print(X_pca2.shape)
print(pca2.explained_variance_ratio_)

##################### PCA 2 c (2 components) ######################
# Create PCA with exactly 2 components
# PC1 captures the largest possible variance,
# while PC2 captures the next largest variance.
pca2c = PCA(n_components=2)
X_pca2c = pca2c.fit_transform(X_train)

colormap = plt.colormaps['coolwarm']

plt.figure()
scatter = plt.scatter(
    X_pca2c[:, 0],
    X_pca2c[:, 1],
    c=y_train,
    cmap=colormap
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Playoffs')
plt.show()


##################### PCA 3 c (3 components) ######################
# Create PCA with 3 components
# PC1 captures the most variance
# PC2 captures the second most variance
# PC3 captures the third most variance
pca3c = PCA(n_components=3)
X_pca3c = pca3c.fit_transform(X_train)
print(pca3c.explained_variance_ratio_)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca3c[:, 0],
    X_pca3c[:, 1],
    X_pca3c[:, 2],
    c=y_train,
    cmap='coolwarm'
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

fig.colorbar(scatter, ax=ax, label='Playoffs')

plt.show()


