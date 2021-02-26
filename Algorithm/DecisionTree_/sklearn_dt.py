from sklearn import tree
import numpy as np
X = [[0, 0], [1, 1]]
Y = [0,1]

# Y = np.matrix([0,1])
# X[0][0] = np.nan
# Y[0] = np.nan

clf = tree.DecisionTreeClassifier(max_features=0.1)
clf = clf.fit(X, Y)
print(clf)
clf = clf.fit([[1,3]], [1])
print(clf)

print(clf.predict([[2., 2.]]))
print(clf.predict_proba([[2., 2.]]))


# from sklearn.datasets import load_iris
# from sklearn import tree
# X, y = load_iris(return_X_y=True)
# clf = tree.DecisionTreeClassifier()
#
# clf = clf.fit(X, y)
# clf.predict(X[:2, :])
# clf.predict_proba(X[:2, :])