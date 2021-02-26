from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = load_iris(return_X_y=True)
# X[0] = np.nan
X = [[0,1],[2,3]]
y=[0,1]

# 二分类
# y = np.where(y==2,1,y)
clf = LogisticRegression(random_state=0, solver='liblinear', C=1000000,penalty='l1', n_jobs=1,verbose=-1).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)