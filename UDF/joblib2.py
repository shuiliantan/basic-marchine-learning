from sklearn import tree

from sklearn import datasets
from sklearn.externals import joblib


clf = tree.DecisionTreeClassifier(max_features=0.1)
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)


joblib_file = "joblib_model.pkl"
joblib.dump(clf, joblib_file)

# Load from file
joblib_model = joblib.load(joblib_file)

# Calculate the accuracy and predictions
score = joblib_model.score(X, y)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = joblib_model.predict(X)





