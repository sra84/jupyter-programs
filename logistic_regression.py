from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])
X=iris["data"][:,3:]
y=(iris["target"] ==2).astype(np.int)
# print(y)
# print(X)

#train a logistic regresssion classifier
clf = LogisticRegression()
clf.fit(X,y)
example= clf.predict(([[2.7]]))
print(example)

#using matplotlib to plot the visualiztion
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
plt.plot(X_new,y_prob[:,1], "g-", label = "virginica")
plt.show()