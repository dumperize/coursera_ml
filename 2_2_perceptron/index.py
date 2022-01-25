import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


colnames=['Goal', 'First', 'Second'] 
train = pd.read_csv('_3abd237d917280ba0d83bfe6bd49776f_perceptron-train.csv', names=colnames, delimiter=',')
test = pd.read_csv('_3abd237d917280ba0d83bfe6bd49776f_perceptron-test.csv', names=colnames, delimiter=',')
scaler = StandardScaler()

# ytrain = train[train.columns[0]]
# Xtrain = train[train.columns[1:]]
# Xtrain_scaled = scaler.fit_transform(Xtrain)
ytrain = train[train.columns[0]]
Xtrain = train[train.columns[1:]]
Xtrain_scaled = scaler.fit_transform(Xtrain)

# ytest = test[test.columns[0]]
# Xtest = test[test.columns[1:]]
# Xtest_scaled = scaler.transform(Xtest)
ytest = test[test.columns[0]]
Xtest = test[test.columns[1:]]
Xtest_scaled = scaler.transform(Xtest)

clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(Xtrain, ytrain)

y_test_predict = clf.predict(Xtest)
accuracy = accuracy_score(ytest, y_test_predict)

clf.fit(Xtrain_scaled, ytrain)

y_test_predict_scaler = clf.predict(Xtest_scaled)
accuracy_scaled = accuracy_score(ytest, y_test_predict_scaler)

print(accuracy_scaled - accuracy)
