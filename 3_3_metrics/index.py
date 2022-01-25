import pandas as pd

from sklearn import metrics


data = pd.read_csv('_8b9c6d9ae39e206610c6fd96894615a5_classification.csv', delimiter=',')

y_true = data['true']
y_pred = data['pred']

TP = data.loc[(data['true'] == 1) & (data['pred'] == 1)].shape[0]
FP = data.loc[(data['true'] == 0) & (data['pred'] == 1)].shape[0]

FN = data.loc[(data['true'] == 1) & (data['pred'] == 0)].shape[0]
TN = data.loc[(data['true'] == 0) & (data['pred'] == 0)].shape[0]

print(TP)
print(FP)
print(FN)
print(TN)

accuracy = metrics.accuracy_score(y_true, y_pred)
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

print('accuracy', accuracy)
print('precision', precision)
print('recall', recall)
print('f1', f1)