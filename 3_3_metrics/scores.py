import pandas as pd
import numpy as np

from sklearn import metrics


data = pd.read_csv(
    '_eee1b9e8188f61bc35d954fbeb94e325_scores.csv', delimiter=',')

# true,score_logreg,score_svm,score_knn,score_tree

def f(data, name): 
    precision, recall, thresholds = metrics.precision_recall_curve(data['true'], data[name])
    gen = (x for x in recall if x > 0.7)
    return max(list(map(lambda i: precision[i[0]], enumerate(gen))))

logreg = metrics.roc_auc_score(data['true'], data['score_logreg'])
svm = metrics.roc_auc_score(data['true'], data['score_svm'])
knn = metrics.roc_auc_score(data['true'], data['score_knn'])
tree = metrics.roc_auc_score(data['true'], data['score_tree'])

# print('logreg', logreg)
# print('svm', svm)
# print('knn', knn)
# print('tree', tree)

print('logreg', f(data, 'score_logreg'))
print('svm', f(data, 'score_svm'))
print('knn', f(data, 'score_knn'))
print('tree', f(data, 'score_tree'))

# print(type(precision))

# print(m)
