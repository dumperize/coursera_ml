import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Загрузите данные из файла data-logistic.csv. Это двумерная выборка, 
# целевая переменная на которой принимает значения -1 или 1.
colnames = [
    'Goal',
    'First', 'Second']

data = pd.read_csv('data-logistic.csv', names=colnames, delimiter=',')

y = data['Goal'].to_numpy()
X = data[data.columns[1:]].to_numpy()

w = np.array([0, 0])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Реализуйте градиентный спуск для обычной и L2-регуляризованной
# (с коэффициентом регуляризации 10) логистической регрессии. Ис-
# пользуйте длину шага k=0.1. В качестве начального приближения
# используйте вектор (0, 0).
def log_regression(X, y, k, w, C, epsilon, max_iter):
    for i in range(max_iter):
        wnew = w + k * np.mean((X.transpose() * y) *
                               (1 - sigmoid(y * np.dot(X, w))), axis=1) - k * C * w
        if np.sqrt(np.sum(np.square(wnew-w))) < epsilon:
            break
        w = wnew

    predictions = sigmoid(np.dot(X, w))
    return predictions

# Запустите градиентный спуск и доведите до сходимости (евклидово
# расстояние между векторами весов на соседних итерациях долж-
# но быть не больше 1e-5). Рекомендуется ограничить сверху число
# итераций десятью тысячами.
y_with_regul = log_regression(X, y, 0.1, w, 10, 0.00001, 10000)
y_without_regul = log_regression(X, y, 0.1, w, 0, 0.00001, 10000)

# Какое значение принимает AUC-ROC на обучении без регуляри-
# зации и при ее использовании?
auc_score_with_regul = roc_auc_score(y, y_with_regul)
auc_score_without_regul = roc_auc_score(y, y_without_regul)

f = open('3-2.txt', 'w')
f.write(str(auc_score_with_regul) + ' ' + str(auc_score_without_regul))
f.close()