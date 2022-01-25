import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Загрузите данные из файла abalone.csv. Это датасет, в котором требуется
# предсказать возраст ракушки (число колец) по физическим измерениям.
data = pd.read_csv('abalone.csv', delimiter=',')

# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
data['Sex'] = data['Sex'].map(
    lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных — признаки.
X = data[data.columns[:-1]].to_numpy()
y = data[data.columns[-1]]

# Обучите случайный лес с различным числом деревьев: от 1 до 50 (не забудьте выставить "random_state=1" в конструкторе).
# Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам.
# Используйте параметры "random_state=1" и "shuffle=True" при создании генератора кросс-валидации sklearn.cross_validation.KFold.
# В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
kf = KFold(n_splits=5, shuffle=True, random_state=1)

flag = False
for n in range(1, 51):
    model = RandomForestRegressor(n_estimators=n, random_state=1)
    metrics_array = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics_array += [metrics.r2_score(y_test, y_pred)]
        mean_metrics = np.mean(metrics_array)
    print("n=", n, ' metrics=', mean_metrics)
    # Определите, при каком минимальном количестве деревьев случай-
    # ный лес показывает качество на кросс-валидации выше 0.52. Это
    # количество и будет ответом на задание.
    if round(mean_metrics, 2) > 0.52 and not flag:
        print('n = ', n)
        f = open('5-1-1.txt', 'w')
        f.write(str(n))
        f.close()
        flag = True

