import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Загрузите выборку из файла svm-data.csv. В нем записана двумерная выборка 
# (целевая переменная указана в первом столбце, признаки — во втором и третьем).
colnames=['Goal', 'First', 'Second'] 
data = pd.read_csv('svm-data.csv', names=colnames, delimiter=',')

X = data[data.columns[1:]]
y = data['Goal']

# Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241. 
# Такое значение параметра нужно использовать, чтобы убедиться, что SVM работает с выборкой как с линейно разделимой. 
# При более низких значениях параметра алгоритм будет настраиваться с учетом слагаемого в функционале, 
# штрафующего за маленькие отступы, из-за чего результат может не совпасть 
# с решением классической задачи SVM для линейно разделимой выборки.
model = SVC(kernel='linear', C=100000, random_state=241)
model.fit(X,y)
# Найдите номера объектов, которые являются опорными (нумерация с единицы). Они будут являться ответом на задание. 
# Обратите внимание, что в качестве ответа нужно привести номера объектов 
# в возрастающем порядке через запятую или пробел. Нумерация начинается с 1.
print(model.support_)

file = open('3-1-1.txt', 'w')
file.write(' '.join(map(str, model.support_)))
file.close()