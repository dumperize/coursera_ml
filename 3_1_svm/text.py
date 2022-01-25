import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
# (инструкция приведена выше). Обратите внимание, что загрузка данных может занять несколько минут
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
# Вычислите TF-IDF-признаки для всех текстов.
# Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным.
# При таком подходе получается, что признаки на обучающем множестве используют информацию из
# тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения
# целевой переменной из теста. На практике нередко встречаются ситуации, когда признаки
# объектов тестовой выборки известны на момент обучения, и поэтому можно ими пользоваться
# при обучении алгоритма.
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5]
# для SVM с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам.
# Укажите параметр random_state=241 и для SVM, и для KFold.
# В качестве меры качества используйте долю верных ответов (accuracy).
C_grid = {'C': np.power(10.0, np.arange(-5, 6))}
kf = KFold(n_splits=5, shuffle=True, random_state=241)

model = SVC(kernel='linear', C=1, random_state=241)
gs = GridSearchCV(model, C_grid, scoring='accuracy', cv=kf)
gs.fit(X, y)


# Обучите SVM по всей выборке с лучшим параметром C, найденным
# на предыдущем шаге.
model = gs.best_estimator_
model.fit(X, y)

# Найдите 10 слов с наибольшим по модулю весом. Они являются
# ответом на это задание. Укажите их через запятую, в нижнем регистре, 
# в лексикографическом порядке.
feature_mapping = vectorizer.get_feature_names_out()

weights = np.absolute(model.coef_.toarray())
max_weights = sorted(zip(weights[0], feature_mapping))[-10:]
max_weights.sort(key=lambda x: x[1])

f = open('3-1-2.txt', 'w')
for w, c in max_weights[:-1]:
	f.write(c)
	f.write(',')
f.write(max_weights[-1][1])
f.close()