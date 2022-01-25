import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def create_file(filename, *content):
    file = open(filename, 'w')
    file.write(' '.join(map(str, content)))
    file.close()

# 1. Загрузите данные close_prices.csv. В этом файле приведены цены
# акций 30 компаний на закрытии торгов за каждый день периода.
data_prices = pd.read_csv('close_prices.csv', delimiter=',')
data_djia = pd.read_csv('djia_index.csv', delimiter=',')

# 2. На загруженных данных обучите преобразование PCA с числом
# компоненты равным 10. Скольких компонент хватит, чтобы объяс-
# нить 90% дисперсии?
# pca = PCA(n_components=10)
pca = PCA(n_components=0.9, svd_solver='full')
transformed_data = pca.fit_transform(data_prices[data_prices.columns[1:]])
# np.cumsum(pca.explained_variance_ratio_)
create_file('4-2-1.txt', len(pca.explained_variance_ratio_))

# 3. Примените построенное преобразование к исходным данным и возь-
# мите значения первой компоненты.
first_component = transformed_data[:, 0]

# 4. Загрузите информацию об индексе Доу-Джонса из файла djia_prices.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
djia_column = data_djia[data_djia.columns[1]]
korr = np.corrcoef(first_component, djia_column)
create_file('4-2-2.txt', korr[0, 1])

# 5. Какая компания имеет наибольший вес в первой компоненте?
company = data_prices[data_prices.columns[1:]].keys()[pca.components_[0].argmax()]
create_file('4-2-3.txt', company)
