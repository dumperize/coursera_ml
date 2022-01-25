import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack


# 1. Загрузите данные об описаниях вакансий и соответствующих годо-
# вых зарплатах из файла salary-train.csv.
data_train = pd.read_csv('salary.csv', delimiter=',')
data_test = pd.read_csv('salary-test-mini.csv', delimiter=',')

# 2. Проведите предобработку:
# • Приведите тексты к нижнему регистру.
# • Замените все, кроме букв и цифр, на пробелы — это облегчит
full_description = pd.Series(data_train['FullDescription']).str.lower(
).replace('[^a-zA-Z0-9]', ' ', regex=True)
full_description_test = pd.Series(data_test['FullDescription']).str.lower(
).replace('[^a-zA-Z0-9]', ' ', regex=True)

# Примените TfidfVectorizer для преобразования текстов в векторы
# признаков. Оставьте только те слова, которые встречаются хотя бы в 5 объектах.
# группируем по словам
vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(full_description)
X_test = vectorizer.transform(full_description_test)

# Замените пропуски в столбцах LocationNormalized и ContractTime
# на специальную строку ’nan’.
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования
# признаков LocationNormalized и ContractTime.
enc = DictVectorizer()
X_train_categ = enc.fit_transform(
    data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(
    data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки".
merged = hstack([X, X_train_categ])
merged_test = hstack([X_test, X_test_categ])

y = data_train['SalaryNormalized']

# X_test = data_test[data_test.columns[0:3]]
# y_test = data_test['SalaryNormalized']

# 3. Обучите гребневую регрессию с параметром alpha=1. Целевая переменная
# записана в столбце SalaryNormalized.
# linear regression
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(merged, y)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
predict = clf.predict(merged_test)


file = open('4-1.txt', 'w')
file.write(str(predict)[1:-1])
file.close()
