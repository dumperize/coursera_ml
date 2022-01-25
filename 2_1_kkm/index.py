import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

colnames=[
    'Type',
    'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'] 

data = pd.read_csv('_805605c804bae8c5c24785f433b230ce_wine.data', names=colnames, delimiter=',')
X = scale(data.loc[:, 'Alcohol':])
y = data['Type']

# инициализация KFold с 5тью фолдами
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# knn.fit(X, y)
# empty list to store scores
k_scores = {}
k_range = range(1, 51)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()
    k_scores[k] = score

print('Length of list', len(k_scores))
print('Max of list',  max(zip(k_scores.values(), k_scores.keys())))