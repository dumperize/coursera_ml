import pandas as pd
import numpy as np
from scipy.sparse.construct import random
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.utils import shuffle

colnames=[
    'CRIM',
    'ZN', 'INDUS', 'CHAS', 
    'NOX', 'RM', 
    'AGE', 'DIS', 'RAD', 
    'TAX', 'PTRATIO', 'B',
    'LSTAT', 'MEDV'] 

# data = pd.read_csv('housing.data', names=colnames, delimiter='\t')
boston = load_boston()
scale_data = scale(boston.data)

# инициализация KFold с 5тью фолдами
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ran = np.linspace(1.0, 10.0, num=200)

k_scores = {}
for k in ran:
    knn = KNeighborsRegressor(
        n_neighbors=5, 
        weights='distance', 
        metric='minkowski', 
        p=k)
    score = cross_val_score(knn, scale_data, boston.target, cv=kf, scoring='neg_mean_squared_error').mean()
    k_scores[k] = score

print(k_scores)
print('Length of list', len(k_scores))
print('Max of list',  max(zip(k_scores.values(), k_scores.keys())))    
