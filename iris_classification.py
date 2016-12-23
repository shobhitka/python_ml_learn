from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()

features = data['data']
feature_names = data['feature_names']
target = data['target']

for t, marker, c in zip(xrange(3), ">ox", "rgb"):
    plt.scatter(features[target == t,0],
                features[target == t,1],
                marker = marker,
                c = c)

plt.show()

# Second column of data
plength = features[:, 2]

is_setosa = (target == 0)

max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

print('Maximum of setosa: %1.1f' % max_setosa)
print('Minimum of others: %1.1f' % min_non_setosa)

# Simple model
for petal_length in features[:, 2]:
    if petal_length < 2:
        print 'Iris Setosa'
    else:
        print 'Iris Virginica or Iris Versicolour'
