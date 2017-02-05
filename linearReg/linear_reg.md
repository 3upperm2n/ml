

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
```

    /home/leiming/anaconda2/lib/python2.7/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')



```python
def load_data(filename):
    cols=["N", "Time_in_us"]
    return pd.read_csv(filename, names=cols)

df = load_data('dev0_h2d_stepsize4.csv')

X = df["N"]
y  = df["Time_in_us"]
    
X = np.array(X)
y = np.array(y)

X = X.reshape(-1,1)
y = y.reshape(-1,1)
```


```python
lr_model = linear_model.LinearRegression()
lr_model.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print lr_model.coef_
print lr_model.intercept_
```

    [[ 0.00062165]]
    [ 1.8122791]



```python
print "Mean squared error: %.6f" % np.mean((lr_model.predict(X) - y) ** 2)
print('Variance score: %.6f' % lr_model.score(X, y))
```

    Mean squared error: 0.001865
    Variance score: 0.999422


### permutation


```python
np.random.seed(42)

sample = np.random.choice(df.index, size= int(len(df) * 0.9), replace=False)

data, test_data = df.ix[sample], df.drop(sample)
```


```python
X_train, y_train = data["N"], data["Time_in_us"]
X_test, y_test = test_data["N"], test_data["Time_in_us"]

X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)

X_test = X_train.reshape(-1,1)
y_test = y_train.reshape(-1,1)


lr_model.fit(X_train, y_train)

print lr_model.coef_
print lr_model.intercept_
```

    [[ 0.0006215]]
    [ 1.81329683]



```python
print "Mean squared error: %.6f" % np.mean((lr_model.predict(X_test) - y_test) ** 2)
print('Variance score: %.6f' % lr_model.score(X_test, y_test))
```

    Mean squared error: 0.001904
    Variance score: 0.999411

