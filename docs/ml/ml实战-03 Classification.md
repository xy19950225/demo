## MNIST

第一步就遇到问题。。。

版本问题

问题描述：

from sklearn.datasets import fetch_mldata

ImportError: cannot import name 'fetch_mldata' from 'sklearn.datasets' 

解决方法：

https://github.com/ageron/handson-ml/issues/529

from sklearn.datasets import fetch_openml

dataset = fetch_openml("mnist_784")


```python
# 加载数据
from sklearn.datasets import fetch_openml

mnist = fetch_openml("MNIST_784")
```


```python
# 查看keys
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'DESCR', 'details', 'categories', 'url'])




```python
# 创建数据集
X,y = mnist["data"], mnist["target"]
```


```python
# 查看数组形状 结果表示70000张图片（示例），784个特征，每个特征代表像素点的强度
X.shape
```




    (70000, 784)




```python
y.shape
```




    (70000,)




```python
import matplotlib
import matplotlib.pyplot as plt
```


```python
# 改变数组形状 28x28数组
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
```


```python
# 用matplotlib的imshow()函数绘图 (好烦啊，matplotlib参数真多，有没有替代的？)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()
```


![png](output_8_0.png)



```python
# 验证
y[0]
```




    '5'




```python
# 划分训练集和测试集
X_train, X_test, y_train, y_test = X[:60000], X[60000:],  y[:60000], y[60000:]
```


```python
# 随机采样
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```

## 训练一个二分类器

这里遇到一个问题

问题描述：

ValueError: The number of classes has to be greater than one; got 1 class

解决方法：

https://github.com/ageron/handson-ml/issues/360


```python
# 布尔型
y_train_5 = (y_train == "5")
y_test_5 = (y_test == "5" )
```


```python
# 随机梯度下降分类
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```




    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                  max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='l2',
                  power_t=0.5, random_state=42, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False)




```python
# 预测
sgd_clf.predict([X[8]])
```




    array([False])



## 效能考核

### 交叉验证


```python

```
