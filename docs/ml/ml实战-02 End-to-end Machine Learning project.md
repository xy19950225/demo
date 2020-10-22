## 工作流程

## 获取数据

!!! tip "下载数据"


```python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```


```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

!!! tip "查看数据结构"


```python
housing = load_housing_data()
```


```python
# 查看前5行
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查看数据描述
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    


```python
# 查看数据分类
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
# 查看属性摘要
housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 绘制所有属性直方图
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))
plt.show()
```


![png](output_11_0.png)


!!! tip "创建测试集"

问题描述：

再次运行程序，产生一个不同的测试集

解决方法：

1.设置随机种子 np.random.seed(42)

2.每个实例都使用一个标识符 (这个不太懂，暂时跳过)


```python
# 用python定义一个split_train_test函数
import numpy as np

def split_train_test(data, test_radio):
    seed = np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_radio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```


```python
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +" ,len(test_set), "test")
```

    16512 train + 4128 test
    


```python
train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14196</th>
      <td>-117.03</td>
      <td>32.71</td>
      <td>33.0</td>
      <td>3126.0</td>
      <td>627.0</td>
      <td>2300.0</td>
      <td>623.0</td>
      <td>3.2596</td>
      <td>103000.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>8267</th>
      <td>-118.16</td>
      <td>33.77</td>
      <td>49.0</td>
      <td>3382.0</td>
      <td>787.0</td>
      <td>1314.0</td>
      <td>756.0</td>
      <td>3.8125</td>
      <td>382100.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>17445</th>
      <td>-120.48</td>
      <td>34.66</td>
      <td>4.0</td>
      <td>1897.0</td>
      <td>331.0</td>
      <td>915.0</td>
      <td>336.0</td>
      <td>4.1563</td>
      <td>172600.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>14265</th>
      <td>-117.11</td>
      <td>32.69</td>
      <td>36.0</td>
      <td>1421.0</td>
      <td>367.0</td>
      <td>1418.0</td>
      <td>355.0</td>
      <td>1.9425</td>
      <td>93400.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2271</th>
      <td>-119.80</td>
      <td>36.78</td>
      <td>43.0</td>
      <td>2382.0</td>
      <td>431.0</td>
      <td>874.0</td>
      <td>380.0</td>
      <td>3.5542</td>
      <td>96500.0</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 还是sklearn省事
from sklearn.model_selection import train_test_split
```


```python
# 划分训练集和测试集 train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```


```python
# 查看收入中位数
housing["median_income"]
```




    0        8.3252
    1        8.3014
    2        7.2574
    3        5.6431
    4        3.8462
              ...  
    20635    1.5603
    20636    2.5568
    20637    1.7000
    20638    1.8672
    20639    2.3886
    Name: median_income, Length: 20640, dtype: float64




```python
# 
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
```


```python
housing["income_cat"]
```




    0        6.0
    1        6.0
    2        5.0
    3        4.0
    4        3.0
            ... 
    20635    2.0
    20636    2.0
    20637    2.0
    20638    2.0
    20639    2.0
    Name: income_cat, Length: 20640, dtype: float64




```python
# 数组过滤数据 回过头来再看《利用python进行数据分析》中将条件逻辑表述为数组运算这一节内容
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
```


```python
housing["income_cat"]
```




    0        5.0
    1        5.0
    2        5.0
    3        4.0
    4        3.0
            ... 
    20635    2.0
    20636    2.0
    20637    2.0
    20638    2.0
    20639    2.0
    Name: income_cat, Length: 20640, dtype: float64



!!! tip "分层抽样"

方法：StratifiedShuffleSplit

官方示例：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html?highlight=stratifiedshufflesplit


```python
# 分层抽样 StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```


```python
# 划分训练集和测试集
for train_index, test_index in split.split(housing, housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]
```


```python
# 验证分层抽样是否正确
start_test_set["income_cat"].value_counts() / len(start_test_set)
```




    3.0    0.350533
    2.0    0.318798
    4.0    0.176357
    5.0    0.114583
    1.0    0.039729
    Name: income_cat, dtype: float64




```python
housing["income_cat"].value_counts() / len(housing)
```




    3.0    0.350581
    2.0    0.318847
    4.0    0.176308
    5.0    0.114438
    1.0    0.039826
    Name: income_cat, dtype: float64



## 从数据探索和可视化中获得洞见


```python
# 创建一个副本
housing = start_train_set.copy()
```


```python
# 查看前5行
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>286600.0</td>
      <td>&lt;1H OCEAN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>340600.0</td>
      <td>&lt;1H OCEAN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>196900.0</td>
      <td>NEAR OCEAN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>46300.0</td>
      <td>INLAND</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>254500.0</td>
      <td>&lt;1H OCEAN</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



!!! tip "地理数据可视化"


```python
# 绘制散点图
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.6)
plt.show()
```


![png](output_32_0.png)



```python
import matplotlib.pyplot as plt
housing.plot(kind="scatter",x="longitude", y="latitude",alpha=0.4,
            s=housing["population"]/100, label="population",figsize=(20,7),
            c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,
            sharex=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27fb8255388>




![png](output_33_1.png)



```python
import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-47-b4b47ed4e188> in <module>
          1 import matplotlib.image as mpimg
    ----> 2 california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
          3 ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
          4                        s=housing['population']/100, label="Population",
          5                        c="median_house_value", cmap=plt.get_cmap("jet"),
    

    NameError: name 'PROJECT_ROOT_DIR' is not defined


!!! tip "寻找相关性"


```python
# corr() 方法 相关性
corr_matrix = housing.corr()
```


```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    income_cat            0.642274
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64




```python
# scatter_matrix 函数
from pandas.plotting import scatter_matrix
```


```python
attributes = ["median_house_value", "median_income", 
              "total_rooms","housing_median_age"]
```


```python
scatter_matrix(housing[attributes], figsize=(12, 8))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD520EC8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD542FC8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD82D2C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD85FC88>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD895A08>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD8CD788>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBD903508>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBED28288>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBED320C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBED677C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBEDCCD88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBEE05E88>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBEE3BF48>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBEE7D0C8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBEEB4208>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000027FBEEEC308>]],
          dtype=object)




![png](output_40_1.png)



```python
# save_fig is not defined
# plt.savefig 替代 save_fig
import pandas as pd
housing.plot(kind="scatter",
             x="median_income",
             y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.savefig("income_vs_house_value_scatterplot")
```


![png](output_41_0.png)



```python
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"]=housing["population"] / housing["households"]
```


```python
corr_matrix = housing.corr()
```


```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    income_cat                  0.642274
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64




```python
corr_matrix = housing.corr()
```


```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    income_cat                  0.642274
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64




```python
housing.plot(kind="scatter", 
            x="rooms_per_household",
            y="median_house_value",
           alpha=0.2,)
plt.axis([0, 5, 0, 520000])
```




    [0, 5, 0, 520000]




![png](output_47_1.png)



```python
housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>income_cat</th>
      <th>rooms_per_household</th>
      <th>bedrooms_per_room</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.575834</td>
      <td>35.639577</td>
      <td>28.653101</td>
      <td>2622.728319</td>
      <td>534.973890</td>
      <td>1419.790819</td>
      <td>497.060380</td>
      <td>3.875589</td>
      <td>206990.920724</td>
      <td>3.006541</td>
      <td>5.440341</td>
      <td>0.212878</td>
      <td>3.096437</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.001860</td>
      <td>2.138058</td>
      <td>12.574726</td>
      <td>2138.458419</td>
      <td>412.699041</td>
      <td>1115.686241</td>
      <td>375.720845</td>
      <td>1.904950</td>
      <td>115703.014830</td>
      <td>1.054602</td>
      <td>2.611712</td>
      <td>0.057379</td>
      <td>11.584826</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
      <td>1.000000</td>
      <td>1.130435</td>
      <td>0.100000</td>
      <td>0.692308</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.940000</td>
      <td>18.000000</td>
      <td>1443.000000</td>
      <td>295.000000</td>
      <td>784.000000</td>
      <td>279.000000</td>
      <td>2.566775</td>
      <td>119800.000000</td>
      <td>2.000000</td>
      <td>4.442040</td>
      <td>0.175304</td>
      <td>2.431287</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.510000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2119.500000</td>
      <td>433.000000</td>
      <td>1164.000000</td>
      <td>408.000000</td>
      <td>3.540900</td>
      <td>179500.000000</td>
      <td>3.000000</td>
      <td>5.232284</td>
      <td>0.203031</td>
      <td>2.817653</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.720000</td>
      <td>37.000000</td>
      <td>3141.000000</td>
      <td>644.000000</td>
      <td>1719.250000</td>
      <td>602.000000</td>
      <td>4.744475</td>
      <td>263900.000000</td>
      <td>4.000000</td>
      <td>6.056361</td>
      <td>0.239831</td>
      <td>3.281420</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6210.000000</td>
      <td>35682.000000</td>
      <td>5358.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
      <td>5.000000</td>
      <td>141.909091</td>
      <td>1.000000</td>
      <td>1243.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## 学习算法的数据准备


```python
housing = start_train_set.drop("median_house_value", axis=1)
```


```python
housing_labels = start_train_set["median_house_value"].copy()
```

!!! tip "数据清理"


```python
housing.dropna(subset=["total_bedrooms"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>&lt;1H OCEAN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>&lt;1H OCEAN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>NEAR OCEAN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>INLAND</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>&lt;1H OCEAN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>-118.13</td>
      <td>34.20</td>
      <td>46.0</td>
      <td>1271.0</td>
      <td>236.0</td>
      <td>573.0</td>
      <td>210.0</td>
      <td>4.9312</td>
      <td>INLAND</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>-117.56</td>
      <td>33.88</td>
      <td>40.0</td>
      <td>1196.0</td>
      <td>294.0</td>
      <td>1052.0</td>
      <td>258.0</td>
      <td>2.0682</td>
      <td>INLAND</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>-116.40</td>
      <td>34.09</td>
      <td>9.0</td>
      <td>4855.0</td>
      <td>872.0</td>
      <td>2098.0</td>
      <td>765.0</td>
      <td>3.2723</td>
      <td>INLAND</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>-118.01</td>
      <td>33.82</td>
      <td>31.0</td>
      <td>1960.0</td>
      <td>380.0</td>
      <td>1356.0</td>
      <td>356.0</td>
      <td>4.0625</td>
      <td>&lt;1H OCEAN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>-122.45</td>
      <td>37.77</td>
      <td>52.0</td>
      <td>3095.0</td>
      <td>682.0</td>
      <td>1269.0</td>
      <td>639.0</td>
      <td>3.5750</td>
      <td>NEAR BAY</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>16354 rows × 10 columns</p>
</div>




```python
housing.drop("total_bedrooms", axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>&lt;1H OCEAN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>&lt;1H OCEAN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>NEAR OCEAN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>INLAND</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>&lt;1H OCEAN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>-118.13</td>
      <td>34.20</td>
      <td>46.0</td>
      <td>1271.0</td>
      <td>573.0</td>
      <td>210.0</td>
      <td>4.9312</td>
      <td>INLAND</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>-117.56</td>
      <td>33.88</td>
      <td>40.0</td>
      <td>1196.0</td>
      <td>1052.0</td>
      <td>258.0</td>
      <td>2.0682</td>
      <td>INLAND</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>-116.40</td>
      <td>34.09</td>
      <td>9.0</td>
      <td>4855.0</td>
      <td>2098.0</td>
      <td>765.0</td>
      <td>3.2723</td>
      <td>INLAND</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>-118.01</td>
      <td>33.82</td>
      <td>31.0</td>
      <td>1960.0</td>
      <td>1356.0</td>
      <td>356.0</td>
      <td>4.0625</td>
      <td>&lt;1H OCEAN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>-122.45</td>
      <td>37.77</td>
      <td>52.0</td>
      <td>3095.0</td>
      <td>1269.0</td>
      <td>639.0</td>
      <td>3.5750</td>
      <td>NEAR BAY</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 9 columns</p>
</div>




```python
median = housing["total_bedrooms"].median()
```


```python
housing["total_bedrooms"].fillna(median)
```




    17606     351.0
    18632     108.0
    14650     471.0
    3230      371.0
    3555     1525.0
              ...  
    6563      236.0
    12053     294.0
    13908     872.0
    11159     380.0
    15775     682.0
    Name: total_bedrooms, Length: 16512, dtype: float64




```python
housing_num = housing.select_dtypes([np.number])
```


```python
housing_num
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>-118.13</td>
      <td>34.20</td>
      <td>46.0</td>
      <td>1271.0</td>
      <td>236.0</td>
      <td>573.0</td>
      <td>210.0</td>
      <td>4.9312</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>12053</th>
      <td>-117.56</td>
      <td>33.88</td>
      <td>40.0</td>
      <td>1196.0</td>
      <td>294.0</td>
      <td>1052.0</td>
      <td>258.0</td>
      <td>2.0682</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>13908</th>
      <td>-116.40</td>
      <td>34.09</td>
      <td>9.0</td>
      <td>4855.0</td>
      <td>872.0</td>
      <td>2098.0</td>
      <td>765.0</td>
      <td>3.2723</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>11159</th>
      <td>-118.01</td>
      <td>33.82</td>
      <td>31.0</td>
      <td>1960.0</td>
      <td>380.0</td>
      <td>1356.0</td>
      <td>356.0</td>
      <td>4.0625</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>15775</th>
      <td>-122.45</td>
      <td>37.77</td>
      <td>52.0</td>
      <td>3095.0</td>
      <td>682.0</td>
      <td>1269.0</td>
      <td>639.0</td>
      <td>3.5750</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 9 columns</p>
</div>




```python
# Warning: Since Scikit-Learn 0.20, the sklearn.preprocessing.Imputer class was replaced by the sklearn.impute.SimpleImputer class.
from sklearn.impute import SimpleImputer
```


```python
imputer = SimpleImputer(strategy="median")
```


```python
imputer
```




    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
                  missing_values=nan, strategy='median', verbose=0)




```python
# impute 官方示例
>>> import numpy as np
>>> from sklearn.impute import SimpleImputer
>>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
SimpleImputer()
>>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
>>> print(imp_mean.transform(X))
```

    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]
    


```python
from  sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```


```python
imputer.fit(housing_num)
```




    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
                  missing_values=nan, strategy='median', verbose=0)




```python
imputer.statistics_
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409,    3.    ])




```python
# 检查这是否与手动计算每个属性的中值相同
housing_num.median().values
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409,    3.    ])




```python
# Transform the training set
X = imputer.transform(housing_num)
```


```python
housing_tr = pd.DataFrame(X, 
                         columns=housing_num.columns,
                         index=housing.index)
```


```python
housing_tr.loc[12, "latitude"]
```




    37.85




```python
sample = housing_tr.iloc[10:15]
sample["housing_median_age"].sort_values(ascending=False)
```




    19684    41.0
    16365    24.0
    13956    21.0
    19234    18.0
    2390     12.0
    Name: housing_median_age, dtype: float64




```python
sample[ sample > 38]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16365</th>
      <td>NaN</td>
      <td>38.02</td>
      <td>NaN</td>
      <td>4157.0</td>
      <td>951.0</td>
      <td>2734.0</td>
      <td>879.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19684</th>
      <td>NaN</td>
      <td>39.14</td>
      <td>41.0</td>
      <td>2183.0</td>
      <td>559.0</td>
      <td>1202.0</td>
      <td>506.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19234</th>
      <td>NaN</td>
      <td>38.51</td>
      <td>NaN</td>
      <td>3364.0</td>
      <td>501.0</td>
      <td>1442.0</td>
      <td>506.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13956</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2520.0</td>
      <td>582.0</td>
      <td>416.0</td>
      <td>151.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2390</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2980.0</td>
      <td>495.0</td>
      <td>1184.0</td>
      <td>429.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
imputer.strategy
```




    'median'




```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)
```


```python
housing_tr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing_cat = housing.select_dtypes(exclude=[np.number])
```


```python
housing_cat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



!!! tip "处理文本和分类属性"


```python

```


```python

```


```python

```

!!! tip "自定义转化器"


```python

```


```python

```


```python

```

!!! tip "特征缩放"


```python

```


```python

```


```python

```

## 选择和训练模型


```python

```

## 微调模型


```python

```

## 网格搜索


```python

```

## 启动监控和维护系统


```python

```
