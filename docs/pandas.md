## 导入 Pandas 与 NumPy


```python
import pandas as pd
```


```python
import numpy as np
```

## 生成对象


```python
# 用值列表生成 Series 时，Pandas 默认自动生成整数索引
s = pd.Series([1, 3, 5, np.nan, 6, 8])
```


```python
s
```




    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64




```python
# 用含日期时间索引与标签的 NumPy 数组生成 DataFrame
df = pd.DataFrame(np.random.randn(6, 4), 
                  index=pd.date_range("20200901", periods=6), 
                  columns=list("ABCD"))
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>-1.042130</td>
      <td>0.051650</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>-0.186728</td>
      <td>-1.005533</td>
      <td>1.239826</td>
      <td>1.058215</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用 Series 字典对象生成 DataFrame
df2 = pd.DataFrame({"A":1,
                   "B":pd.Timestamp("20200901"),
                   "C":pd.Series(1, index=list(range(4)), dtype="float32"),
                   "D":np.array([3] * 4, dtype="int32"),
                   "E":pd.Categorical(["test", "train", "test", "train"]),
                   "F":"foo"})
```


```python
df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2020-09-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2020-09-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2020-09-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2020-09-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# DataFrame的列有不同数据类型
df2.dtypes 
```




    A             int64
    B    datetime64[ns]
    C           float32
    D             int32
    E          category
    F            object
    dtype: object



## 查看数据

### 查看头部和尾部数据


```python
# 查看头部数据
df.head()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>-1.042130</td>
      <td>0.051650</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查看尾部数据
df.tail(3)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>-1.042130</td>
      <td>0.051650</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>-0.186728</td>
      <td>-1.005533</td>
      <td>1.239826</td>
      <td>1.058215</td>
    </tr>
  </tbody>
</table>
</div>



### 显示索引与列名


```python
# 显示索引
df.index
```




    DatetimeIndex(['2020-09-01', '2020-09-02', '2020-09-03', '2020-09-04',
                   '2020-09-05', '2020-09-06'],
                  dtype='datetime64[ns]', freq='D')




```python
# 显示列名
df.columns
```




    Index(['A', 'B', 'C', 'D'], dtype='object')



###  查看数据统计摘要


```python
df.describe()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.166543</td>
      <td>0.198184</td>
      <td>0.669769</td>
      <td>0.691578</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.524185</td>
      <td>1.089605</td>
      <td>1.495333</td>
      <td>0.840421</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.526610</td>
      <td>-1.131221</td>
      <td>-1.042130</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.128426</td>
      <td>-0.646852</td>
      <td>-0.451726</td>
      <td>0.130316</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.157166</td>
      <td>0.491035</td>
      <td>0.442533</td>
      <td>0.388393</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.368443</td>
      <td>0.591369</td>
      <td>1.962828</td>
      <td>0.896279</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.455894</td>
      <td>2.230473</td>
    </tr>
  </tbody>
</table>
</div>



### 转置数据


```python
df.T
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
      <th>2020-09-01</th>
      <th>2020-09-02</th>
      <th>2020-09-03</th>
      <th>2020-09-04</th>
      <th>2020-09-05</th>
      <th>2020-09-06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.267854</td>
      <td>0.401973</td>
      <td>-0.526610</td>
      <td>0.996290</td>
      <td>0.046479</td>
      <td>-0.186728</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-1.131221</td>
      <td>0.552876</td>
      <td>0.429193</td>
      <td>1.739588</td>
      <td>0.604200</td>
      <td>-1.005533</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.354760</td>
      <td>-0.484048</td>
      <td>2.455894</td>
      <td>2.203829</td>
      <td>-1.042130</td>
      <td>1.239826</td>
    </tr>
    <tr>
      <th>D</th>
      <td>2.230473</td>
      <td>0.410471</td>
      <td>0.366316</td>
      <td>0.032343</td>
      <td>0.051650</td>
      <td>1.058215</td>
    </tr>
  </tbody>
</table>
</div>



### 按轴排序


```python
df.sort_index(axis=1, ascending=False)
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
      <th>D</th>
      <th>C</th>
      <th>B</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>2.230473</td>
      <td>-0.354760</td>
      <td>-1.131221</td>
      <td>0.267854</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.410471</td>
      <td>-0.484048</td>
      <td>0.552876</td>
      <td>0.401973</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>0.366316</td>
      <td>2.455894</td>
      <td>0.429193</td>
      <td>-0.526610</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.032343</td>
      <td>2.203829</td>
      <td>1.739588</td>
      <td>0.996290</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.051650</td>
      <td>-1.042130</td>
      <td>0.604200</td>
      <td>0.046479</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>1.058215</td>
      <td>1.239826</td>
      <td>-1.005533</td>
      <td>-0.186728</td>
    </tr>
  </tbody>
</table>
</div>



### 按值排序


```python
df.sort_values(by="A", axis=0, ascending=False)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
    </tr>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>-1.042130</td>
      <td>0.051650</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>-0.186728</td>
      <td>-1.005533</td>
      <td>1.239826</td>
      <td>1.058215</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
    </tr>
  </tbody>
</table>
</div>



## 选择数据

### 获取数据


```python
df.A
```




    2020-09-01    0.267854
    2020-09-02    0.401973
    2020-09-03   -0.526610
    2020-09-04    0.996290
    2020-09-05    0.046479
    2020-09-06   -0.186728
    Freq: D, Name: A, dtype: float64




```python
# 选择单列，产生 Series，与 df.A 等效
df["A"]
```




    2020-09-01    0.267854
    2020-09-02    0.401973
    2020-09-03   -0.526610
    2020-09-04    0.996290
    2020-09-05    0.046479
    2020-09-06   -0.186728
    Freq: D, Name: A, dtype: float64




```python
# 用 [ ] 切片行
df[0:3]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
    </tr>
  </tbody>
</table>
</div>



### 按标签选择


```python
# 用标签提取一行数据
df.loc["20200901"]
```




    A    0.267854
    B   -1.131221
    C   -0.354760
    D    2.230473
    Name: 2020-09-01 00:00:00, dtype: float64




```python
# 用标签选择多列数据
df.loc[:, ["A","B"]]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>-0.186728</td>
      <td>-1.005533</td>
    </tr>
  </tbody>
</table>
</div>



### 按位置选择


```python
# 用整数位置选择
df.iloc[3]
```




    A    0.996290
    B    1.739588
    C    2.203829
    D    0.032343
    Name: 2020-09-04 00:00:00, dtype: float64




```python
# 类似 NumPy用整数切片
df.iloc[3:5, 0:2]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
    </tr>
  </tbody>
</table>
</div>



### 布尔索引


```python
# 用单列的值选择数据
df[df["A"] > 0]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>-1.042130</td>
      <td>0.051650</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选择 DataFrame里满足条件的值
df[df > 0]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.230473</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>NaN</td>
      <td>0.410471</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>NaN</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>NaN</td>
      <td>0.051650</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.239826</td>
      <td>1.058215</td>
    </tr>
  </tbody>
</table>
</div>



### 赋值


```python
# 用索引自动对齐新增列的数据
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20200901", periods=6))
```


```python
s1
```




    2020-09-01    1
    2020-09-02    2
    2020-09-03    3
    2020-09-04    4
    2020-09-05    5
    2020-09-06    6
    Freq: D, dtype: int64




```python
df["F"] = s1
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2020-09-05</th>
      <td>0.046479</td>
      <td>0.604200</td>
      <td>-1.042130</td>
      <td>0.051650</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2020-09-06</th>
      <td>-0.186728</td>
      <td>-1.005533</td>
      <td>1.239826</td>
      <td>1.058215</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## 缺失值


```python
# 重建索引（reindex）可以更改、添加、删除指定轴的索引，并返回数据副本，即不更改原数据
df1 = df.reindex(index=df.index[0:4], columns=list(df.columns) + ["E"])
```


```python
df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.loc[df.index[0]:df.index[1], "E"] = 1
```


```python
df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 删除缺失值


```python
df1.dropna()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 填充缺失值


```python
df1.fillna(0)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>0.267854</td>
      <td>-1.131221</td>
      <td>-0.354760</td>
      <td>2.230473</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>0.401973</td>
      <td>0.552876</td>
      <td>-0.484048</td>
      <td>0.410471</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>-0.526610</td>
      <td>0.429193</td>
      <td>2.455894</td>
      <td>0.366316</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-09-04</th>
      <td>0.996290</td>
      <td>1.739588</td>
      <td>2.203829</td>
      <td>0.032343</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 运算

### 统计


```python
# 描述性统计
df.mean()
```




    A    0.166543
    B    0.198184
    C    0.669769
    D    0.691578
    F    3.500000
    dtype: float64




```python
# Apply函数
df.apply(lambda x: x.max() - x.min())
```




    A    1.522901
    B    2.870809
    C    3.498024
    D    2.198130
    F    5.000000
    dtype: float64



## 合并（Merge）

### 结合（Concat）


```python
df1 = pd.DataFrame({"A": ["A0", "A1", "A2", "A3"],
                    "B": ["B0", "B1", "B2", "B3"],
                    "C": ["C0", "C1", "C2", "C3"],
                    "D": ["D0", "D1", "D2", "D3"]},
                    index=[0, 1, 2, 3])
```


```python
df2 = pd.DataFrame({"A": ["A4", "A5", "A6", "A7"],
                    "B": ["B4", "B5", "B6", "B7"],
                    "C": ["C4", "C5", "C6", "C7"],
                    "D": ["D4", "D5", "D6", "D7"]},
                    index=[4, 5, 6, 7])
```


```python
df3 = pd.DataFrame({"A": ["A8", "A9", "A10", "A11"],
                    "B": ["B8", "B9", "B10", "B11"],
                    "C": ["C8", "C9", "C10", "C11"],
                    "D": ["D8", "D9", "D10", "D11"]},
                    index=[8, 9, 10, 11])
```


```python
frame = pd.concat([df1, df2, df3])
```


```python
frame
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>



### 连接（Join）


```python
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
```


```python
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
```


```python
left
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
      <th>key</th>
      <th>lval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
right
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
      <th>key</th>
      <th>rval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(left, right, on="key")
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
      <th>key</th>
      <th>lval</th>
      <th>rval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>foo</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>foo</td>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### 追加（Append）


```python
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.926318</td>
      <td>-1.110969</td>
      <td>0.217602</td>
      <td>-0.094516</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.349012</td>
      <td>1.385598</td>
      <td>-0.172996</td>
      <td>-0.790864</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.417921</td>
      <td>-0.004624</td>
      <td>-0.509224</td>
      <td>-1.550437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.491223</td>
      <td>-0.642262</td>
      <td>0.175682</td>
      <td>0.721095</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.058397</td>
      <td>0.283183</td>
      <td>-0.497554</td>
      <td>0.739755</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.712672</td>
      <td>0.628955</td>
      <td>-0.472966</td>
      <td>0.318093</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.279831</td>
      <td>0.074890</td>
      <td>-1.236430</td>
      <td>0.592133</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.132221</td>
      <td>-0.676778</td>
      <td>-0.701541</td>
      <td>-1.210815</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = df.loc[3]
```


```python
s
```




    A   -0.491223
    B   -0.642262
    C    0.175682
    D    0.721095
    Name: 3, dtype: float64




```python
df.append(s, ignore_index=True)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.926318</td>
      <td>-1.110969</td>
      <td>0.217602</td>
      <td>-0.094516</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.349012</td>
      <td>1.385598</td>
      <td>-0.172996</td>
      <td>-0.790864</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.417921</td>
      <td>-0.004624</td>
      <td>-0.509224</td>
      <td>-1.550437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.491223</td>
      <td>-0.642262</td>
      <td>0.175682</td>
      <td>0.721095</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.058397</td>
      <td>0.283183</td>
      <td>-0.497554</td>
      <td>0.739755</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.712672</td>
      <td>0.628955</td>
      <td>-0.472966</td>
      <td>0.318093</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.279831</td>
      <td>0.074890</td>
      <td>-1.236430</td>
      <td>0.592133</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.132221</td>
      <td>-0.676778</td>
      <td>-0.701541</td>
      <td>-1.210815</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.491223</td>
      <td>-0.642262</td>
      <td>0.175682</td>
      <td>0.721095</td>
    </tr>
  </tbody>
</table>
</div>



## 分组（Group by）


```python
df = pd.DataFrame([("bird", "Falconiformes", 389.0),
                    ("bird", "Psittaciformes", 24.0),
                    ("mammal", "Carnivora", 80.2),
                    ("mammal", "Primates", np.nan),
                    ("mammal", "Carnivora", 58)],
                   index=["falcon", "parrot", "lion", "monkey", "leopard"],
                   columns=("class", "order", "max_speed"))
```


```python
df
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
      <th>class</th>
      <th>order</th>
      <th>max_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>falcon</th>
      <td>bird</td>
      <td>Falconiformes</td>
      <td>389.0</td>
    </tr>
    <tr>
      <th>parrot</th>
      <td>bird</td>
      <td>Psittaciformes</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>lion</th>
      <td>mammal</td>
      <td>Carnivora</td>
      <td>80.2</td>
    </tr>
    <tr>
      <th>monkey</th>
      <td>mammal</td>
      <td>Primates</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>leopard</th>
      <td>mammal</td>
      <td>Carnivora</td>
      <td>58.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(by=["class", "order"]).sum()
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
      <th></th>
      <th>max_speed</th>
    </tr>
    <tr>
      <th>class</th>
      <th>order</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bird</th>
      <th>Falconiformes</th>
      <td>389.0</td>
    </tr>
    <tr>
      <th>Psittaciformes</th>
      <td>24.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mammal</th>
      <th>Carnivora</th>
      <td>138.2</td>
    </tr>
    <tr>
      <th>Primates</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 重塑（Reshaping）

### 堆叠（Stack）

![image.png](https://pandas.pydata.org/pandas-docs/stable/_images/reshaping_stack.png)


```python
tuples = list(zip(*[["bar", "bar", "baz", "baz",
                    "foo", "foo", "qux", "qux"],
                    ["one", "two", "one", "two",
                    "one", "two", "one", "two"]]))
```


```python
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
```


```python
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
```


```python
df
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>1.457984</td>
      <td>0.864189</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.808289</td>
      <td>1.558983</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>1.388420</td>
      <td>-0.746224</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.215613</td>
      <td>1.050263</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">foo</th>
      <th>one</th>
      <td>-0.536742</td>
      <td>0.383083</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.751930</td>
      <td>-1.717304</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">qux</th>
      <th>one</th>
      <td>0.929121</td>
      <td>-1.075260</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.238485</td>
      <td>-1.581831</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df[:4]
```


```python
df2
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>1.457984</td>
      <td>0.864189</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.808289</td>
      <td>1.558983</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>1.388420</td>
      <td>-0.746224</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.215613</td>
      <td>1.050263</td>
    </tr>
  </tbody>
</table>
</div>




```python
stacked = df2.stack()
```


```python
stacked
```




    first  second   
    bar    one     A    1.457984
                   B    0.864189
           two     A    1.808289
                   B    1.558983
    baz    one     A    1.388420
                   B   -0.746224
           two     A    0.215613
                   B    1.050263
    dtype: float64




```python
stacked.unstack()
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>1.457984</td>
      <td>0.864189</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.808289</td>
      <td>1.558983</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>1.388420</td>
      <td>-0.746224</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.215613</td>
      <td>1.050263</td>
    </tr>
  </tbody>
</table>
</div>



## 数据透视表（Pivot Tables）


```python
import datetime
```


```python
df = pd.DataFrame({"A": ["one", "one", "two", "three"] * 3,
                   "B": ["A", "B", "C"] * 4,
                   "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
                   "D": np.random.randn(12),
                   "E": np.random.randn(12),
                   "F":[datetime.datetime(2020, i, 1) for i in range(1, 7)]
                    + [datetime.datetime(2020, i, 15) for i in range(1, 7)]})
```


```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>A</td>
      <td>foo</td>
      <td>0.613701</td>
      <td>0.145033</td>
      <td>2020-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>B</td>
      <td>foo</td>
      <td>-1.501934</td>
      <td>-2.349162</td>
      <td>2020-02-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>two</td>
      <td>C</td>
      <td>foo</td>
      <td>-0.626282</td>
      <td>1.619653</td>
      <td>2020-03-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>three</td>
      <td>A</td>
      <td>bar</td>
      <td>0.281824</td>
      <td>-0.355017</td>
      <td>2020-04-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>B</td>
      <td>bar</td>
      <td>0.777277</td>
      <td>0.141533</td>
      <td>2020-05-01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>one</td>
      <td>C</td>
      <td>bar</td>
      <td>0.788832</td>
      <td>1.354495</td>
      <td>2020-06-01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>A</td>
      <td>foo</td>
      <td>-0.182135</td>
      <td>-0.177494</td>
      <td>2020-01-15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>three</td>
      <td>B</td>
      <td>foo</td>
      <td>1.635379</td>
      <td>-0.504184</td>
      <td>2020-02-15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>one</td>
      <td>C</td>
      <td>foo</td>
      <td>1.287765</td>
      <td>0.642662</td>
      <td>2020-03-15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>one</td>
      <td>A</td>
      <td>bar</td>
      <td>1.534828</td>
      <td>0.089097</td>
      <td>2020-04-15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>two</td>
      <td>B</td>
      <td>bar</td>
      <td>0.109630</td>
      <td>0.508206</td>
      <td>2020-05-15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>three</td>
      <td>C</td>
      <td>bar</td>
      <td>-0.543666</td>
      <td>-0.779394</td>
      <td>2020-06-15</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])
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
      <th>C</th>
      <th>bar</th>
      <th>foo</th>
    </tr>
    <tr>
      <th>A</th>
      <th>B</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>A</th>
      <td>1.534828</td>
      <td>0.613701</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.777277</td>
      <td>-1.501934</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.788832</td>
      <td>1.287765</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">three</th>
      <th>A</th>
      <td>0.281824</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>B</th>
      <td>NaN</td>
      <td>1.635379</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.543666</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">two</th>
      <th>A</th>
      <td>NaN</td>
      <td>-0.182135</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.109630</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C</th>
      <td>NaN</td>
      <td>-0.626282</td>
    </tr>
  </tbody>
</table>
</div>



## 时间序列(TimeSeries)


```python
rng = pd.date_range("1/1/2020", periods=10, freq="M")
```


```python
rng
```




    DatetimeIndex(['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30',
                   '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31',
                   '2020-09-30', '2020-10-31'],
                  dtype='datetime64[ns]', freq='M')



## 类别型（Categoricals）


```python
df = pd.DataFrame({"A": ["a", "b", "c", "a"]})
```


```python
df
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
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>




```python
 df["B"] = df["A"].astype("category")
```


```python
df
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>


