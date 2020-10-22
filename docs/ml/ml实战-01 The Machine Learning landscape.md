!!! tip "人均GDP和生活满意度"


```python
# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

# 加载数据
oecd_bli = pd.read_csv("oecd_bli_2015.csv",thousands=",")
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands="," , sep="\t", encoding="latin1")

# 生活满意度数据
# 查看数据
oecd_bli.info()

# 查看前五行
oecd_bli.head()

oecd_bli["INEQUALITY"].value_counts()

# 筛选数据
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]

oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator",values="Value")

# 国家GDP数据
# 查看数据
gdp_per_capita.info()

# 查看前五行
gdp_per_capita.head()

# 将"2015"列名改为"GDP per capita"
gdp_per_capita.rename(columns={"2015":"GDP per capita"}, inplace=True)

# 将Country设为索引
gdp_per_capita.set_index("Country",inplace=True)

gdp_per_capita.head()

# 根据Country合并两表
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)

# 查看前三行
full_country_stats.head(3)

# GDP per capita和Life satisfaction
country_stats = full_country_stats[["GDP per capita","Life satisfaction"]]

# 根据GDP per capita排序
country_stats.sort_values(by="GDP per capita", ascending=False).head()

# 绘制散点图
plt.scatter(x=country_stats["GDP per capita"], y=country_stats["Life satisfaction"])

# 保存数据
country_stats.to_csv("country_stats111.csv")
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3292 entries, 0 to 3291
    Data columns (total 17 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   LOCATION               3292 non-null   object 
     1   Country                3292 non-null   object 
     2   INDICATOR              3292 non-null   object 
     3   Indicator              3292 non-null   object 
     4   MEASURE                3292 non-null   object 
     5   Measure                3292 non-null   object 
     6   INEQUALITY             3292 non-null   object 
     7   Inequality             3292 non-null   object 
     8   Unit Code              3292 non-null   object 
     9   Unit                   3292 non-null   object 
     10  PowerCode Code         3292 non-null   int64  
     11  PowerCode              3292 non-null   object 
     12  Reference Period Code  0 non-null      float64
     13  Reference Period       0 non-null      float64
     14  Value                  3292 non-null   float64
     15  Flag Codes             1120 non-null   object 
     16  Flags                  1120 non-null   object 
    dtypes: float64(3), int64(1), object(13)
    memory usage: 437.3+ KB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 190 entries, 0 to 189
    Data columns (total 7 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   Country                        190 non-null    object 
     1   Subject Descriptor             189 non-null    object 
     2   Units                          189 non-null    object 
     3   Scale                          189 non-null    object 
     4   Country/Series-specific Notes  188 non-null    object 
     5   2015                           187 non-null    float64
     6   Estimates Start After          188 non-null    float64
    dtypes: float64(2), object(5)
    memory usage: 10.5+ KB



![png](output_1_1.png)



```python

```
