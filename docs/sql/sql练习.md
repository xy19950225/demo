## **[简单](https://leetcode-cn.com/problemset/database/?difficulty=%E7%AE%80%E5%8D%95)**



## 176 第二高的薪水

### 题目描述

```
编写一个 SQL 查询，获取 Employee 表中第二高的薪水（Salary） 。

+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
例如上述 Employee 表，SQL查询应该返回 200 作为第二高的薪水。如果不存在第二高的薪水，那么查询应返回 null。

+---------------------+
| SecondHighestSalary |
+---------------------+
| 200                 |
+---------------------+
```

### 解题方法

```SQL
方法一：使用子查询和 LIMIT 子句    # 此方法可适用于求第N高的薪水，且数据越复杂，速度优势越明显
SELECT
    (SELECT DISTINCT salary 
    FROM employee
    ORDER BY salary DESC
    LIMIT 1,1) AS SecondHighestSalary;
    
方法二：使用 IFNULL 和 LIMIT 子句
SELECT
   IFNULL(
    (SELECT DISTINCT salary
    FROM employee
    ORDER BY salary DESC
    LIMIT 1,1),NULL)  AS SecondHighestSalary;
    
方法三：使用子查询和 MAX() 函数
SELECT MAX(salary) AS SecondHighestSalary
FROM employee
WHERE salary < 
    (SELECT MAX(salary)
    FROM employee);
```



## 596 超过5名学生的课

### 题目描述

```
有一个courses 表 ，有: student (学生) 和 class (课程)。

请列出所有超过或等于5名学生的课。

例如，表：

+---------+------------+
| student | class      |
+---------+------------+
| A       | Math       |
| B       | English    |
| C       | Math       |
| D       | Biology    |
| E       | Math       |
| F       | Computer   |
| G       | Math       |
| H       | Math       |
| I       | Math       |
+---------+------------+
应该输出:

+---------+
| class   |
+---------+
| Math    |
+---------+
 

提示：

学生在每个课中不应被重复计算。
```

### 解题方法

```SQL
方法：使用 GROUP BY 和 HAVING 条件
SELECT class
FROM courses
GROUP BY class
HAVING COUNT(DISTINCT student) >=5;
```



## 197 上升的温度



#### 题目描述

```
给定一个 Weather 表，编写一个 SQL 查询，来查找与之前（昨天的）日期相比温度更高的所有日期的 Id。

+---------+------------------+------------------+
| Id(INT) | RecordDate(DATE) | Temperature(INT) |
+---------+------------------+------------------+
|       1 |       2015-01-01 |               10 |
|       2 |       2015-01-02 |               25 |
|       3 |       2015-01-03 |               20 |
|       4 |       2015-01-04 |               30 |
+---------+------------------+------------------+
例如，根据上述给定的 Weather 表格，返回如下 Id:

+----+
| Id |
+----+
|  2 |
|  4 |
+----+
```

#### 解题方法

```SQL
方法一：使用 JOIN 和 DATEDIFF() 子句
SELECT a.id AS id
FROM weather a
JOIN weather b
ON DATEDIFF(a.recorddate,b.recorddate)=1
AND a.temperature > b.temperature;

方法二：使用 WHERE 和 DATEDIFF() 子句
SELECT a.id
FROM weather a, weather b 
WHERE DATEDIFF(a.recorddate,b.recorddate)=1 
    AND a.temperature > b.temperature;
```



## **[中等](https://leetcode-cn.com/problemset/database/?difficulty=%E4%B8%AD%E7%AD%89)**

## 177 第N高的薪水



#### 题目描述

```
编写一个 SQL 查询，获取 Employee 表中第 n 高的薪水（Salary）。

+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
例如上述 Employee 表，n = 2 时，应返回第二高的薪水 200。如果不存在第 n 高的薪水，那么查询应返回 null。

+------------------------+
| getNthHighestSalary(2) |
+------------------------+
| 200                    |
+------------------------+
```

#### 解题方法

```SQL
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    SET N = N-1;
  RETURN (
      # Write your MySQL query statement below.
      SELECT DISTINCT salary 
      FROM employee
      ORDER BY salary DESC
      LIMIT N,1
  );
END
```



## 184 部门工资最高的员工



#### 题目描述

```
Employee 表包含所有员工信息，每个员工有其对应的 Id, salary 和 department Id。

+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Jim   | 90000  | 1            |
| 3  | Henry | 80000  | 2            |
| 4  | Sam   | 60000  | 2            |
| 5  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
Department 表包含公司所有部门的信息。

+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+
编写一个 SQL 查询，找出每个部门工资最高的员工。对于上述表，您的 SQL 查询应返回以下行（行的顺序无关紧要）。

+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| IT         | Jim      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+
解释：

Max 和 Jim 在 IT 部门的工资都是最高的，Henry 在销售部的工资最高。
```

#### 解题方法

```SQL
# 我的错误写法
SELECT 
    d.name AS Department,
    e.name AS Employee,
    MAX(e.salary) AS Salary
FROM employee e 
JOIN department d 
ON e.departmentid=d.id
GROUP BY Department,Employee,Salary;

# 正确写法
方法：使用 JOIN 和 IN 语句
SELECT
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM employee e
JOIN department d ON e.departmentid = d.id
WHERE (e.departmentid , e.salary) IN
    (SELECT departmentid, MAX(salary)
    FROM employee
    GROUP BY departmentid
	)
;
```



## 180 连续出现的数字



#### 题目描述

```
编写一个 SQL 查询，查找所有至少连续出现三次的数字。

+----+-----+
| Id | Num |
+----+-----+
| 1  |  1  |
| 2  |  1  |
| 3  |  1  |
| 4  |  2  |
| 5  |  1  |
| 6  |  2  |
| 7  |  2  |
+----+-----+
例如，给定上面的 Logs 表， 1 是唯一连续出现至少三次的数字。

+-----------------+
| ConsecutiveNums |
+-----------------+
| 1               |
+-----------------+
```

#### 解题方法

```

```



## 178 分数排名



#### 题目描述

```
编写一个 SQL 查询来实现分数排名。

如果两个分数相同，则两个分数排名（Rank）相同。请注意，平分后的下一个名次应该是下一个连续的整数值。换句话说，名次之间不应该有“间隔”。

+----+-------+
| Id | Score |
+----+-------+
| 1  | 3.50  |
| 2  | 3.65  |
| 3  | 4.00  |
| 4  | 3.85  |
| 5  | 4.00  |
| 6  | 3.65  |
+----+-------+
例如，根据上述给定的 Scores 表，你的查询应该返回（按分数从高到低排列）：

+-------+------+
| Score | Rank |
+-------+------+
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 2    |
| 3.65  | 3    |
| 3.65  | 3    |
| 3.50  | 4    |
+-------+------+
重要提示：对于 MySQL 解决方案，如果要转义用作列名的保留字，可以在关键字之前和之后使用撇号。例如 `Rank`
```

#### 解题方法

```SQL
# 窗口函数 DENSE_RANK()
SELECT 
    score,
    DENSE_RANK() OVER(ORDER BY score DESC) 'Rank'
FROM scores; 

# 理解rank, dense_rank, row_number

# 窗口函数
```



## **[困难](https://leetcode-cn.com/problemset/database/?difficulty=%E5%9B%B0%E9%9A%BE)**

## 185. 部门工资前三高的所有员工



#### 题目描述

```

```

#### 解题方法

```

```

