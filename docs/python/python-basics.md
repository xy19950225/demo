## **Python 速查表**



![](https://s1.ax1x.com/2020/09/09/w8CL8I.png)

## **数据结构**

!!! note "Python3 的 6 个标准数据类型" 

 	不可变（3 个）：数字、字符串、元组 ; 可变（3 个）：列表、字典、集合

### **字符串**

```Python
s1 = 'hello, world!'
s2 = "hello, world!"
# 以三个双引号或单引号开头的字符串可以换行
s3 = """
hello, 
world!
"""
print(s1, s2, s3, end='')
```

!!! note "字符串转义"

​    	在字符串中使用`\`（反斜杠）来表示转义

​		`\n`表示换行，`\t`表示制表符

   	 如果想在字符串中表示`'`要写成`\'`，同理想表示`\`要写成`\\`


```Python
s1 = '\'hello, world!\''
s2 = '\n\\hello, world!\\\n'
print(s1, s2, end='')
```

在`\`后面还可以跟一个八进制或者十六进制数来表示字符

例如`\141`和`\x61`都代表小写字母`a`，前者是八进制，后者是十六进制

也可以在`\`后面跟Unicode字符编码来表示字符，例如`\u83dc\u9e1f`代表的是中文“菜鸟”

```Python
s1 = '\141\142\143\x61\x62\x63'
s2 = '\u83dc\u9e1f'
print(s1, s2)
```

如果不希望字符串中的`\`表示转义，通过在字符串的最前面加上字母`r`加以说明

```Python
s1 = r'\'hello, world!\''
s2 = r'\n\\hello, world!\\\n'
print(s1, s2, end='')
```

!!! note "字符串运算符"

​	`+`运算符实现字符串的拼接

​	`*`运算符重复一个字符串的内容

​	`in`和`not in`判断一个字符串是否包含另外一个字符串（成员运算）

​	`[]`和`[:]`运算符从字符串取出某个字符或某些字符（切片运算）

```Python
s1 = 'hello ' * 3
print(s1) # hello hello hello 
s2 = 'world'
s1 += s2
print(s1) # hello hello hello world
print('ll' in s1) # True
print('good' in s1) # False
str2 = 'abc123456'
# 从字符串中取出指定位置的字符(索引)
print(str2[2]) # c
# 字符串切片(从指定的开始索引到指定的结束索引)
print(str2[2:5]) # c12
print(str2[2:]) # c123456
print(str2[2::2]) # c246
print(str2[::2]) # ac246
print(str2[::-1]) # 654321cba
print(str2[-3:-1]) # 45
```

!!! note "字符串方法"

```Python
str1 = 'hello, world!'
# 通过内置函数len计算字符串的长度
print(len(str1)) # 13
# 获得字符串首字母大写的拷贝
print(str1.capitalize()) # Hello, world!
# 获得字符串每个单词首字母大写的拷贝
print(str1.title()) # Hello, World!
# 获得字符串变大写后的拷贝
print(str1.upper()) # HELLO, WORLD!
# 从字符串中查找子串所在位置
print(str1.find('or')) # 8
print(str1.find('shit')) # -1
# 与find类似但找不到子串时会引发异常
# print(str1.index('or'))
# print(str1.index('shit'))
# 检查字符串是否以指定的字符串开头
print(str1.startswith('He')) # False
print(str1.startswith('hel')) # True
# 检查字符串是否以指定的字符串结尾
print(str1.endswith('!')) # True
# 将字符串以指定的宽度居中并在两侧填充指定的字符
print(str1.center(50, '*'))
# 将字符串以指定的宽度靠右放置左侧填充指定的字符
print(str1.rjust(50, ' '))
str2 = 'abc123456'
# 检查字符串是否由数字构成
print(str2.isdigit())  # False
# 检查字符串是否以字母构成
print(str2.isalpha())  # False
# 检查字符串是否以数字和字母构成
print(str2.isalnum())  # True
str3 = '  jackfrued@126.com '
print(str3)
# 获得字符串修剪左右两侧空格之后的拷贝
print(str3.strip())
```

!!! note "字符串格式化"

可以用下面的方式来格式化输出字符串

```Python
a, b = 5, 10
print('%d * %d = %d' % (a, b, a * b))
```

可以用字符串提供的方法来完成字符串的格式

```Python
a, b = 5, 10
print('{0} * {1} = {2}'.format(a, b, a * b))
```

Python 3.6 以后，格式化字符串还有更为简洁的书写方式，就是在字符串前加上字母`f`，可以使用下面的语法糖来简化上面的代码

```Python
a, b = 5, 10
print(f'{a} * {b} = {a * b}')
```

### **列表**

!!! question "字符串和数值的区别"

数值类型是标量类型，这种类型的对象没有可以访问的内部结构

字符串类型是一种结构化的、非标量类型，所以才会有一系列的属性和方法

列表（`list`），也是一种结构化的、非标量类型，它是值的有序序列，每个值都可以通过索引进行标识

定义列表可以将列表的元素放在`[]`中，多个元素用`,`进行分隔

使用`for`循环对列表元素进行遍历

使用`[]`或`[:]`运算符取出列表中的一个或多个元素

!!! note "创建列表、遍历列表以及列表索引"

```Python
list1 = [1, 3, 5, 7, 100]
print(list1) # [1, 3, 5, 7, 100]
# 乘号表示列表元素的重复
list2 = ['hello'] * 3
print(list2) # ['hello', 'hello', 'hello']
# 计算列表长度(元素个数)
print(len(list1)) # 5
# 下标(索引)运算
print(list1[0]) # 1
print(list1[4]) # 100
# print(list1[5])  # IndexError: list index out of range
print(list1[-1]) # 100
print(list1[-3]) # 5
list1[2] = 300
print(list1) # [1, 3, 300, 7, 100]
# 通过循环用下标遍历列表元素
for index in range(len(list1)):
    print(list1[index])
# 通过for循环遍历列表元素
for elem in list1:
    print(elem)
# 通过enumerate函数处理列表之后再遍历可以同时获得元素索引和值
for index, elem in enumerate(list1):
    print(index, elem)
```

!!! note "列表添加元素、移除元素"

```Python
list1 = [1, 3, 5, 7, 100]
# 添加元素
list1.append(200)
list1.insert(1, 400)
# 合并两个列表
# list1.extend([1000, 2000])
list1 += [1000, 2000]
print(list1) # [1, 400, 3, 5, 7, 100, 200, 1000, 2000]
print(len(list1)) # 9
# 先通过成员运算判断元素是否在列表中，如果存在就删除该元素
if 3 in list1:
	list1.remove(3)
if 1234 in list1:
    list1.remove(1234)
print(list1) # [1, 400, 5, 7, 100, 200, 1000, 2000]
# 从指定的位置删除元素
list1.pop(0)
list1.pop(len(list1) - 1)
print(list1) # [400, 5, 7, 100, 200, 1000]
# 清空列表元素
list1.clear()
print(list1) # []
```

!!! note "列表的切片"

```Python
fruits = ['grape', 'apple', 'strawberry', 'waxberry']
fruits += ['pitaya', 'pear', 'mango']
# 列表切片
fruits2 = fruits[1:4]
print(fruits2) # apple strawberry waxberry
# 可以通过完整切片操作来复制列表
fruits3 = fruits[:]
print(fruits3) # ['grape', 'apple', 'strawberry', 'waxberry', 'pitaya', 'pear', 'mango']
fruits4 = fruits[-3:-1]
print(fruits4) # ['pitaya', 'pear']
# 可以通过反向切片操作来获得倒转后的列表的拷贝
fruits5 = fruits[::-1]
print(fruits5) # ['mango', 'pear', 'pitaya', 'waxberry', 'strawberry', 'apple', 'grape']
```

!!! note "列表的排序"

```Python
list1 = ['orange', 'apple', 'zoo', 'internationalization', 'blueberry']
list2 = sorted(list1)
# sorted函数返回列表排序后的拷贝不会修改传入的列表
# 函数的设计就应该像sorted函数一样尽可能不产生副作用
list3 = sorted(list1, reverse=True)
# 通过key关键字参数指定根据字符串长度进行排序而不是默认的字母表顺序
list4 = sorted(list1, key=len)
print(list1)
print(list2)
print(list3)
print(list4)
# 给列表对象发出排序消息直接在列表对象上进行排序
list1.sort(reverse=True)
print(list1)
```

!!! note "列表推导式"

```Python
f = [x for x in range(1, 10)]
print(f)
f = [x + y for x in 'ABCDE' for y in '1234567']
print(f)
# 用列表推导式创建列表容器
# 用这种语法创建列表之后元素已经准备就绪所以需要耗费较多的内存空间
f = [x ** 2 for x in range(1, 1000)]
print(sys.getsizeof(f))  # 查看对象占用内存的字节数
print(f)
# 请注意下面的代码创建的不是一个列表而是一个生成器对象
# 通过生成器可以获取到数据但它不占用额外的空间存储数据
# 每次需要数据的时候就通过内部的运算得到数据(需要花费额外的时间)
f = (x ** 2 for x in range(1, 1000))
print(sys.getsizeof(f))  # 相比生成式生成器不占用存储数据的空间
print(f)
for val in f:
    print(val)
```



### **元组**

元组与列表类似，不同之处在于元组的元素不能修改

```Python
# 定义元组
t = ('菜鸟', 18, True, '四川成都')
print(t)
# 获取元组中的元素
print(t[0])
print(t[3])
# 遍历元组中的值
for member in t:
    print(member)
# 重新给元组赋值
# t[0] = '王大锤'  # TypeError
# 变量t重新引用了新的元组原来的元组将被垃圾回收
t = ('王大锤', 20, True, '云南昆明')
print(t)
# 将元组转换成列表
person = list(t)
print(person)
# 列表是可以修改它的元素的
person[0] = '李小龙'
person[1] = 25
print(person)
# 将列表转换成元组
fruits_list = ['apple', 'banana', 'orange']
fruits_tuple = tuple(fruits_list)
print(fruits_tuple)
```

!!! question "我们已经有了列表这种数据结构，为什么还需要元组呢？"

1. 元组中的元素是无法修改的，在项目中尤其是多线程环境中可能更喜欢使用的是那些不变对象（一方面因为对象状态不能修改，所以可以避免由此引起的不必要的程序错误，简单的说就是一个不变的对象要比可变的对象更加容易维护；另一方面因为没有任何一个线程能够修改不变对象的内部状态，一个不变对象自动就是线程安全的，这样就可以省掉处理同步化的开销。一个不变对象可以方便的被共享访问）。所以结论就是：如果不需要对元素进行添加、删除、修改的时候，可以考虑使用元组，当然如果一个方法要返回多个值，使用元组也是不错的选择
2. 元组在创建时间和占用的空间上面都优于列表



### **集合**

跟数学上的集合是一致的，不允许有重复元素，可以进行交集、并集、差集等运算

!!! note "创建集合"

```Python
# 创建集合的字面量语法
set1 = {1, 2, 3, 3, 3, 2}
print(set1)
print('Length =', len(set1))
# 创建集合的构造器语法
set2 = set(range(1, 10))
set3 = set((1, 2, 3, 3, 2, 1))
print(set2, set3)
# 创建集合的推导式语法(推导式也可以用于推导集合)
set4 = {num for num in range(1, 100) if num % 3 == 0 or num % 5 == 0}
print(set4)
```

!!! note "集合添加元素、删除元素"

```Python
set1.add(4)
set1.add(5)
set2.update([11, 12])
set2.discard(5)
if 4 in set2:
    set2.remove(4)
print(set1, set2)
print(set3.pop())
print(set3)
```

!!! note "集合的成员、交集、并集、差集等运算"

```Python
# 集合的交集、并集、差集、对称差运算
print(set1 & set2)
# print(set1.intersection(set2))
print(set1 | set2)
# print(set1.union(set2))
print(set1 - set2)
# print(set1.difference(set2))
print(set1 ^ set2)
# print(set1.symmetric_difference(set2))
# 判断子集和超集
print(set2 <= set1)
# print(set2.issubset(set1))
print(set3 <= set1)
# print(set3.issubset(set1))
print(set1 >= set2)
# print(set1.issuperset(set2))
print(set1 >= set3)
# print(set1.issuperset(set3))
```

> **说明：**集合进行运算的时候可以调用集合对象的方法，也可以直接使用对应的运算符，例如`&`运算符跟intersection方法的作用是一样的，但是使用运算符让代码更加直观



### **字典**

字典可以存储任意类型对象，它的每个元素都是由一个键和一个值组成的“键值对”

```Python
# 创建字典的字面量语法
scores = {'菜鸟': 95, '白元芳': 78, '狄仁杰': 82}
print(scores)
# 创建字典的构造器语法
items1 = dict(one=1, two=2, three=3, four=4)
# 通过zip函数将两个序列压成字典
items2 = dict(zip(['a', 'b', 'c'], '123'))
# 创建字典的推导式语法
items3 = {num: num ** 2 for num in range(1, 10)}
print(items1, items2, items3)
# 通过键可以获取字典中对应的值
print(scores['菜鸟'])
print(scores['狄仁杰'])
# 对字典中所有键值对进行遍历
for key in scores:
    print(f'{key}: {scores[key]}')
# 更新字典中的元素
scores['白元芳'] = 65
scores['诸葛王朗'] = 71
scores.update(冷面=67, 方启鹤=85)
print(scores)
if '武则天' in scores:
    print(scores['武则天'])
print(scores.get('武则天'))
# get方法也是通过键获取对应的值但是可以设置默认值
print(scores.get('武则天', 60))
# 删除字典中的元素
print(scores.popitem())
print(scores.popitem())
print(scores.pop('菜鸟', 100))
# 清空字典
scores.clear()
print(scores)
```



## **条件控制**



### **if 语句**



```
if condition_1:
    statement_block_1
elif condition_2:
    statement_block_2
else:
    statement_block_3
```



### **if 嵌套**



```
num = int(input("输入一个数字："))
if num % 2 == 0:
    if num % 3 == 0:
        print ("你输入的数字可以整除 2 和 3")
    else:
        print ("你输入的数字可以整除 2，但不能整除 3")
else:
    if num % 3 == 0:
        print ("你输入的数字可以整除 3，但不能整除 2")
    else:
        print ("你输入的数字不能整除 2 和 3")
```



## **循环语句**



在Python中构造循环结构有两种方法，一种是 `while` 循环，一种是 `for-in` 循环



### **while 循环**



```
# 猜数字游戏
import random

answer = random.randint(1, 100)
counter = 0
while True:
    counter += 1
    number = int(input('请输入: '))
    if number < answer:
        print('大一点')
    elif number > answer:
        print('小一点')
    else:
        print('恭喜你猜对了!')
        break
print("f 你总共猜了 {counter} 次")
if counter > 7:
    print('你的智商余额明显不足')
```



### **for-in 循环**



```
# 用for循环实现1~100之间的偶数求和

sum = 0
for x in range(2, 101, 2):
    sum += x
print(sum)
```

```
# for循环的嵌套
for i in range(1, 6):
    for j in range(1, i + 1):
        print("*", end="")
    print("\r")
        
*
**
***
****
*****
```

```
# 使用内置enumerate函数进行遍历
for index, item in enumerate(sequence):
    process(index, item)
```

```
sequence = [12, 34, 34, 23, 45, 76, 89]
```

```
for i, j in enumerate(sequence):
    print(i, j)
```

```
0 12
1 34
2 34
3 23
4 45
5 76
6 89
```



## **迭代器与生成器**

迭代器是实现了迭代器协议的对象。

Python中没有像`protocol`或`interface`这样的定义协议的关键字。

Python中用魔术方法表示协议。

`__iter__`和`__next__`魔术方法就是迭代器协议

生成器是语法简化版的迭代器。

生成器进化为协程。

生成器对象可以使用`send()`方法发送数据，发送的数据会成为生成器函数中通过`yield`表达式获得的值。这样，生成器就可以作为协程使用，协程简单的说就是可以相互协作的子程序。

```Python
class Fib(object):
    """迭代器"""
    
    def __init__(self, num):
        self.num = num
        self.a, self.b = 0, 1
        self.idx = 0
   
    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.num:
            self.a, self.b = self.b, self.a + self.b
            self.idx += 1
            return self.a
        raise StopIteration()
```

```Python
def fib(num):
    """生成器"""
    a, b = 0, 1
    for _ in range(num):
        a, b = b, a + b
        yield a
```

```Python
def calc_avg():
    """流式计算平均值"""
    total, counter = 0, 0
    avg_value = None
    while True:
        value = yield avg_value
        total, counter = total + value, counter + 1
        avg_value = total / counter


gen = calc_avg()
next(gen)
print(gen.send(10))
print(gen.send(20))
print(gen.send(30))
```



## **函数和模块**

先来研究一道数学题，请说出下面的方程有多少组正整数解

$$
x1 + x2 + x3 + x4 = 8
$$


上面的问题等同于将8个苹果分成四组，每组至少一个苹果有多少种方案

```Python
"""
输入M和N计算C(M,N)
"""
m = int(input('m = '))
n = int(input('n = '))
fm = 1
for num in range(1, m + 1):
    fm *= num
fn = 1
for num in range(1, n + 1):
    fn *= num
fm_n = 1
for num in range(1, m - n + 1):
    fm_n *= num
print(fm // fn // fm_n)
```



### **函数的作用**

*Martin Fowler* 曾经说过：“**代码有很多种坏味道，重复是最坏的一种！**”，要写出高质量的代码首先要解决的就是重复代码的问题

函数是组织好的，可重复使用的，用来实现单一，或相关联功能的代码段

函数能提高应用的模块性，和代码的重复利用率



### **定义函数**

使用`def`关键字来定义函数，在函数名后面的括号中可以放置传递给函数的参数，和数学上的函数非常相似

程序中函数的参数就相当于是数学上函数的自变量

函数执行完成后通过`return`关键字来返回一个值，相当于数学上函数的因变量

对上面的代码进行重构

```Python
"""
输入M和N计算C(M,N)
"""
def fac(num):
    """求阶乘"""
    result = 1
    for n in range(1, num + 1):
        result *= n
    return result


m = int(input('m = '))
n = int(input('n = '))
# 当需要计算阶乘的时候不用再写循环求阶乘而是直接调用已经定义好的函数
print(fac(m) // fac(n) // fac(m - n))
```

> **说明：** Python的`math`模块中其实已经有一个名为`factorial`函数实现了阶乘运算，事实上求阶乘并不用自己定义函数。**实际开发中并不建议做这种低级的重复劳动**



### **函数参数**

在Python中，函数的参数可以有默认值，也支持使用可变参数

```Python
from random import randint


def roll_dice(n=2):
    """摇色子"""
    total = 0
    for _ in range(n):
        total += randint(1, 6)
    return total


def add(a=0, b=0, c=0):
    """三个数相加"""
    return a + b + c


# 如果没有指定参数那么使用默认值摇两颗色子
print(roll_dice())
# 摇三颗色子
print(roll_dice(3))
print(add())
print(add(1))
print(add(1, 2))
print(add(1, 2, 3))
# 传递参数时可以不按照设定的顺序进行传递
print(add(c=50, a=100, b=200))
```

我们给上面两个函数的参数都设定了默认值，这也就意味着如果在调用函数的时候如果没有传入对应参数的值时将使用该参数的默认值，所以在上面的代码中我们可以用各种不同的方式去调用`add`函数，这跟其他很多语言中函数重载的效果是一致的。

其实上面的`add`函数还有更好的实现方案，因为我们可能会对0个或多个参数进行加法运算，而具体有多少个参数是由调用者来决定，我们作为函数的设计者对这一点是一无所知的，因此在不确定参数个数的时候，我们可以使用可变参数，代码如下所示。

```Python
# 在参数名前面的*表示args是一个可变参数
def add(*args):
    total = 0
    for val in args:
        total += val
    return total


# 在调用add函数时可以传入0个或多个参数
print(add())
print(add(1))
print(add(1, 2))
print(add(1, 2, 3))
print(add(1, 3, 5, 7, 9))
```



### **用模块管理函数**

从 Python 解释器退出再进入，定义的所有的方法和变量就都消失了

为此 Python 提供了一个办法，把这些定义存放在文件中，为一些脚本或者交互式的解释器实例使用，这个文件被称为模块

模块是一个包含所有你定义的函数和变量的文件，其后缀名是.py

模块可以被别的程序引入，以使用该模块中的函数等功能，这也是使用 python 标准库的方法



### **匿名函数**

python使用lambda来创建匿名函数

lambda只是一个表达式，函数体比def简单很多

lambda的主体是一个表达式，而不是一个代码块

lambda函数拥有自己的命名空间，且不能访问自己参数列表之外或全局命名空间里的参数

虽然lambda函数看起来只能写一行，却不等同于C或C++的内联函数，后者的目的是调用小函数时不占用栈内存从而增加运行效率



## **面向对象**

**类(Class):** 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。

**方法：**类中定义的函数。

**类变量：**类变量在整个实例化的对象中是公用的。类变量定义在类中且在函数体之外。类变量通常不作为实例变量使用。

**数据成员：**类变量或者实例变量用于处理类及其实例对象的相关的数据。

**方法重写：**如果从父类继承的方法不能满足子类的需求，可以对其进行改写，这个过程叫方法的覆盖（override），也称为方法的重写。

**局部变量：**定义在方法中的变量，只作用于当前实例的类。

**实例变量：**在类的声明中，属性是用变量来表示的，这种变量就称为实例变量，实例变量就是一个用 self 修饰的变量。

**继承：**即一个派生类（derived class）继承基类（base class）的字段和方法。继承也允许把一个派生类的对象作为一个基类对象对待。例如，有这样一个设计：一个Dog类型的对象派生自Animal类，这是模拟"是一个（is-a）"关系（例图，Dog是一个Animal）。

**实例化：**创建一个类的实例，类的具体对象。

**对象：**通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法。



## **命名空间**

一般有三种命名空间：

**内置名称（built-in names**）， Python 语言内置的名称，比如函数名 abs、char 和异常名称 BaseException、Exception 等等。

**全局名称（global names）**，模块中定义的名称，记录了模块的变量，包括函数、类、其它导入的模块、模块级的变量和常量。

**局部名称（local names）**，函数中定义的名称，记录了函数的变量，包括函数的参数和局部定义的变量。（类中定义的也是）

![img](https://www.runoob.com/wp-content/uploads/2014/05/types_namespace-1.png)

命名空间查找顺序: **局部的命名空间 -> 全局命名空间 -> 内置命名空间**

