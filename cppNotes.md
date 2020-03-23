C++代码规范

# 代码规范

[TOC]

-----------------------------

##  文件
- 文件名**全部采用下划线方式**，如message_process.h
- 一般每个.cc或.cpp都有对应同名的.h文件，单元测试代码和main函数文件例外。

##  头文件
所有头文件都应该使用 `#define` 防止头文件被多重包含, 命名格式当是: `<PROJECT>_<PATH>_<FILE>_H_`

比如项目foo中，有文件foo/src/bar/baz.h，宏定义为
``` 
#ifndef FOO_BAR_BAZ_H_
#define FOO_BAR_BAZ_H_

...

#endif  // FOO_BAR_BAZ_H_
```

##  函数参数顺序
定义函数时, 参数顺序依次为: 输入参数, 然后是输出参数.

C/C++ 函数参数分为输入参数, 输出参数, 和输入/输出参数三种. 输入参数一般传值或传 `const` 引用, 输出参数或输入/输出参数则是非 const 指针或引用. **即使是新加的只输入参数也要放在输出参数之前**.


##  `#include` 的路径及顺序
项目内头文件应按照项目源代码目录树结构排列, 避免使用 UNIX 特殊的快捷目录`.`或`..` 例如, google-awesome-project/src/base/logging.h 应该按如下方式包含:
``` 
#include "base/logging.h"
```
包含顺序

1. 本.cpp或.cc对应的.h头文件
2. C 系统文件
3. C++ 系统文件
4. 其他库的 .h 文件
5. 本项目内 .h 文件

##  命名

###  类型
包括类、结构体、枚举类型，命名全部采用**头字母大写驼峰**方式。比如类`MyClass`, 结构体`MyStruct`, 枚举`MyEnum`

###  函数
除构造、析构函数外，其他函数名全部采用**下划线小写**，比如 `void my_function();`

###  变量

**不使用匈牙利命名法**

全部采用**下划线小写**方式。类成员变量名前面加`m_`标记，比如`m_variable`，全局变量名前面加`g_`标记，比如`g_variable`，其他无需加前缀标记。

###  枚举值、常量及宏定义
全部采用**下划线大写**方式，如
``` 
#define MAX_NUM 100

enum MyEnum
{
    MY_ENUM_SUCC    = 0,
    MY_ENUM_FAIL    = 1
};

const double NUM = 100;
```
##  空格及换行
**只使用空格**, 不要使用tab符！每次缩进 4 个空格.

大括号`{`全部单独一行，如
``` 
class MyClass
{
};

if (...)
{
    // some code
}
else
{
    // some code
}
```
条件语句后加一空格，左右括号与语句间不加空格，各语句间加一空格，如
``` 
if (a > 0 && b < 10)
{
    return foo;
}
else
{
    return bar;
}

for (int i = 0, j = argc; i < argc; i++, j--)
{
}
```
变量赋值可适当加空格对齐，如
```
int         age     = 0;
string      name    = "yosolin";
Setting*    setting = new Setting();
```
##  注释
行注释使用//，块注释使用/**/

行注释//后接一空格，如
```
int age = 0;    // the person's age
```
###  函数声明注释
每个函数原则上都需要有函数声明注释，需要注释的内容包含：

- 函数使用简介
- 函数具体说明（可选）
- 输入参数说明
- 输出参数说明
- 返回值说明

格式如下：
```
/**
 * save data to database
 *
 * after all processes done, you should call this.
 *
 * @param [in] age the person's age
 * @param [in] name the person's name
 * @param [out] idx index of the record in database
 * @return 0 on success, other on failure
 */
int save_data(int age, const string& name, string& idx);
```
##  文件编码
使用UTF-8

##  其他

指针*，引用&紧随类型，如
```
int* a; 
const string& str;
```

常量与变量相等比较判断，常量放前面，如
```
if (0 == a)
{
    ...
}
```

所有条件判断都要有大括号，无论是否只有一条语句，如
```
if (a > 0)
{
    do_something();
}
```

数值类型使用uint16_t或int16_t及以上，不要使用uint8_t或int8_t，即使数值大小不超过128
```
uint16_t type = 10;
```

