# -*- coding:utf-8 -*-

# 理解is
e = 1
es = 1.0
ess = 1

print u"""is就是比对id的值，看是否指向同一对象，
这里需要指出：同一对象，不是值相等"""

print id(e)
print id(es)
print id(ess)
print e is es
print e is ess

#理解lambda
g = lambda:"lambda test."
print g()
num1 = lambda x,y=1:x+y
def num2(x,y=1):
    return x+y
print num1(1)
print num1(10,10)
print num2(10,10)
