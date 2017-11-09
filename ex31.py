# _*_ coding:utf-8 _*_

def test_yield(n):
    for i in range(n):
        yield i*2

for j in test_yield(8):
    print j,",",
print "结束理解yield"
#利用yield输出斐波那契数列
#############

def fab(max):
    a,b = 0, 1
    while a < max:
        yield a
       # print "a = %d" % a
        a, b = b, a+b
        print "b = %d" % b
print "斐波那契数列"
for i in fab(20):
    print i,",",
