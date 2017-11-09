# -*- coding: utf-8 -*-

formatter = "%r %r %r %r"

print formatter % (1,2,3,4)
print formatter % ("one","tow",'three','four')
print formatter % (False,True,True,False)
print formatter % (formatter,formatter,formatter,formatter)
print formatter % ("I love","play","basketball","!")

