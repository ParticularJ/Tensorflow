# -*- coding:utf-8 -*-
from sys import argv
from os.path import exists

script, from_file, to_file = argv
open(to_file,'w').write(open(from_file).read())

#print "Copying from %s to %s" % (from_file, to_file)

#We could do these weo on one line too, how?
#打开对象
#in_file = open(from_file)
#读取文件
#indata = in_file.read()
#统计字节
#print "The input file is %d bytes long" % len(indata)

#print "Does the output file exist? %r" % exists(to_file)
#print "Ready, hit RETURN to continue, ctrl+c to abort."
#raw_input()

#out_file = open(to_file,'w')
#out_file.write(indata)

#print "Alright, all done."

#out_file.close()
#in_file.close()
